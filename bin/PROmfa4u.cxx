#include "PROconfig.h"
#include "PROspec.h"
#include "PROcovariancegen.h"

#include <algorithm>
#include <limits>

#include "CLI11.h"
#include "LBFGSB.h"

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/mpi.hpp>
#include <diy/serialization.hpp>
#include <diy/partners/broadcast.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/io/block.hpp>

#include <mfa/mfa.hpp>
#include <mfa/block_base.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/NumericalDiff>

#define FMT_HEADER_ONLY
//#include <fmt/format.h>


using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;



class MFAvalues
{

    public:
        size_t size;
        size_t nbins;
        std::vector<int> dimensions;
        
        Eigen::ArrayXXd data;
        size_t mfadim;

        MFAvalues(int signalgridsize, int inbins, std::vector<int> dims) : size(signalgridsize), nbins(inbins), dimensions(dims){
            data.resize(size,nbins);	
            mfadim = dimensions.size();
        }

        std::vector<int> unflatten(int index) {
            std::vector<int> coordinates(dimensions.size());
            int product = 1;

            for (int i = dimensions.size() - 1; i >= 0; --i) {
                coordinates[i] = (index / product) % dimensions[i];
                product *= dimensions[i];
            }

            for(auto &c : coordinates){
                if( c!=c || !std::isfinite(c)){
                std::cout<<"BLARG "<<c<<std::endl;
                }
            }

            return coordinates;
        }
};




class ChiTest
{
    private:
        int n;
    public:
        ChiTest(int n_) : n(n_) {}
        double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &grad)
        {
            double fx = 0.0;
            for(int i = 0; i < n; i += 2)
            {
                double t1 = 1.0 - x[i];
                double t2 = 10 * (x[i + 1] - x[i] * x[i]);
                grad[i + 1] = 20 * t2;
                grad[i]     = -2.0 * (x[i] * grad[i + 1] + t1);
                fx += t1 * t1 + t2 * t2;
            }
            return fx;
        }
};

// arguments to block foreach functions
struct DomainArgs
{
    DomainArgs(int dom_dim, int nvars) 
    {
        tot_ndom_pts = 0;
        starts.resize(dom_dim);
        ndom_pts.resize(dom_dim);
        full_dom_pts.resize(dom_dim);
        min.resize(dom_dim);
        max.resize(dom_dim);
        s.resize(nvars);
        f.resize(nvars);
        for (auto i = 0; i < nvars; i++)
        {
            s[i] = 1.0;
            f[i] = 1.0;
        }
        r = 0;
        t = 0;
        n = 0;
        multiblock = false;
        structured = true;   // Assume structured input by default
        rand_seed  = -1;
    }
    size_t              tot_ndom_pts;
    vector<int>         starts;                     // starting offsets of ndom_pts (optional, usually assumed 0)
    vector<int>         ndom_pts;                   // number of points in domain (possibly a subset of full domain)
    vector<int>         full_dom_pts;               // number of points in full domain in case a subset is taken
    vector<double>      min;                        // minimum corner of domain
    vector<double>      max;                        // maximum corner of domain
    vector<double>      s;                          // scaling factor for each variable or any other usage
    double              r;                          // x-y rotation of domain or any other usage
    vector<double>      f;                          // frequency multiplier for each variable or any other usage
    double              t;                          // waviness of domain edges or any other usage
    double              n;                          // noise factor [0.0 - 1.0]
    string              infile;                     // input filename
    bool                multiblock;                 // multiblock domain, get bounds from block
    bool                structured;                 // input data lies on unstructured grid
    int                 rand_seed;                  // seed for generating random data. -1: no randomization, 0: choose seed at random
};


// block
template <typename T>
struct Block : public BlockBase<T>
{
    using Base = BlockBase<T>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core_mins;
    using Base::core_maxs;
    using Base::bounds_mins;
    using Base::bounds_maxs;
    using Base::overlaps;
    using Base::input;

    static
        void* create()              { return mfa::create<Block>(); }

    static
        void destroy(void* b)       { mfa::destroy<Block>(b); }

    static
        void add(                                   // add the block to the decomposition
                int                 gid,                // block global id
                const Bounds<T>&    core,               // block bounds without any ghost added
                const Bounds<T>&    bounds,             // block bounds including any ghost region added
                const Bounds<T>&    domain,             // global data bounds
                const RCLink<T>&    link,               // neighborhood
                diy::Master&        master,             // diy master
                int                 dom_dim,            // domain dimensionality
                int                 pt_dim,             // point dimensionality
                T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
        {
            mfa::add<Block, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
        }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<Block, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<Block, T>(b_, bb); }


    // read a doubleing point 2d scalar dataset from HDF5
    // reads masses for geometry dimension 0 from same HDF5 file
    // assigns integer values for the geometry dimension 1 from 0 to n_pts - 1
    // f = (mass, y, value)
    //template <typename V>               // type of science value being read
    void read_3d_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args,
            MFAvalues& vals,
            bool  rescale)            // rescale science values
    {
        //std::cout << "$$$$ dom_dim: " << a->dom_dim << std::endl; 
        DomainArgs* a = &args;

        const int nvars       = mfa_info.nvars();
        const VectorXi mdims  = mfa_info.model_dims();

        // Resize the vectors that record error metrics
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);

        // Set points per direction and compute total points in domain
        VectorXi ndom_pts(dom_dim);
        for (int i = 0; i < dom_dim; i++)
            ndom_pts(i)     =  a->ndom_pts[i];

        // Create input data set and add to block
        std::cout << "dom_dim, mdims, ndom_pts.prod(), ndom_pts: " << dom_dim << ", " << mdims << ", " << ndom_pts.prod() << ", " << ndom_pts << std::endl;
        input = new mfa::PointSet<T>(dom_dim, mdims, ndom_pts.prod(), ndom_pts);
        //std::cout<<" vals(0): \n "<<vals(0)<<" \n ndom_pts(0): \n "<<ndom_pts(0)<<std::endl;
        //std::cout<<vals(1)<<" "<<ndom_pts(1)<<std::endl;
        //std::cout<<vals(2)<<" "<<ndom_pts(2)<<std::endl;
        //assert(vals(0) == ndom_pts(0));
        //assert(vals(1) == ndom_pts(1));
        //assert(vals(2) == ndom_pts(2));
        // set geometry values

        for(size_t k = 0; k< vals.size; k++){
            
            //For this point, whats the dimensions it correlated to?
            std::vector<int> unflat_dim = vals.unflatten(k);

            //First dom_dim size points are the grid indicies
            for(int t =0; t< dom_dim; t++){
                input->domain(k,t) = unflat_dim.at(t);
            }
            //Next Nbins is the values of the data
            for(size_t b=0; b<vals.nbins; b++){
                input->domain(k,dom_dim+b) = vals.data(k,b); 
            }
        }

        // Init params from input data (must fill input->domain first)
        std::cout<<"Starting to init_params"<<std::endl;
        input->init_params();   

        // Construct the MFA object
        std::cout<<"Starting to construct_MFA"<<std::endl;
        this->setup_MFA(cp, mfa_info);

        // find extent of masses, values, and science variables (bins)
        // NOTE: the difference between core_mins/maxs and bounds_mins/maxs only matters
        //       if DIY is used to partition the domain space, but we set them appropriately anyway
        bounds_mins.resize(pt_dim);
        bounds_maxs.resize(pt_dim);
        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        bounds_mins = input->domain.colwise().minCoeff();
        bounds_maxs = input->domain.colwise().maxCoeff();
        core_mins = bounds_mins.head(dom_dim);
        core_maxs = bounds_maxs.head(dom_dim);

        //std::cout<<"Core Mins: "<<core_mins<<std::endl;
        //std::cout<<"Core Maxs: "<<core_maxs<<std::endl;
        //std::cout<<"Bound Mins: "<<bounds_mins<<std::endl;
        //std::cout<<"Bound Maxs: "<<bounds_maxs<<std::endl;
        // debug
        //cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

};


inline void makeSignalModel(diy::mpi::communicator world, Block<double>* b, const diy::Master::ProxyWithLink& cp, int nbins, int mfadim, std::vector<int> nctrl_pts, int deg)
{
    // default command line arguments
    int    dom_dim      = mfadim;               // dimension of domain (<= pt_dim)
    int    pt_dim       = nbins+dom_dim;        // dimension of input points
    double noise        = 0.0;                  // fraction of noise

    vector<int> v_degrees(dom_dim, deg);
    vector<int> v_nctrls = nctrl_pts;

    std::cout<<"start to construct ModelInfo"<<std::endl;
    // Info classes for construction of MFA
    ModelInfo geom_info(dom_dim);
    ModelInfo var_info(dom_dim, pt_dim - dom_dim, v_degrees, v_nctrls);
    MFAInfo   mfa_info(dom_dim, 1, geom_info, var_info);
    mfa_info.weighted = 0;

    std::cout<<"Settting DomainArgs "<<v_nctrls[0]-1<<" "<<v_nctrls[1]-1<<" "<<v_nctrls[2]-1<<std::endl;
    // set input domain arguments
    DomainArgs d_args(dom_dim, pt_dim);
    d_args.n   = noise;
    std::fill(d_args.min.begin(), d_args.min.end(), 0);
    for(int i =0; i< dom_dim; i++){
        d_args.max[i] = v_nctrls[i] - 1;
    }
   
    for(int i = 0; i < d_args.ndom_pts.size(); i++){
        d_args.ndom_pts[i] = v_nctrls[i];   
    }


    //How many TOTAL control points
    int signalgridsize = 1;
    for(int i =0; i< dom_dim; i++){
        signalgridsize *= nctrl_pts[i];
    }

    MFAvalues values(signalgridsize, nbins, nctrl_pts);	


    std::cout<<"Starting signalgridsize "<<std::endl;
    double t0 = MPI_Wtime();
    for (size_t i=0; i<signalgridsize; ++i) {
        double t_a = MPI_Wtime();
        Eigen::VectorXd res(nbins);
        res.setRandom();
        values.data.row(i) = res;//signal.predict3D(world, cp, i, false, gridx, gridy, gridz);

        //values.grid_indicies[i] =  {i % mass, (i / mass) % dim2, i / (dim2 * mass)};
        //map_bin_to_grid[bin][gridx_index][gridy_index][gridz_index] = values(i,bin); 
        double t_d = MPI_Wtime();
    }

    double t1 = MPI_Wtime();
    std::cout << "time to create the tensor: " << t1-t0 << std::endl;
    std::cout << "read 3d data" << std::endl;
    double t2 = MPI_Wtime();
    b->read_3d_data(cp, mfa_info, d_args, values, false);
    double t3 = MPI_Wtime();
    std::cout << "took: " << t3-t1 << " seconds" << std::endl;
    std::cout << "fixed encode block" << std::endl;
    double t4 = MPI_Wtime();
    b->fixed_encode_block(cp, mfa_info);
    double t5 = MPI_Wtime();
    std::cout << "took: " << t5-t4 << " seconds" << std::endl;
    //std::cout << "range error" << std::endl;
    double t6 = MPI_Wtime();
    //b->range_error(cp, 0, true, true);
    double t7 = MPI_Wtime();
    std::cout << "took: " << t7-t6 << " seconds" << std::endl;
    std::cout << "print block" << std::endl;
    double t8 = MPI_Wtime();
    b->print_block(cp, 0);
    double t9 = MPI_Wtime();
    std::cout << "took: " << t9-t8 << " seconds" << std::endl;
    std::cout << "done with makeSignalModel" << std::endl;

}








int main(int argc, char* argv[])
{


    CLI::App app{"PROfit: EpEm MFA4u"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    int maxevents = 100;

    //doubles
    app.add_option("-x,--xml", xmlname, "Input PROfit XML config.");
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");
    CLI11_PARSE(app, argc, argv);


    //Some DIY MPI setup. 
    diy::mpi::environment env(argc, argv);
    diy::mpi::communicator world;

    std::cout<<"World Rank: "<<world.rank()<<std::endl;
    size_t blocks = world.size();
    if (world.rank()==0) std::cout<<"We have blocks: "<< blocks<<std::endl;

    //Some core inputs
    int mfadim = 3; //Dimension of the MFA model (input)
    int nBins = 1020; //Dimension of output of MFA? I believe so 
    int degree = 3;//Science degree? Not sure... polynomial degree?
    std::vector<int> ncontrol_pts(mfadim); // mfadim each has dim 7 for +/-1,2,3 sigma and CV
    std::fill(ncontrol_pts.begin(), ncontrol_pts.end(), 7);

    Bounds<double> domain(mfadim);
    for(int i=0; i<mfadim; i++){
        domain.min[i] = 0.;
        domain.max[i] = blocks-1;
        std::cout<<i<<" domain "<<domain.min[i]<<" "<<domain.max[i]<<std::endl;
    }

    diy::FileStorage               diy_storage("./DIY.XXXXXX");
    diy::RoundRobinAssigner        diy_assigner(world.size(), blocks);
    diy::RegularDecomposer<Bounds<double>> diy_decomposer(mfadim, domain, blocks);
    diy::RegularBroadcastPartners  diy_comm(    diy_decomposer, mfadim, true);
    diy::RegularMergePartners      diy_partners(diy_decomposer, mfadim, true);
    diy::Master                    diy_master(world, 1, -1, &Block<double>::create, &Block<double>::destroy, &diy_storage, &Block<double>::save, &Block<double>::load);
    diy::ContiguousAssigner   assigner(world.size(), blocks);

    //What is a decomposer here eh, I think its assigning blocks to the MPI nodes
    std::cout<<"diy_decomposer.decompose"<<std::endl;
    diy_decomposer.decompose(world.rank(),
            assigner,
            [&](int gid, const Bounds<double>& core, const Bounds<double>& bounds, const Bounds<double>& domain, const RCLink<double>& link)
            { Block<double>::add(gid, core, bounds, domain, link, diy_master, mfadim, nBins+mfadim, 0.0); });


    // This is whats building the SignalModel
    std::cout<<"diy_master.foreach"<<std::endl;
    double T10 = MPI_Wtime();
    diy_master.foreach([world,nBins, mfadim, ncontrol_pts, degree](Block<double>* b, const diy::Master::ProxyWithLink& cp){
            makeSignalModel(world, b, cp,  nBins, mfadim, ncontrol_pts, degree);});

    double T11   = MPI_Wtime();
    if (world.rank()==0) std::cout << "time to build model: " << T11-T10 << " seconds." << std::endl;


    //And decoding
    std::cout<<"Testing Decodin"<<std::endl;
    diy_master.foreach([&](Block<double>* b, const diy::Master::ProxyWithLink& cp){
            Eigen::VectorXd out_pt(b->pt_dim);
            Eigen::VectorXd in_param(b->dom_dim);
            in_param.setConstant(0.25);
            
            std::cout<<" Inside: "<<b->pt_dim<<" "<<b->dom_dim<<std::endl;

            double t1 = MPI_Wtime();
            b->decode_point(cp, in_param, out_pt);
            double t0 = MPI_Wtime();
           
            std::cout << "time to decode a single point: " << t0-t1 <<" seconds? "<< std::endl;
            std::cout<<"Final Res: \n "<<out_pt<<std::endl;               
            });


    std::cout<<" OK, things are setup. "<<std::endl;

    PROconfig myConf(xmlname);

    //PROspec mySpec(myConf);
    //TH1D hmm = mySpec.toTH1D(myConf);

    std::cout<<"Minimizer test"<<std::endl;

    LBFGSpp::LBFGSBParam<double> param;  
    param.epsilon = 1e-6;
    param.max_iterations = 100;
    LBFGSpp::LBFGSBSolver<double> solver(param); 

    int n=78;
    ChiTest fun(n);

    // Bounds
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, 0.0);
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

    // Initial guess
    Eigen::VectorXd x = Eigen::VectorXd::Constant(n, 2.0);


    // x will be overwritten to be the best point found
    double fx;
    int niter = solver.minimize(fun, x, fx, lb, ub);


    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;

    return 0;
}

