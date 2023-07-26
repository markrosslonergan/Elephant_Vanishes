#include "PROconfig.h"
#include "PROspec.h"
#include "PROcovariancegen.h"

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
    vector<float>      min;                        // minimum corner of domain
    vector<float>      max;                        // maximum corner of domain
    vector<float>      s;                          // scaling factor for each variable or any other usage
    float              r;                          // x-y rotation of domain or any other usage
    vector<float>      f;                          // frequency multiplier for each variable or any other usage
    float              t;                          // waviness of domain edges or any other usage
    float              n;                          // noise factor [0.0 - 1.0]
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


    // read a floating point 2d scalar dataset from HDF5
    // reads masses for geometry dimension 0 from same HDF5 file
    // assigns integer values for the geometry dimension 1 from 0 to n_pts - 1
    // f = (mass, y, value)
    //template <typename V>               // type of science value being read
    void read_3d_data(
            const       diy::Master::ProxyWithLink& cp,
            MFAInfo&    mfa_info,
            DomainArgs& args,
            Eigen::Tensor<double, 4> vals,
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
        assert(vals(0) == ndom_pts(0));
        assert(vals(1) == ndom_pts(1));
        assert(vals(2) == ndom_pts(2));
        // set geometry values
        int n = 0;
        int pd = mfa_info.pt_dim();
        for (size_t k = 0; k < (size_t)(ndom_pts(2)); k++) {
            for (size_t j = 0; j < (size_t)(ndom_pts(1)); j++) {
                for (size_t i = 0; i < (size_t)(ndom_pts(0)); i++) {
                    input->domain(n, 0) = i;
                    input->domain(n, 1) = j;
                    input->domain(n, 2) = k;
                    for (int m = 0; m < pd-dom_dim; m++) {
                        //std::cout << "index n, 3+m, i, j, k, vals: " << n << ", " << 3+m << ", " << i << ", " << j << ", " << k << ", ";
                        input->domain(n, dom_dim+m) = vals(m, i, j, k);
                        //std::cout << vals(m, i, j, k) << std::endl;

                    }	
                    n++;
                }
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

        //std::cout << "tot_ndom_pt, input->domain(tot_ndom_pts - 1, 1), this->dom_dim = " << tot_ndom_pts << ", " << input->domain(tot_ndom_pts - 1, 1) << ", " << this->dom_dim << std::endl;

        // debug
        //cerr << "domain extent:\n min\n" << this->bounds_mins << "\nmax\n" << this->bounds_maxs << endl;
    }

};


inline void makeSignalModel(diy::mpi::communicator world, Block<Real_t>* b, const diy::Master::ProxyWithLink& cp, int nbins, int mfadim, std::vector<int> nctrl_pts, int deg)
{
    // default command line arguments
    int    dom_dim      = mfadim;                    // dimension of domain (<= pt_dim)
    int    pt_dim       = nbins+dom_dim;        // dimension of input points
    Real_t noise        = 0.0;                  // fraction of noise

    vector<int> v_degrees(dom_dim, deg);
    vector<int> v_nctrls = nctrl_pts;

    // Info classes for construction of MFA
    ModelInfo geom_info(dom_dim);
    ModelInfo var_info(dom_dim, pt_dim - dom_dim, v_degrees, v_nctrls);
    MFAInfo   mfa_info(dom_dim, 1, geom_info, var_info);
    mfa_info.weighted = 0;
    
    // set input domain arguments
    DomainArgs d_args(dom_dim, pt_dim);
    d_args.n            = noise;
    d_args.min = {0, 0, 0};
    d_args.max = {v_nctrls[0] - 1, v_nctrls[1] - 1, v_nctrls[2] - 1};

    for (int i = 0; i < d_args.ndom_pts.size(); i++)
            d_args.ndom_pts[i] = v_nctrls[i];   

    Eigen::VectorXd vec_gridx(v_nctrls[0]);
    Eigen::VectorXd vec_gridy(v_nctrls[1]);
    Eigen::VectorXd vec_gridz(v_nctrls[2]);

    //tmp
    int signalgridsize = 10;
    int mass = v_nctrls[0];
    int dim2 = v_nctrls[1];
    int dim3 = v_nctrls[2];

    Eigen::Tensor<double, 4> map_bin_to_grid(nbins,mass,dim2,dim3);
    Eigen::ArrayXXd values(signalgridsize,nbins);	

    int gridx = -1;
    int gridy = -1;
    int gridz = -1;


    double t0 = MPI_Wtime();
    for (size_t i=0; i<signalgridsize; ++i) {
        double t_a = MPI_Wtime();
        values.row(i) = 10; //signal.predict3D(world, cp, i, false, gridx, gridy, gridz);
        double t_b = MPI_Wtime();
        int gridz_index = i / (dim2 * mass);
        int gridy_index = (i / mass) % dim2;
        int gridx_index = i % mass;

        //std::cout << "x, y, z: " << gridx_index << ", " << gridy_index << ", " << gridz_index << std::endl;
        vec_gridz(gridz_index) = gridz;
        vec_gridy(gridy_index) = gridy;
        vec_gridx(gridx_index) = gridx;

        double t_c = MPI_Wtime();
        for( int bin=0; bin < nbins; bin++ ) map_bin_to_grid(bin,gridx_index,gridy_index,gridz_index) = values(i,bin); 
        double t_d = MPI_Wtime();
    }
    double t1 = MPI_Wtime();
    std::cout << "time to create the tensor: " << t1-t0 << std::endl;
    std::cout << "read 3d data" << std::endl;
    double t2 = MPI_Wtime();
    b->read_3d_data(cp, mfa_info, d_args, map_bin_to_grid, false);
    double t3 = MPI_Wtime();
    std::cout << "took: " << t3-t1 << " seconds" << std::endl;
    std::cout << "fixed encode block" << std::endl;
    double t4 = MPI_Wtime();
    b->fixed_encode_block(cp, mfa_info);
    double t5 = MPI_Wtime();
    std::cout << "took: " << t5-t4 << " seconds" << std::endl;
    std::cout << "range error" << std::endl;
    double t6 = MPI_Wtime();
    b->range_error(cp, 0, true, true);
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

    diy::mpi::environment env(argc, argv);
    diy::mpi::communicator world;

    std::cout<<"World Rank: "<<world.rank()<<std::endl;
    size_t blocks = world.size();
    if (world.rank()==0) std::cout<<"We have blocks: "<< blocks<<std::endl;

    int mfadim = 3; //Dimension of the MFA model
    int nBins = 1; //Dimension of output of MFA? I believe so
    int degree = 2;//Science degree? Not sure
    int mass = 10;//number of mass pts
    int dim2 = 10;//number of dim2 pts?
    int dim3 = 1;
    std::vector<int> ncontrol_pts = {mass,dim2,dim3};




    Bounds<float> fc_domain(mfadim);
    for(int i=0; i<mfadim; i++){
        fc_domain.min[i] = 0.;
        fc_domain.max[i] = blocks-1;
    }

    diy::FileStorage               storage("./DIY.XXXXXX");
    diy::RoundRobinAssigner        fc_assigner(world.size(), blocks);
    diy::RegularDecomposer<Bounds<float>> fc_decomposer(mfadim, fc_domain, blocks);
    diy::RegularBroadcastPartners  fc_comm(    fc_decomposer, mfadim, true);
    diy::RegularMergePartners      fc_partners(fc_decomposer, mfadim, true);
    diy::Master                    fc_master(world, 1, -1, &Block<float>::create, &Block<float>::destroy, &storage, &Block<float>::save, &Block<float>::load);
    diy::ContiguousAssigner   assigner(world.size(), blocks);

        fc_decomposer.decompose(world.rank(),
            assigner,
            [&](int gid, const Bounds<Real_t>& core, const Bounds<Real_t>& bounds, const Bounds<Real_t>& domain, const RCLink<Real_t>& link)
            { Block<Real_t>::add(gid, core, bounds, domain, link, fc_master, mfadim, nBins+mfadim, 0.0); });


        double T10 = MPI_Wtime();
        fc_master.foreach([world,nBins, mfadim, ncontrol_pts, degree](Block<Real_t>* b, const diy::Master::ProxyWithLink& cp){
                makeSignalModel(world, b, cp,  nBins, mfadim, ncontrol_pts, degree);});

        double T11   = MPI_Wtime();
        if (world.rank()==0) std::cout << "time to build model: " << T11-T10 << " seconds." << std::endl;




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

