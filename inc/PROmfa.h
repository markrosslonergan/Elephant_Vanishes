#ifndef PROMFA_H_
#define PROMFA_H_

//PROfit
#include "PROlog.h"
#include "PROconfig.h"
#include "PROspec.h"
#include "PROtocall.h"
#include "PROcreate.h"

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

namespace PROfit{

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

            std::vector<int> unflatten(int index);
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
                    std::cout << "index n, 3+m, i, j, k, vals: " << k << ", " << dom_dim+b << ", " << unflat_dim[0] << ", " << unflat_dim[1] << ", " << unflat_dim[2] << ", "<<vals.data(k,b)<<std::endl;
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


    /* Function: Actually filles and encodes the MFA model. 
    */
    inline void makeSignalModel(diy::mpi::communicator world, Block<double>* b, const diy::Master::ProxyWithLink& cp, int nbins, int mfadim, std::vector<int> nctrl_pts, int deg);


    /* Function: Actually filles and encodes the MFA model. 
    */
    int runMFA();






};
#endif
