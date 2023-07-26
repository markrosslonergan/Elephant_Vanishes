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

#include <vector>
#include <iostream>
#include <cmath>
#include <string>

#include "/home/mark/work/PROfit/build/mfa-prefix/src/mfa/examples/block.hpp"

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



    //mfa-prefix/src/mfa/examples/fixed/fixed.cpp
    // initialize MPI
    diy::mpi::environment  env(argc, argv);     // equivalent of MPI_Init(argc, argv)/MPI_Finalize()
    diy::mpi::communicator world;               // equivalent of MPI_COMM_WORLD

    int nblocks     = 1;                        // number of local blocks
    int tot_blocks  = nblocks * world.size();   // number of global blocks
    int mem_blocks  = -1;                       // everything in core for now
    int num_threads = 1;                        // needed in order to do timing

    // default command line arguments, right now keep em
    int    pt_dim       = 3;                    // dimension of input points
    int    dom_dim      = 2;                    // dimension of domain (<= pt_dim)
    int    geom_degree  = 1;                    // degree for geometry (same for all dims)
    int    vars_degree  = 4;                    // degree for science variables (same for all dims)
    int    ndomp        = 100;                  // input number of domain points (same for all dims)
    int    ntest        = 0;                    // number of input test points in each dim for analytical error tests
    int    geom_nctrl   = -1;                   // input number of control points for geometry (same for all dims)
    int    vars_nctrl   = 11;                   // input number of control points for all science variables (same for all dims)
    string input        = "sinc";               // input dataset
    int    weighted     = 1;                    // solve for and use weights (bool 0/1)
    Real_t rot          = 0.0;                  // rotation angle in degrees
    Real_t twist        = 0.0;                  // twist (waviness) of domain (0.0-1.0)
    Real_t noise        = 0.0;                  // fraction of noise
    int    error        = 1;                    // decode all input points and check error (bool 0/1)
    string infile;                              // input file name
    int    structured   = 1;                    // input data format (bool 0/1)
    int    rand_seed    = -1;                   // seed to use for random data generation (-1 == no randomization)
    float  regularization = 0;                  // smoothing parameter for models with non-uniform input density (0 == no smoothing)
    int    fixed_ray    = 0;
    int     reg1and2 = 0;                       // flag for regularizer: 0 --> regularize only 2nd derivs. 1 --> regularize 1st and 2nd
    bool   help         = false;                // show help

    // initialize DIY
    diy::FileStorage          storage("./DIY.XXXXXX"); // used for blocks to be moved out of core
    diy::Master               master(world,
                                     num_threads,
                                     mem_blocks,
                                     &Block<Real_t>::create,
                                     &Block<Real_t>::destroy,
                                     &storage,
                                     &Block<Real_t>::save,
                                     &Block<Real_t>::load);
    diy::ContiguousAssigner   assigner(world.size(), tot_blocks);

    // even though this is a single-block example, we want diy to do a proper decomposition with a link
    // so that everything works downstream (reading file with links, e.g.)
    // therefore, set some dummy global domain bounds and decompose the domain
    Bounds<Real_t> dom_bounds(dom_dim);
    for (int i = 0; i < dom_dim; ++i)
    {
        dom_bounds.min[i] = 0.0;
        dom_bounds.max[i] = 1.0;
    }
    Decomposer<Real_t> decomposer(dom_dim, dom_bounds, tot_blocks);
    decomposer.decompose(world.rank(),
                         assigner,
                         [&](int gid, const Bounds<Real_t>& core, const Bounds<Real_t>& bounds, const Bounds<Real_t>& domain, const RCLink<Real_t>& link)
                         { Block<Real_t>::add(gid, core, bounds, domain, link, master, dom_dim, pt_dim, 0.0); });

    // set MFA arguments
    ModelInfo   geom_info(dom_dim, dom_dim);
    ModelInfo   var_info(dom_dim, 1, vector<int>(dom_dim, vars_degree), vector<int>(dom_dim, vars_nctrl));
    vector<ModelInfo> all_vars_info(pt_dim - dom_dim, var_info);
    MFAInfo     mfa_info(dom_dim, 1, geom_info, all_vars_info);
    mfa_info.weighted       = weighted;
    mfa_info.regularization = regularization;
    mfa_info.reg1and2       = reg1and2;

    // set default args for diy foreach callback functions
    DomainArgs d_args(dom_dim, pt_dim);
    d_args.n            = noise;
    d_args.t            = twist;
    d_args.multiblock   = false;
    d_args.structured   = structured;
    d_args.rand_seed    = rand_seed;
    d_args.ndom_pts     = vector<int>(dom_dim, ndomp);
    
    // initialize input data




     return 0;
}

