#include "PROconfig.h"
#include "PROspec.h"

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
        size_t nBins;
        std::vector<int> dimensions;

        Eigen::ArrayXXd data;
        size_t mfadim;

        MFAvalues(int signalgridsize, int inbins, std::vector<int> dims) : size(signalgridsize), nBins(inbins), dimensions(dims){
            data.resize(size,nBins);	
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
            std::reverse(coordinates.begin(), coordinates.end());
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


    //Some core inputs
    int mfadim = 1; //Dimension of the MFA model (input)
    int nBins = 20; //Dimension of output of MFA? I believe so 
    int degree = 2;//Science degree? Not sure... polynomial degree?
    int pt_dim       = nBins+mfadim;        // dimension of input points

    std::vector<int> ncontrol_pts(mfadim); // mfadim each has dim 7 for +/-1,2,3 sigma and CV
    std::fill(ncontrol_pts.begin(), ncontrol_pts.end(), 7);


    vector<int> v_degrees(mfadim, degree);


    std::cout<<"start to construct ModelInfo"<<std::endl;
    // Info classes for construction of MFA
    // TODO find out about this a bit more
    ModelInfo geom_info(mfadim);
    ModelInfo var_info(mfadim, pt_dim - mfadim, v_degrees, ncontrol_pts);
    MFAInfo   mfa_info(mfadim, 1, geom_info, var_info);
    mfa_info.weighted = 0;

    //How many TOTAL control points
    int signalgridsize = 1;
    for(int i =0; i< mfadim; i++){
        signalgridsize *= ncontrol_pts[i];
    }
    MFAvalues values(signalgridsize, nBins, ncontrol_pts);	

    std::cout<<"Starting signalgridsize "<<std::endl;
    double t0 = MPI_Wtime();
    
    for (size_t i=0; i<signalgridsize; ++i) {
        Eigen::VectorXd res(nBins);
        res.setRandom();
        values.data.row(i) = res;
    }

    double t1 = MPI_Wtime();
    std::cout << "time to create the tensor: " << t1-t0 << std::endl;
    std::cout << "read 3d data" << std::endl;
    double t2 = MPI_Wtime();



    // Set points per direction and compute total points in domain
    VectorXi ndom_pts(mfadim);
    for (int i = 0; i < mfadim; i++)
        ndom_pts(i)     =  ncontrol_pts[i];

    // Create input data set and add to block
    std::cout << "mfadim, mdims, ndom_pts.prod(), ndom_pts: " << mfadim << ", " << mfa_info.model_dims() << ", " << ndom_pts.prod() << ", " << ndom_pts << std::endl;

    mfa::PointSet<double> input(mfadim, mfa_info.model_dims(), ndom_pts.prod(), ndom_pts);
    //mfa::PointSet<double> * input = new mfa::PointSet<double>(mfadim, mfa_info.model_dims(), ndom_pts.prod(), ndom_pts);

    for(size_t k = 0; k< values.size; k++){

        //For this point, whats the dimensions it correlated to?
        std::vector<int> unflat_dim = values.unflatten(k);

        //First mfadim size points are the grid indicies
        for(int t =0; t< mfadim; t++){
            input.domain(k,t) = unflat_dim.at(t);
        }
        //Next Nbins is the values of the data
        for(size_t b=0; b<values.nBins; b++){
            input.domain(k,mfadim+b) = values.data(k,b); 
            //std::cout << "index n, 3+m, i, j, k, vals: " << k << ", " << mfadim+b << ", " << unflat_dim[0] << ", " << unflat_dim[1] << ", " << unflat_dim[2] << ", "<<values.data(k,b)<<std::endl;
        }

    }

    // Init params from input data (must fill input->domain first)
    std::cout<<"Starting to init_params"<<std::endl;
    input.init_params();   

    // Construct the MFA object
    std::cout<<"Starting to construct_MFA"<<std::endl;
    //this->setup_MFA(cp, mfa_info);
    //mfa::MFA<double> * tmfa = new mfa::MFA<double>(mfa_info);
    mfa::MFA<double>  tmfa(mfa_info);
    double t3 = MPI_Wtime();
    std::cout << "took: " << t3-t1 << " seconds" << std::endl;
    std::cout << "fixed encode block" << std::endl;
    double t4 = MPI_Wtime();
    //b->fixed_encode_block(cp, mfa_info);
    tmfa.FixedEncode(input, mfa_info.regularization, mfa_info.reg1and2, mfa_info.weighted, false);
    double t5 = MPI_Wtime();
    std::cout << "took: " << t5-t4 << " seconds" << std::endl;
    //std::cout << "range error" << std::endl;
    double t6 = MPI_Wtime();
    //b->range_error(cp, 0, true, true);
    double t7 = MPI_Wtime();
    //std::cout << "took: " << t7-t6 << " seconds" << std::endl;
    //std::cout << "print block" << std::endl;
    double t8 = MPI_Wtime();
    //b->print_block(cp, 0);
    double t9 = MPI_Wtime();
    //std::cout << "took: " << t9-t8 << " seconds" << std::endl;
    std::cout << "done with makeSignalModel" << std::endl;


    //And decoding

    Eigen::VectorXd out_pt(pt_dim);
    Eigen::VectorXd in_param(mfadim);
    in_param.setConstant(0.4512412);

    std::cout<<" Inside: "<<pt_dim<<" "<<mfadim<<std::endl;

    double t10 = MPI_Wtime();
    // b->decode_point(*cp, in_param, out_pt);
    tmfa.Decode(in_param, out_pt);
    double t11 = MPI_Wtime();

    std::cout << "time to decode a single point: " << t11-t10 <<" seconds? "<< std::endl;
    std::cout<<"Final Res: \n "<<out_pt<<std::endl;               


    std::cout<<" OK, things are setup. "<<std::endl;

    return 0;
}

