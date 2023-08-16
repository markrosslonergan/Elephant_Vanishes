#include "PROconfig.h"
#include "PROspec.h"
#include "PROmfa.h"

#include <algorithm>
#include <limits>

#include "CLI11.h"

#define FMT_HEADER_ONLY

using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;





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


   //runMFA(env, world);
  

    std::cout<<"World Rank: "<<world.rank()<<std::endl;
    size_t blocks = world.size();
    if (world.rank()==0) std::cout<<"We have blocks: "<< blocks<<std::endl;

    //Some core inputs
    int mfadim = 3; //Dimension of the MFA model (input)
    int nBins = 20; //Dimension of output of MFA? I believe so 
    int degree = 2;//Science degree? Not sure... polynomial degree?
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
            PROfit::makeSignalModel(world, b, cp,  nBins, mfadim, ncontrol_pts, degree);});

    double T11   = MPI_Wtime();
    if (world.rank()==0) std::cout << "time to build model: " << T11-T10 << " seconds." << std::endl;


    //And decoding
    std::cout<<"Testing Decodin"<<std::endl;
    diy_master.foreach([&](Block<double>* b, const diy::Master::ProxyWithLink& cp){
            Eigen::VectorXd out_pt(b->pt_dim);
            Eigen::VectorXd in_param(b->dom_dim);
            in_param.setConstant(0.4512412);

            std::cout<<" Inside: "<<b->pt_dim<<" "<<b->dom_dim<<std::endl;

            double t1 = MPI_Wtime();
            b->decode_point(cp, in_param, out_pt);
            double t0 = MPI_Wtime();

            std::cout << "time to decode a single point: " << t0-t1 <<" seconds? "<< std::endl;
            std::cout<<"Final Res: \n "<<out_pt<<std::endl;               
            });


    std::cout<<" OK, things are setup. "<<std::endl;




}
