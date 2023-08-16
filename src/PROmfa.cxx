#include "PROmfa.h"

namespace PROfit{


std::vector<int> MFAvalues::unflatten(int index){
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






int runMFA( diy::mpi::environment &env,  diy::mpi::communicator &world){

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












}
