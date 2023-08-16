#include "PROcovariancegen.h"
using namespace PROfit;


int generateFracCovarianceFromXML(const PROconfig &inconfig, Eigen::MatrixXd &out_frac_covar){

    log<LOG_DEBUG>(L"%1% || Starting to construct CovarianceMatrixGeneration in EventWeight Mode  ") % __func__ ;

    double tolerence_positivesemi = 1e-5;
    double is_small_negative_eigenvalue = false;

    int num_files = inconfig.m_num_mcgen_files;

    std::vector<int> nentries(num_files,0);
    std::vector<int> used_montecarlos(num_files,0);

    std::vector<std::unique_ptr<TFile>> files(num_files);
    std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
    std::vector<std::map<std::string, std::vector<eweight_type> >* > f_weights(num_files,nullptr);

    //inconfig.m_mcgen_additional_weight.resize(num_files,1.0); its const, not allowed

    double FIX_plot_pot = 1;

    //std::cout << " -------------------------------------------------------------" << std::endl;
    //std::cout << " Initilizing " << universes_used << " universes." << std::endl;
    //std::cout << " -------------------------------------------------------------" << std::endl;

    std::vector<double> base_vec (inconfig.m_num_bins_total,0.0);

    std::cout << " Full concatanated vector has : " << inconfig.m_num_bins_total << std::endl;

    std::vector<std::vector<double>> multi_vecspec; ///FIX REplace with Eigen
    multi_vecspec.clear();
    //multi_vecspec.resize(universes_used,base_vec);

    //std::cout << " multi_vecspec now initilized of size :" << multi_vecspec.size() << std::endl;
    //std::cout << " Reading the data files" << std::endl;
    //watch.Reset();
    //watch.Start();

    for(int j=0; j < num_files; j++){
        int nevents = std::min(inconfig.m_mcgen_maxevents[j], nentries[j]);
        std::cout << " Starting @ data file=" << files[j]->GetName() <<" which has "<<nevents<<" Events. "<<std::endl;
        size_t nbytes = 0;
        for(int i=0; i < nevents; i++) {
            if(i%100==0)std::cout<<" -- uni :"<<i<<" / "<<nevents<<std::endl;
            nbytes+= trees[j]->GetEntry(i);
            //ProcessEvent(*(f_weights[j]),j,i);
            //INPUT PROCESS FIX




        } //end of entry loop
        std::cout << " nbytes read=" << nbytes << std::endl;

    } //end of file loop

    //watch.Stop();
    //std::cout << " done CpuTime=" << watch.CpuTime() << " RealTime=" << watch.RealTime() << std::endl;

    /***************************************************************
     *		Now some clean-up and Writing
     * ************************************************************/

    //FIX check?
    //NOPE In fact, calling Close() on a TFile object managed by a std::unique_ptr can lead to undefined behavior because the std::unique_ptr may delete the object before the Close() method finishes executing. So, it is best to rely on the std::unique_ptr to manage the lifetime of the TFile object and not call Close() explicitly.
    //for(auto f: files){
    //    std::cout << " TFile::Close() file=" << f->GetName() <<  std::endl;
    //    f->Close();
    //}
    std::cout << " End" << std::endl;


    return 0;
}


Eigen::MatrixXd GenerateCovarMatrix(const SystStruct& sys_obj){

    int n_universe = sys_obj.GetNUniverse(); 
    std::string sys_name = sys_obj.GetSysName();
    
    const PROspec& cv_spec = sys_obj.CV();
    long int nbins = cv_spec.GetNbins();

    //build full covariance matrix 
    Eigen::MatrixXd full_covar_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
    for(int i = 0; i != n_universe; ++i){
	auto spec_diff  = cv_spec - sys_obj.Variation(i);
	full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<double>(n_universe);
    }
 
    //build fractional covariance matrix 
    //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
    Eigen::MatrixXd cv_spec_matrix =  Eigen::MatrixXd::Identity(nbins, nbins);
    for(int i =0; i != nbins; ++i)
	cv_spec_matrix(i, i) = 1.0/cv_spec.GetBinContent(i);

    //second, get fractioal covar
    Eigen::MatrixXd frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;

    return frac_covar_matrix;
}
