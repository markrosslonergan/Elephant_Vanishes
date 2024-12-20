#include "PROchi.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
using namespace PROfit;


PROchi::PROchi(const std::string tag, const PROconfig *conin, const PROpeller *pin, const PROsyst *systin, const PROsc *oscin, const PROspec &datain, int nparams, int nsyst, EvalStrategy strat, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), osc(oscin), data(datain), nparams(nparams), nsyst(nsyst), strat(strat), physics_param_fixed(physics_param_fixed) {
    last_value = 0.0; last_param = Eigen::VectorXd::Zero(nparams); 
    fixed_index = -999;
}

void PROchi::fixSpline(int fix, double valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
double PROchi::operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient){
    return PROchi::operator()(param, gradient, true);
}


double PROchi::operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient, bool rungradient){

    // Get Spectra from FillRecoSpectra
    Eigen::VectorXd subvector1 = param.segment(0, nparams - nsyst);
    std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());
    if(fitparams.size() == 0 && physics_param_fixed.size()!=0 ) {
        fitparams = physics_param_fixed;
    }

    Eigen::VectorXd subvector2 = param.segment(nparams - nsyst, nsyst);
    std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());

    log<LOG_DEBUG>(L"%1% || Shifts size is %2%") % __func__ % shifts.size();

    PROspec result = FillRecoSpectra(*config, *peller, *syst, osc, shifts, fitparams, strat == BinnedChi2);

    //std::cout<<"Spec "<< result.Spec()<<" .. "<<std::endl;
    //result.plotSpectrum(*config,"TTPT");

    Eigen::MatrixXd inverted_collapsed_full_covariance(config->m_num_bins_total_collapsed,config->m_num_bins_total_collapsed);
    
    Eigen::MatrixXd stat_covariance = data.Spec().array().matrix().asDiagonal();
    log<LOG_DEBUG>(L"%1% || stat %2% x %3% ") % __func__% stat_covariance.cols() % stat_covariance.rows();
    Eigen::MatrixXd collapsed_stat_covariance = CollapseMatrix(*config, stat_covariance); 
    log<LOG_DEBUG>(L"%1% || Collapsed the first matrix") % __func__;

    //std::cout<<"cStat: "<<collapsed_stat_covariance.size()<<std::endl;
    //std::cout<<collapsed_stat_covariance<<std::endl;
    
    //only calculate a syst covariance if we have any covariance parameters as defined in the xml
    if(syst->GetNCovar()){

      // Calculate Full Syst Covariance matrix
      Eigen::MatrixXd diag = result.Spec().array().matrix().asDiagonal(); 
      Eigen::MatrixXd full_covariance =  diag*(syst->fractional_covariance)*diag;
      //std::cout<<"Full: "<<full_covariance.size()<<std::endl;
      //std::cout<<full_covariance<<std::endl;

      // Collapse Covariance and Spectra 
      Eigen::MatrixXd collapsed_full_covariance =  CollapseMatrix(*config,full_covariance);  
      log<LOG_DEBUG>(L"%1% || Collapsed second matrix") % __func__;

      //std::cout<<"cFull: "<<collapsed_full_covariance.size()<<std::endl;
      //std::cout<<collapsed_full_covariance<<std::endl;

      // Invert Collaped Matrix Matrix 
      inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();
      }

    else{
        
    	inverted_collapsed_full_covariance = (collapsed_stat_covariance).inverse();
         
       }

      //std::cout<<"shape: "<<inverted_collapsed_full_covariance.size()<<std::endl;
      //std::cout<<inverted_collapsed_full_covariance<<std::endl;

    // Calculate Chi^2  value
    Eigen::VectorXd delta  = result.Spec() - data.Spec(); 
    //Collapse
    delta = CollapseMatrix(*config, delta);  

    if(!(fixed_index<0)){
        //subvector2[fixed_index]=0;   
        //Dont do this anymore
    }

    float pull = subvector2.array().square().sum(); 
    float dmsq_penalty = 0;
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion + dmsq_penalty + pull;


    if(rungradient){
        float dval = 1e-4;
        for (int i = 0; i < nparams; i++) {
            //Eigen::VectorXd tmpParams = last_param;
            Eigen::VectorXd tmpParams = param;
            int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
            if(!sgn) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) < -4 + dval) sgn = 1;
            //else if(fitparams.size() != 0 && i == 1 && param(i) > 0 - dval) sgn = -1;
            tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;
            
            Eigen::VectorXd subvector1 = tmpParams.segment(0, nparams - nsyst);
            std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());
            if(fitparams.size() == 0 && physics_param_fixed.size() != 0) {
                fitparams = physics_param_fixed;
            }
            Eigen::VectorXd subvector2 = tmpParams.segment(nparams - nsyst, nsyst);
            std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());
            PROspec result = FillRecoSpectra(*config, *peller, *syst, osc, shifts, fitparams, strat != EventByEvent);
            // Calcuate Full Covariance matrix
            Eigen::MatrixXd stat_covariance = data.Spec().array().matrix().asDiagonal();
            Eigen::MatrixXd collapsed_stat_covariance = CollapseMatrix(*config, stat_covariance); 
            Eigen::MatrixXd inverted_collapsed_full_covariance(config->m_num_bins_total_collapsed,config->m_num_bins_total_collapsed);

            if(syst->GetNCovar()){

                Eigen::MatrixXd diag = result.Spec().array().matrix().asDiagonal(); 
                Eigen::MatrixXd full_covariance =  diag*(syst->fractional_covariance)*diag;
                // Collapse Covariance and Spectra 
                Eigen::MatrixXd collapsed_full_covariance =  CollapseMatrix(*config,full_covariance);  
                // Invert Collaped Matrix Matrix 
                inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();
            }
            else{
    	        inverted_collapsed_full_covariance = (collapsed_stat_covariance).inverse();
                }
           
            // Calculate Chi^2  value
            Eigen::VectorXd delta  = result.Spec() - data.Spec(); 
            delta = CollapseMatrix(*config, delta);  

            if(!(fixed_index<0)){
                //subvector2[fixed_index]=0;   
                //dont do this, cinlude it
            }
            float pull = subvector2.array().square().sum(); 
            float dmsq_penalty = 0;
            float value_grad = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + dmsq_penalty + pull;
            
            gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
        }
    }
    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}


