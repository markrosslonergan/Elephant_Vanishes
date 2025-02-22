#include "PROchi.h"
#include "PROcess.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
using namespace PROfit;


PROchi::PROchi(const std::string tag, const PROconfig &conin, const PROpeller &pin, const PROsyst *systin, const PROmodel &modelin, const PROspec &datain, EvalStrategy strat, std::vector<float> physics_param_fixed) : PROmetric(), model_tag(tag), config(conin), peller(pin), syst(systin), model(modelin), data(datain), strat(strat), physics_param_fixed(physics_param_fixed), correlated_systematics(false) {
    last_value = 0.0; last_param = Eigen::VectorXf::Zero(model.nparams+syst->GetNSplines()); 
    fixed_index = -999;

    // Build the correlation matrix between priors if configured to
    if (conin.m_mcgen_correlations.size()) {
        correlated_systematics = true;
        prior_covariance = Eigen::MatrixXf::Identity(syst->GetNSplines(), syst->GetNSplines());
        for (auto const &t: conin.m_mcgen_correlations) {
          auto itA = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<0>(t));
          if (itA == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<0>(t).c_str();
            continue;
          }

          auto itB = std::find(systin->spline_names.begin(), systin->spline_names.end(), std::get<1>(t));
          if (itB == systin->spline_names.end()) {
            log<LOG_WARNING>(L"%1% || Systematic correlation %2% not in list. Skipping.") % __func__ % std::get<1>(t).c_str();
            continue;
          }
         
          int iA = std::distance(systin->spline_names.begin(), itA);
          int iB = std::distance(systin->spline_names.begin(), itB);

          // set correlations
          prior_covariance(iA, iB) = std::get<2>(t);
          prior_covariance(iB, iA) = std::get<2>(t);
        }
    }

    collapsed_stat_covariance = data.Spec().array().matrix().asDiagonal();
}

float PROchi::Pull(const Eigen::VectorXf &systs) {
    // No correlations: sum of squares
    if (!correlated_systematics) return systs.array().square().sum();

    // Otherwise dot onto covariance
    return systs.dot(prior_covariance.inverse() * systs);
}

void PROchi::fixSpline(int fix, float valin){
    fixed_index=fix;
    fixed_val=valin;
    return;
}
float PROchi::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient){
    return PROchi::operator()(param, gradient, true);
}


float PROchi::operator()(const Eigen::VectorXf &param, Eigen::VectorXf &gradient, bool rungradient){
    size_t nparams = model.nparams+syst->GetNSplines();
    size_t nsyst = syst->GetNSplines();
    log<LOG_DEBUG>(L"%1% || nparams is %2%, nsyst is %3% ") % __func__ % nparams % nsyst;    

    // Get Spectra from FillRecoSpectra
    Eigen::VectorXf subvector1 = param.segment(0, nparams - nsyst);
    Eigen::VectorXf subvector2 = param.segment(nparams - nsyst, nsyst);
    log<LOG_DEBUG>(L"%1% || after subvectors ") % __func__ ;
    
    PROspec result = FillRecoSpectra(config, peller, *syst, model, param, strat == BinnedChi2);

    log<LOG_DEBUG>(L"%1% || before inverted ") % __func__ ;
    Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_bins_total_collapsed,config.m_num_bins_total_collapsed);
    
    //only calculate a syst covariance if we have any covariance parameters as defined in the xml
    log<LOG_DEBUG>(L"%1% || before covariance ") % __func__ ;
    if(syst->GetNCovar()){

      // Calculate Full Syst Covariance matrix
      Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
      Eigen::MatrixXf full_covariance =  diag*(syst->fractional_covariance)*diag;
      //std::cout<<"Full: "<<full_covariance.size()<<std::endl;
      //std::cout<<full_covariance<<std::endl;

      // Collapse Covariance and Spectra 
      Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  

      //std::cout<<"cFull: "<<collapsed_full_covariance.size()<<std::endl;
      //std::cout<<collapsed_full_covariance<<std::endl;

      // Invert Collaped Matrix Matrix 
      inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();
      }

    else{
        
    	inverted_collapsed_full_covariance = (collapsed_stat_covariance).inverse();
         
       }

    log<LOG_DEBUG>(L"%1% || before delta def ") % __func__ ;
    Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec();
    log<LOG_DEBUG>(L"%1% || after delta def ") % __func__ ;    
    float pull = Pull(subvector2);
    float covar_portion = (delta.transpose())*inverted_collapsed_full_covariance*(delta);
    float value = covar_portion + pull;

    if(rungradient){
        float dval = 1e-4;
        for (int i = 0; i < nparams; i++) {
            //if(i == fixed_index) gradient(i) = 0;
            //Eigen::VectorXd tmpParams = last_param;
            Eigen::VectorXf tmpParams = param;
            int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
            if(!sgn) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) < -4 + dval) sgn = 1;
            //if(fitparams.size() != 0 && i == 1 && param(i) > 0 - dval) sgn = -1;
            tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;
            
            Eigen::VectorXf subvector1 = tmpParams.segment(0, nparams - nsyst);
            Eigen::VectorXf subvector2 = tmpParams.segment(nparams - nsyst, nsyst);
            PROspec result = FillRecoSpectra(config, peller, *syst, model, tmpParams, strat != EventByEvent);
            // Calcuate Full Covariance matrix
            Eigen::MatrixXf inverted_collapsed_full_covariance(config.m_num_bins_total_collapsed,config.m_num_bins_total_collapsed);

            if(syst->GetNCovar()){

                Eigen::MatrixXf diag = result.Spec().array().matrix().asDiagonal(); 
                Eigen::MatrixXf full_covariance =  diag*(syst->fractional_covariance)*diag;
                // Collapse Covariance and Spectra 
                Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  
                // Invert Collaped Matrix Matrix 
                inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();
            }
            else{
    	        inverted_collapsed_full_covariance = (collapsed_stat_covariance).inverse();
                }
           
            // Calculate Chi^2  value
            Eigen::VectorXf delta  = CollapseMatrix(config,result.Spec()) - data.Spec(); 

            float pull = Pull(subvector2);
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


