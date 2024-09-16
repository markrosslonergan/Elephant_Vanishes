#include "PROchi.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
using namespace PROfit;
using namespace autodiff;

PROchi::PROchi(const std::string tag, const PROconfig *conin, const PROpeller *pin, const PROsyst *systin, const PROsc *oscin, const PROspec &datain, int nparams, int nsyst, double logdmsq, double logsinsq2tmm) : model_tag(tag), config(conin), peller(pin), syst(systin), osc(oscin), data(datain), nparams(nparams), nsyst(nsyst), logdmsq(logdmsq), logsinsq2tmm(logsinsq2tmm) {
    last_value = 0.0; last_param = Eigen::VectorXd::Zero(nparams); 
}

float PROchi::operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient){

    //std::vector<float> shifts = param(Eigen::SeqN(2,1)).array();
    //std::vector<float> fitparams = param(Eigen::SeqN(0,2)).array();

    // Get Spectra from FillRecoSpectra
    Eigen::VectorXd subvector1 = param.segment(0, nparams - nsyst);
    std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());
    if(fitparams.size() == 0 && logdmsq != 0) {
        fitparams.push_back(logdmsq);
        fitparams.push_back(std::pow(10, logsinsq2tmm));
    }
    Eigen::VectorXd subvector2 = param.segment(nparams - nsyst, nsyst);
    std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());

    log<LOG_DEBUG>(L"%1% || Shifts size is %2%") % __func__ % shifts.size();

    PROspec result = FillRecoSpectra(*config, *peller, *syst, osc, shifts, fitparams);

    //std::cout<<"Spec "<< result.Spec()<<" .. "<<std::endl;
    result.Print();

    // Calcuate Full Covariance matrix
    Eigen::MatrixXd diag = result.Spec().array().matrix().asDiagonal(); 
    Eigen::MatrixXd full_covariance =  diag*(syst->fractional_covariance)*diag;
    //std::cout<<"Full: "<<full_covariance.size()<<std::endl;
    //std::cout<<full_covariance<<std::endl;

    // Collapse Covariance and Spectra 
    Eigen::MatrixXd collapsed_full_covariance =  CollapseMatrix(*config,full_covariance);  
    //std::cout<<"cFull: "<<collapsed_full_covariance.size()<<std::endl;
    //std::cout<<collapsed_full_covariance<<std::endl;

    Eigen::MatrixXd stat_covariance = data.Spec().array().matrix().asDiagonal();
    Eigen::MatrixXd collapsed_stat_covariance = CollapseMatrix(*config, stat_covariance); 
    //std::cout<<"cStat: "<<collapsed_stat_covariance.size()<<std::endl;
    //std::cout<<collapsed_stat_covariance<<std::endl;

    log<LOG_DEBUG>(L"%1% || HERE") % __func__ ;

    // Invert Collaped Matrix Matrix 
    Eigen::MatrixXd inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();

    //std::cout<<"shape: "<<inverted_collapsed_full_covariance.size()<<std::endl;
    //std::cout<<inverted_collapsed_full_covariance<<std::endl;

    // Calculate Chi^2  value
    Eigen::VectorXd delta  = result.Spec() - data.Spec(); 
    float pull = subvector2.array().square().sum(); 
    float dmsq_penalty = 0;
    if(osc) {
    //    dmsq_penalty = param(0) < 0.1 ? std::pow(param(0) - 0.1, 2)/0.01 :
    //                         param(0) > 20  ? std::pow(param(0) - 20, 2)/100 : 0;
    }
    float value = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + dmsq_penalty + pull;

    float dval = 1e-4;
    for (int i = 0; i < nparams; i++) {
        //Eigen::VectorXd tmpParams = last_param;
        Eigen::VectorXd tmpParams = param;
        int sgn = ((param(i) - last_param(i)) > 0) - ((param(i) - last_param(i)) < 0);
        if(!sgn) sgn = 1;
        if(fitparams.size() != 0 && i == 1 && param(i) < dval) sgn = 1;
        else if(fitparams.size() != 0 && i == 1 && param(i) > 1 - dval) sgn = -1;
        tmpParams(i) = /*param(i) != last_param(i) ? param(i) :*/ param(i) + sgn * dval;
        //Eigen::VectorXd subvector2 = tmpParams;
        Eigen::VectorXd subvector1 = tmpParams.segment(0, nparams - nsyst);
        std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());
        if(fitparams.size() == 0 && logdmsq != 0) {
            fitparams.push_back(logdmsq);
            fitparams.push_back(std::pow(10, logsinsq2tmm));
        }
        Eigen::VectorXd subvector2 = tmpParams.segment(nparams - nsyst, nsyst);
        std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());
        PROspec result = FillRecoSpectra(*config, *peller, *syst, osc, shifts, fitparams);
        // Calcuate Full Covariance matrix
        Eigen::MatrixXd diag = result.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXd full_covariance =  diag*(syst->fractional_covariance)*diag;
        // Collapse Covariance and Spectra 
        Eigen::MatrixXd collapsed_full_covariance =  CollapseMatrix(*config,full_covariance);  
        Eigen::MatrixXd stat_covariance = data.Spec().array().matrix().asDiagonal();
        Eigen::MatrixXd collapsed_stat_covariance = CollapseMatrix(*config, stat_covariance); 
        // Invert Collaped Matrix Matrix 
        Eigen::MatrixXd inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();
        // Calculate Chi^2  value
        Eigen::VectorXd delta  = result.Spec() - data.Spec(); 
        float pull = subvector2.array().square().sum(); 
        float dmsq_penalty = 0;
        //if(osc) {
        //    dmsq_penalty = tmpParams(0) < 0.1 ? std::pow(tmpParams(0) - 0.1, 2)/0.01 :
        //                         tmpParams(0) > 20  ? std::pow(tmpParams(0) - 20, 2)/100 : 0;
        //}
        float value_grad = (delta.transpose())*inverted_collapsed_full_covariance*(delta) + dmsq_penalty + pull;
        //gradient(i) = (value-last_value)/(tmpParams(i) - last_param(i));
        gradient(i) = (value_grad-value)/(tmpParams(i) - param(i));
    }

    //std::cout<<"Grad: "<<gradient<<std::endl;

    //log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;
    log<LOG_DEBUG>(L"%1% || FINISHED ITERATION got vals: %2% %3%") % __func__ % value % last_value ;

    //Update last param
    last_param = param;
    last_value = value;

    return value;
}


