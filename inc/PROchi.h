#ifndef PROCHI_H_
#define PROCHI_H_

// STANDARD

// OUR INCLUDES
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROsc.h"
#include "PROcess.h"

namespace PROfit{



    class PROchi
    {
        private:
            std::string model_tag;
            const PROconfig *config;
            const PROpeller *peller;
            const PROsyst *syst; 
            const PROsc *osc;
            const PROspec data;

            Eigen::VectorXd last_param;
            float last_value;
        public:
            PROchi(const std::string tag, const PROconfig *conin, const PROpeller *pin, const PROsyst *systin, const PROsc *oscin, const PROspec &datain) : model_tag(tag), config(conin), peller(pin), syst(systin), osc(oscin), data(datain) {last_value = 0.0; last_param = Eigen::VectorXd::Zero(config->m_num_bins_total); }
            float operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient)
            {

                //std::vector<float> shifts = param(Eigen::SeqN(2,1)).array();
                //std::vector<float> fitparams = param(Eigen::SeqN(0,2)).array();
 
                // Get Spectra from FillRecoSpectra
                Eigen::VectorXd subvector1 = param.segment(0, 2);
                std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());
                Eigen::VectorXd subvector2 = param.segment(2,1);
                std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());

                PROspec result = FillRecoSpectra(*config, *peller, *syst, *osc, shifts, fitparams);
          
                std::cout<<"Spec "<<result.Spec()<<" .. "<<std::endl;
                result.Print();

                // Calcuate Full Covariance matrix
                Eigen::MatrixXd diag = result.Spec().array().matrix().asDiagonal(); 
                Eigen::MatrixXd full_covariance =  diag*(syst->fractional_covariance)*diag;
                std::cout<<"Full: "<<full_covariance.size()<<std::endl;
                std::cout<<full_covariance<<std::endl;

                // Collapse Covariance and Spectra 
                Eigen::MatrixXd collapsed_full_covariance =  CollapseMatrix(*config,full_covariance);  
                std::cout<<"cFull: "<<collapsed_full_covariance.size()<<std::endl;
                std::cout<<collapsed_full_covariance<<std::endl;

                Eigen::MatrixXd stat_covariance = data.Spec().array().matrix().asDiagonal();
                Eigen::MatrixXd collapsed_stat_covariance = CollapseMatrix(*config, stat_covariance); 
                std::cout<<"cStat: "<<collapsed_stat_covariance.size()<<std::endl;
                std::cout<<collapsed_stat_covariance<<std::endl;
               

                // Invert Collaped Matrix Matrix 
                Eigen::MatrixXd inverted_collapsed_full_covariance = (collapsed_full_covariance+collapsed_stat_covariance).inverse();

                std::cout<<"shape: "<<inverted_collapsed_full_covariance.size()<<std::endl;
                std::cout<<inverted_collapsed_full_covariance<<std::endl;

                // Calculate Chi^2  value
                Eigen::VectorXd delta  = result.Spec() - data.Spec(); 
                float pull = subvector2.array().square().sum(); 
                float value = (delta.transpose())*inverted_collapsed_full_covariance*(delta);

                // Simple gradient here
                Eigen::VectorXd diff = param-last_param;
                gradient = (value-last_value)/diff.array();

                std::cout<<"Grad: "<<gradient<<std::endl;

                log<LOG_DEBUG>(L"%1% || value %2%, last_value %3%, pull") % __func__ % value  % last_value % pull;

                //Update last param
                last_param = param;
                last_value = value;
                return value;
            }
    };


}
#endif
