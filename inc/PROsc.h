#ifndef PROSC_H_
#define PROSC_H_

// STANDARD
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <functional>

// TINYXML2
#include "tinyxml2.h"

#include <Eigen/Eigen>

//PROfit
#include "PROlog.h"
#include "PROpeller.h"

namespace PROfit{


    /* 
     * Class: Oscillation Class for oscillating event-by-event MC and prodice PROspec
     * Note:
     *  Currently 3+1 SBL approximation is only model and hardcoded. So a very simple class.
     * Todo: 
     *  Add Nuquids interface if desired
     *  Add interfacte for defining own model
     */

    class PROsc {
        private:
        public:

            PROsc(const PROpeller &prop){

                model_functions.push_back([this](std::vector<float>v, float , float ) {(void)this; return 1.0; });//c++14 way of ignoring unused
                model_functions.push_back([this](std::vector<float>v, float c, float d) {return this->Pmumu(v[0],v[1], c, d); });
                model_functions.push_back([this](std::vector<float>v, float c, float d) {return this->Pmue(v[0],v[1], c, d); });

                for(size_t m = 0; m < model_functions.size(); ++m) {
                    hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
                    Eigen::MatrixXf &h = hists.back();
                    for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
                        if(prop.model_rule[i] != (int)m) continue;
                        int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
                        h(tbin, rbin) += prop.added_weights[i];
                    }
                }

            };

            /* Function: 3+1 numu->nue apperance prob in SBL approx */
            float Pmue(float dmsq, float sinsq2thmue, float enu, float baseline) const{
                dmsq =std::pow(10, dmsq);
                sinsq2thmue =std::pow(10, sinsq2thmue);

                if(sinsq2thmue > 1) sinsq2thmue = 1;
                if(sinsq2thmue < 0) sinsq2thmue = 0;

                float sinterm = std::sin(1.27*dmsq*(baseline/enu));
                float prob    = sinsq2thmue*sinterm*sinterm;

                if(prob<0.0 || prob >1.0){
                    //Do we need this when its hardcoded simple cmath above?
                    log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math.") % __func__ % prob;
                    log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
                    exit(EXIT_FAILURE);
                }
                return prob;
            }

            /* Function: 3+1 numu->numue disapperance prob in SBL approx */
            float Pmumu(float dmsq, float sinsq2thmumu, float enu, float baseline) const{
                dmsq =std::pow(10, dmsq);
                sinsq2thmumu =std::pow(10, sinsq2thmumu);

                if(sinsq2thmumu > 1) {
                    log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1") % __func__ % sinsq2thmumu;
                    sinsq2thmumu = 1;
                }
                if(sinsq2thmumu < 0) {
                    log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0") % __func__ % sinsq2thmumu;
                    sinsq2thmumu = 0;
                }

                float sinterm = std::sin(1.27*dmsq*(baseline/enu));
                float prob    = 1.0 - (sinsq2thmumu*sinterm*sinterm);

                if(prob<0.0 || prob >1.0){
                    log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, sinsq2thmumu = %4%, enu = %5%, baseline = %6%") % __func__ % prob % dmsq % sinsq2thmumu % enu % baseline;
                    log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
                    exit(EXIT_FAILURE);
                }

                return prob;
            }

        // TODO: Fix this to do more than numu disappearance
        size_t nphysicsparams = 2;
        Eigen::VectorXf lb{{-2, -std::numeric_limits<float>::infinity()}};
        Eigen::VectorXf ub{{2, 0}};
        std::vector<std::string> param_names{"dmsq", "sinsq2thmm"}; 

        std::vector<std::function<float(std::vector<float>,float,float)>> model_functions;

        std::vector<Eigen::MatrixXf> hists;

    };

}
#endif
