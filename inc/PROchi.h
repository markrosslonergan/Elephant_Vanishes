#ifndef PROCHI_H_
#define PROCHI_H_

// STANDARD
#include <string>
#include <vector>

#include <Eigen/Eigen>

// OUR INCLUDES
#include "PROconfig.h"
#include "PROsyst.h"
#include "PROpeller.h"
#include "PROsc.h"
#include "PROcess.h"
#include "PROmetric.h"

namespace PROfit{

    /* 
     * Class: Class that gathers the MC (PROpeller), Systematics (PROsyst) and model (PROsc) and forms a function calculating a chi^2 that can be minimized over
     * Note:
     *  the PROconfig,PROpeller..etc need to be accessable by the class so that the function operato "()" when passed to minimizer can access them. 
     *  Saved as pointers to the objects created in the primary executable.
     * Todo:
     *  Add capability to define function externally?
     *  Improve gradient calculation
     *  */

    class PROchi : public PROmetric
    {
        public:
            enum EvalStrategy {
                EventByEvent,
                BinnedGrad,
                BinnedChi2
            };

        private:

            std::string model_tag;

            const PROconfig *config;
            const PROpeller *peller;
            const PROsyst *syst; 
            const PROsc *osc;
            const PROspec data;
            int nparams;
            int nsyst;
            EvalStrategy strat;
            std::vector<float> physics_param_fixed;
                        //Do we want to fix any param?
            int fixed_index;
            float fixed_val;

            //Save last values for gradient calculation
            Eigen::VectorXd last_param;
            float last_value;
        public:

            /*Function: Constructor bringing all objects together*/
            PROchi(const std::string tag, const PROconfig *conin, const PROpeller *pin, const PROsyst *systin, const PROsc *oscin, const PROspec &datain, int nparams, int nsyst, EvalStrategy strat = EventByEvent, std::vector<float> physics_param_fixed = std::vector<float>());

            /*Function: operator() is what is passed to minimizer.*/
            virtual double operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient);
            virtual double operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient, bool nograd);

            void fixSpline(int fix, double valin);

    };


}
#endif
