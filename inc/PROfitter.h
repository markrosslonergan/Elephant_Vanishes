#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

namespace PROfit {

    struct PROfitterConfig {
        LBFGSpp::LBFGSBParam<float> param;
        int n_multistart = 2000, n_swarm_particles = 20, n_swarm_iterations=100, n_localfit=1;
        size_t n_max_local_retries = 3;

        void print(){
            log<LOG_INFO>(L"%1% || Printing PROfitterConifg Values.") % __func__;
            log<LOG_INFO>(L"%1% || ------------ PROfitter specific -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || n_multistart: %2% (default 2000) ") % __func__ % n_multistart;
            log<LOG_INFO>(L"%1% || n_swarm_particles: %2% (default 20) ") % __func__ % n_swarm_particles;
            log<LOG_INFO>(L"%1% || n_swarm_iterations: %2% (default 100) ") % __func__ % n_swarm_iterations;
            log<LOG_INFO>(L"%1% || n_localfit: %2% (default 1) ") % __func__ % n_localfit;
            log<LOG_INFO>(L"%1% || n_max_local_retries: %2% (default 3) ") % __func__ % n_max_local_retries;
            log<LOG_INFO>(L"%1% || ------------ LBFGSBParam -------------- ") % __func__ ;
            log<LOG_INFO>(L"%1% || m: %2%  (default %3%) ") % __func__ % param.m % 6;
            log<LOG_INFO>(L"%1% || epsilon: %2%  (default %3%) ") % __func__ % param.epsilon % 1e-5;
            log<LOG_INFO>(L"%1% || epsilon_rel: %2%  (default %3%) ") % __func__ % param.epsilon_rel % 1e-5;
            log<LOG_INFO>(L"%1% || past: %2%  (default %3%) ") % __func__ % param.past % 1;
            log<LOG_INFO>(L"%1% || delta: %2%  (default %3%) ") % __func__ % param.delta % 1e-10;
            log<LOG_INFO>(L"%1% || max_iterations: %2%  (default %3%) ") % __func__ % param.max_iterations % 0;
            log<LOG_INFO>(L"%1% || max_submin: %2%  (default %3%) ") % __func__ % param.max_submin % 20;
            log<LOG_INFO>(L"%1% || max_linesearch: %2%  (default %3%) ") % __func__ % param.max_linesearch % 20;
            log<LOG_INFO>(L"%1% || min_step: %2%  (default %3%) ") % __func__ % param.min_step % 1e-20;
            log<LOG_INFO>(L"%1% || max_step: %2%  (default %3%) ") % __func__ % param.max_step % 1e20;
            log<LOG_INFO>(L"%1% || ftol: %2%  (default %3%) ") % __func__ % param.ftol % 1e-4;
            log<LOG_INFO>(L"%1% || wolfe: %2%  (default %3%) ") % __func__ % param.wolfe % 0.9;
            log<LOG_INFO>(L"%1% || ------------ Check Param LBFGSBParam Below -------------- ") % __func__ ;
            param.check_param();

        }
    };
            
    class PROfitter {
    public:
        Eigen::VectorXf ub, lb, best_fit;
        PROfitterConfig fitconfig;
        LBFGSpp::LBFGSBSolver<float> solver;
        uint32_t seed;

        PROfitter(const Eigen::VectorXf ub, const Eigen::VectorXf lb, PROfitterConfig fitconfig_ = {}, uint32_t inseed = 0)
            : ub(ub), lb(lb), fitconfig(fitconfig_), solver(fitconfig.param), seed(inseed) {}

        float Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt = Eigen::VectorXf());

        Eigen::VectorXf FinalGradient() const {return solver.final_grad();}
        float FinalGradientNorm() const {return solver.final_grad_norm();}
        Eigen::MatrixXf Hessian() const {return solver.final_approx_hessian();}
        Eigen::MatrixXf InverseHessian() const {return solver.final_approx_inverse_hessian();}
        Eigen::MatrixXf Covariance() const {return InverseHessian();}
        Eigen::VectorXf BestFit() const {return best_fit;}

        // If you don't belive the uncertainties on the parameters, you can use the final fit value to estimate the variance
        Eigen::MatrixXf ScaledCovariance(float chi2, int n_datapoint) const {return Covariance()*chi2/float(n_datapoint-best_fit.size());}

    };

}

#endif

