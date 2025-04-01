#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

namespace PROfit {

class PROfitter {
public:
    Eigen::VectorXf ub, lb, best_fit;
    LBFGSpp::LBFGSBParam<float> param;
    LBFGSpp::LBFGSBSolver<float> solver;
    int n_multistart = 1250, n_swarm_iterations_particles = 25, n_swarm_iterations=200;
    size_t n_max_local_retries = 2;

    uint32_t seed;

    PROfitter(const Eigen::VectorXf ub, const Eigen::VectorXf lb, const LBFGSpp::LBFGSBParam<float> &param = {}, uint32_t inseed = 0)
        : ub(ub), lb(lb), param(param), solver(param), seed(inseed) {}

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

