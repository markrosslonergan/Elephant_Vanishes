#ifndef PROFITTER_H
#define PROFITTER_H

#include "PROmetric.h"

#include <Eigen/Eigen>
#include "LBFGSB.h"

namespace PROfit {

class PROfitter {
public:
    Eigen::VectorXd ub, lb, best_fit;
    LBFGSpp::LBFGSBParam<double> param;
    LBFGSpp::LBFGSBSolver<double> solver;
    int n_multistart = 100, n_localfit = 5;

    PROfitter(const Eigen::VectorXd ub, const Eigen::VectorXd lb, const LBFGSpp::LBFGSBParam<double> &param = {})
        : ub(ub), lb(lb), param(param), solver(param) {}

    double Fit(PROmetric &metric);

};

}

#endif

