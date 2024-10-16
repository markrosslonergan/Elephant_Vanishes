#ifndef PROMETRIC_H
#define PROMETRIC_H

#include <Eigen/Eigen>

namespace PROfit {

class PROmetric {
public:
    virtual double operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient) = 0;
    virtual double operator()(const Eigen::VectorXd &param, Eigen::VectorXd &gradient, bool nograd) = 0;
};

};

#endif

