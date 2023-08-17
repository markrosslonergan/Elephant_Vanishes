#ifndef PROMFA_H_
#define PROMFA_H_

#include "PROconfig.h"
#include "PROspec.h"

#include <algorithm>
#include <limits>

#include "CLI11.h"
#include "LBFGSB.h"

#include <diy/master.hpp>
#include <diy/reduce.hpp>
#include <diy/partners/merge.hpp>
#include <diy/decomposition.hpp>
#include <diy/assigner.hpp>
#include <diy/mpi.hpp>
#include <diy/serialization.hpp>
#include <diy/partners/broadcast.hpp>
#include <diy/reduce-operations.hpp>
#include <diy/io/block.hpp>

#include <mfa/mfa.hpp>
#include <mfa/block_base.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/NumericalDiff>

#define FMT_HEADER_ONLY
//#include <fmt/format.h>


namespace PROfit {

};

#endif
