#ifndef PROSURF_H
#define PROSURF_H

#include "PROconfig.h"
#include "PROspec.h"
#include "PROpeller.h"
#include "PROsyst.h"
#include "PROchi.h"

#include "LBFGSB.h"

#include <Eigen/Eigen>

#include <random>
#include <thread>
#include <future>

#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TMarker.h"
#include "TLine.h"
#include "TText.h"

namespace PROfit {

  struct surfOut{
    std::vector<int> grid_index;
    std::vector<float> grid_val;
    float chi;
  };

  struct profOut{
    std::vector<float> knob_vals;
    std::vector<float> knob_chis;
    float chi;
  };

  class PROfile {

  public:
    PROmetric &metric;
    std::vector<float> asym_error_phys;
    std::vector<float> minChiSquares; // min chi square for dmsq and sinsq2th after profiling  
    int profile_fit_failures = 0;
    PROfile(const PROconfig &config, const PROpeller &prop, const PROsyst &systs, const PROmodel &model, const PROspec &data, PROmetric &metric, std::string filename, bool with_osc = false, int nThreads = 1, const Eigen::VectorXf& init_seed = Eigen::VectorXf() ) ;

    std::vector<profOut> PROfilePointHelper(const PROsyst *systs, int start, int end, bool with_osc, const Eigen::VectorXf& init_seed = Eigen::VectorXf());
  };

  class PROsurf {
  public:

    PROmetric &metric;
    size_t x_idx, y_idx, nbinsx, nbinsy;
    Eigen::VectorXf edges_x, edges_y;
    Eigen::MatrixXf surface;

    enum LogLin {
      LinAxis,
      LogAxis,
    };

  PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, const Eigen::VectorXf &edges_x, size_t nbinsy, const Eigen::VectorXf &edges_y) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(edges_x), edges_y(edges_y), surface(nbinsx, nbinsy) { }

    PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi);

    std::vector<surfOut> PointHelper(std::vector<surfOut> multi_physics_params, int start, int end);

    void FillSurfaceStat(const PROconfig &config, std::string filename);
    void FillSurface(std::string filename, int nthreads = 1);

  };

}

#endif
