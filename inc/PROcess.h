#ifndef PROCESS_H
#define PROCESS_H

#include <Eigen/Eigen>

// PROfit include 
#include "PROconfig.h"
#include "PROmodel.h"
#include "PROpeller.h"
#include "PROspec.h"
#include "PROsyst.h"

#include "TH2D.h"

namespace PROfit{

    /* Function: 
     *  The master weighting function that combines all weights and fills into spectrum PROspec, event-by-event
     */

    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned = false);
    PROspec FillOtherCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, size_t other_index);

  //ETW 1/22/2025 Add function to fill spectrum using weights from input histogram
    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = false);

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = true);
    PROspec FillOtherRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, size_t other_index);
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int other_index = -1);
};

#endif
