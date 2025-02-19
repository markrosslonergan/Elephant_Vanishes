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

  //ETW 1/22/2025 Add function to fill spectrum using weights from input histogram
    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, const PROsc *inosc, std::vector<TH2D*> inweighthists, std::vector<float> &physparams, bool binned = false);

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned = true);
    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst);
>>>>>>> 71f12c6c446206394fa7fb6851d14f33af78a9a1
};

#endif
