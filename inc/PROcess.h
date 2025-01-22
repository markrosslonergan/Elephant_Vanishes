#ifndef PROCESS_H
#define PROCESS_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"
#include "PROsc.h"
#include "PROpeller.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"

#include "TH2D.h"

namespace PROfit{

    /* Function: 
     *  The master weighting function that combines all weights and fills into spectrum PROspec, event-by-event
     */
    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc *inosc, std::vector<float> &inshifts, std::vector<float> &physparams, bool binned = false);

  PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, const PROsc *inosc, std::vector<TH2D*> inweighthists, std::vector<float> &physparams, bool binned = false);

    float GetOscWeight(int rule, float le, const PROsc &inosc, std::vector<float> &inphysparams);
    float GetOscWeight(int ev_idx, const PROpeller &inprop, const PROsc &inosc, std::vector<float> &inphysparams);
    float GetSystWeight(int ev_idx, const PROpeller &inprop, const PROsyst &insyst, float inshift, std::string insystname);

};

#endif
