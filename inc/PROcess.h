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

namespace PROfit{

    /* Function: 
     *  The master weighting function that combines all weights and fills into spectrum PROspec, event-by-event
     */

    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned = false);
    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc *inosc, const std::vector<float> &inshifts, const std::vector<float> &physparams, bool binned = false);
    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc *inosc, const std::map<std::string, float> &inshifts, const std::vector<float> &physparams, bool binned = false);
    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const std::map<std::string, float> &inshifts, bool binned = false);

    float GetOscWeight(int rule, float le, const PROsc &inosc, const std::vector<float> &inphysparams);
    float GetOscWeight(int ev_idx, const PROpeller &inprop, const PROsc &inosc, const std::vector<float> &inphysparams);
    float GetSystWeight(int ev_idx, const PROpeller &inprop, const PROsyst &insyst, float inshift, std::string insystname);

};

#endif
