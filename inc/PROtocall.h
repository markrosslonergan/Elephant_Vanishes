#ifndef PRO_TO_CALL_H
#define PRO_TO_CALL_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"

namespace PROfit{


    /* Function: given a value for reconstructed variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the reco value is out of range
     */
    int FindLocalBin(const PROconfig &inconfig, double reco_value, int channel_index);

    /* Function: given a value for reconstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the reco value is out of range, then return value of -1
     */
    int FindGlobalBin(const PROconfig &inconfig, double reco_value, int subchannel_index);
    int FindGlobalBin(const PROconfig &inconfig, double reco_value, const std::string& subchannel_fullname);



};

#endif
