#include "PROcess.h"
#include "PROsyst.h"

namespace PROfit {

PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc &inosc, std::vector<float> &inshifts, std::vector<float> &physparams){

    PROspec myspectrum(inconfig.m_num_bins_total);

    for(int i = 0; i<inprop.truth.size(); ++i){

        float oscw  = GetOscWeight(i, inprop, inosc, physparams);
        float add_w = inprop.added_weights[i]; 

        const int subchannel = FindSubchannelIndexFromGlobalBin(inconfig, inprop.bin_indices[i]);
        const int true_bin = FindGlobalTrueBin(inconfig, inprop.baseline[i] / inprop.truth[i], subchannel);
        float systw = 1;
        for(size_t j = 0; j < inshifts.size(); ++j) {
            systw *= insyst.GetSplineShift(j, inshifts[j], true_bin);
        }
   
        float finalw = oscw * systw * add_w;

        myspectrum.Fill(inprop.bin_indices[i], finalw);
    }

    return myspectrum;

}

float GetOscWeight(int ev_idx, const PROpeller &inprop, const PROsc &inosc, std::vector<float> &inphysparams){

        //get subchannel here from pdg. this will be added when we agree on convention.
        //for now everything is numu disappearance 3+1. 
        // inphysparams[0] is delta-msq
        // inphysparams[1] is sinsq2thmumu

        float prob = inosc.Pmumu(inphysparams[0], inphysparams[1], inprop.truth[ev_idx], inprop.baseline[ev_idx]);
        return prob;
    }
    
};
