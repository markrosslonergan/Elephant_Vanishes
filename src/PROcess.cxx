#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "TH2D.h"

#include <Eigen/Eigen>

#include <random>
#include <vector>

namespace PROfit {
    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(int i = 0; i < inprop.hist.rows(); ++i) {
                for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                    myspectrum.Fill(k, inprop.hist(i, k));
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                myspectrum.Fill(inprop.bin_indices[i], add_w);
            }
        }
        return myspectrum;
    }

    PROspec FillOtherCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, size_t other_index){
        PROspec myspectrum(inconfig.m_num_other_bins_total[other_index]);
        for(size_t i = 0; i<inprop.trueLE.size(); ++i){
            float add_w = inprop.added_weights[i]; 
            if(inprop.other_bin_indices[i][other_index] >= 0)
                myspectrum.Fill(inprop.other_bin_indices[i][other_index], add_w);
        }
        return myspectrum;
    }

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if(binned) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float systw = 1;
                for(int j = 0; j < shifts.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, shifts(j), i);
                }
                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, systw * oscw * inmodel.hists[j](i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
                float add_w = inprop.added_weights[i]; 
                const int true_bin = inprop.true_bin_indices[i]; 

                float systw = 1;
                for(int j = 0; j < shifts.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, shifts(j), true_bin);
                }

                float finalw = oscw * systw * add_w;

                myspectrum.Fill(inprop.bin_indices[i], finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillOtherRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROmodel &inmodel, const Eigen::VectorXf &params, size_t other_index){
        PROspec myspectrum(inconfig.m_num_other_bins_total[other_index]);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        for(size_t i = 0; i<inprop.trueLE.size(); ++i){
            float oscw  =  inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]);
            float add_w = inprop.added_weights[i]; 
            const int true_bin = inprop.true_bin_indices[i]; 

            float systw = 1;
            for(int j = 0; j < shifts.size(); ++j) {
                systw *= insyst.GetSplineShift(j, shifts(j), true_bin);
            }

            float finalw = oscw * systw * add_w;

            if(inprop.other_bin_indices[i][other_index] >= 0)
                myspectrum.Fill(inprop.other_bin_indices[i][other_index], finalw);
        }
        return myspectrum;
    }

    PROspec FillWeightedSpectrumFromHist(const PROconfig &inconfig, const PROpeller &inprop, std::vector<TH2D*> inweighthists, const PROmodel &inmodel, const Eigen::VectorXf &params, bool binned){
        PROspec myspectrum(inconfig.m_num_bins_total);
        Eigen::VectorXf phys   = params.segment(0, inmodel.nparams);
        Eigen::VectorXf shifts = params.segment(inmodel.nparams, params.size() - inmodel.nparams);

        if (binned) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromGlobalTrueBin(inprop.true_bin_indices[i]);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {

                    float pmom = static_cast<float>(inprop.pmom[i]);
                    float pcosth = static_cast<float>(inprop.pcosth[i]);
                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                for(size_t j = 0; j < inmodel.model_functions.size(); ++j) {
                    float oscw = inmodel.model_functions[j](phys, le);
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, hist_w * oscw * inmodel.hists[j](i, k));
                    }
                }
            }
        }
        else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){

                float oscw  = phys.size() != 0 ? 
                    inmodel.model_functions[inprop.model_rule[i]](phys, inprop.trueLE[i]) :
                    1;	
                float add_w = inprop.added_weights[i];
                float hist_w = 1.0 ;

                //Figure out what subchannel the event is in
                size_t subchan = inconfig.GetSubchannelIndexFromGlobalTrueBin(inprop.true_bin_indices[i]);
                std::string name = inconfig.m_fullnames[subchan];

                //Put name for ICARUS study here. How to handle more generically?
                if (name == "nu_ICARUS_numu_numucc") {
                    float pmom = static_cast<float>(inprop.pmom[i]);
                    float pcosth = static_cast<float>(inprop.pcosth[i]);

                    for (size_t j = 0; j<inweighthists.size(); ++j){
                        TH2D h = *inweighthists[j];
                        int bin = h.FindBin(pmom,pcosth);
                        hist_w *= h.GetBinContent(bin);
                    }
                }

                float finalw = oscw * add_w * hist_w;
                myspectrum.Fill(inprop.bin_indices[i], finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, int other_index) {
        int nbins = other_index < 0 ? inconfig.m_num_bins_total : inconfig.m_num_other_bins_total[other_index],
            nbins_collapsed = other_index < 0 ? inconfig.m_num_bins_total_collapsed : inconfig.m_num_other_bins_total_collapsed[other_index];
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(nbins, 0);
        Eigen::VectorXf cvspec = Eigen::VectorXf::Constant(nbins, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::random_device rd{};
        static std::mt19937 rng{rd()};
        std::normal_distribution<float> d;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(nbins_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(int i = 0; i < nbins_collapsed; i++)
            throwC(i) = d(rng);


        if(other_index < 0) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, throws[j], i);
                }
                for(int k = 0; k < nbins; ++k) {
                    spec(k) += systw * inprop.hist(i, k);
                    cvspec(k) += inprop.hist(i, k);
                }
            }
        } else {
            for(size_t i = 0; i<inprop.trueLE.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                const int true_bin = inprop.true_bin_indices[i]; 
                float systw = 1;
                for(size_t j = 0; j < throws.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, throws[j], true_bin);
                }
                float finalw = systw * add_w;
                if(inprop.other_bin_indices[i][other_index] >= 0) {
                    spec(inprop.other_bin_indices[i][other_index]) += finalw;
                    cvspec(inprop.other_bin_indices[i][other_index]) += add_w;
                }
            }
        }

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = other_index < 0 ? CollapseMatrix(inconfig, spec) : CollapseMatrix(inconfig, spec, other_index);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        Eigen::MatrixXf decomp_cov = insyst.DecomposeFractionalCovariance(inconfig, cvspec);
        Eigen::VectorXf collapsed_spec = other_index < 0 ? CollapseMatrix(inconfig, spec) : CollapseMatrix(inconfig, spec, other_index);
        Eigen::VectorXf final_spec = collapsed_spec + decomp_cov * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }
};
