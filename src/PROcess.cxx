#include "PROcess.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"

#include <Eigen/Eigen>

#include <random>
#include <string>
#include <vector>

namespace PROfit {
    PROspec FillCVSpectrum(const PROconfig &inconfig, const PROpeller &inprop, bool binned){

        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(size_t i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float systw = 1;
                for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                    myspectrum.Fill(k, systw * inprop.hist(i, k));
                }
            }
        } else {
            for(size_t i = 0; i<inprop.truth.size(); ++i){

                float add_w = inprop.added_weights[i]; 
                myspectrum.Fill(inprop.bin_indices[i], add_w);
            }
        }
        return myspectrum;

    }

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc *inosc, const std::vector<float> &inshifts, const std::vector<float> &physparams, bool binned){

        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float systw = 1;
                for(size_t j = 0; j < inshifts.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, inshifts[j], i);
                }
                if(physparams.size() != 0) {
                    for(size_t j = 0; j < inosc->model_functions.size(); ++j) {
                        float oscw = GetOscWeight(j, le, *inosc, physparams);
                        for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                            myspectrum.Fill(k, systw * oscw * inosc->hists[j](i, k));
                        }
                    }
                } else {
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, systw * inprop.hist(i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.truth.size(); ++i){

                float oscw  = physparams.size() != 0 ? GetOscWeight(i, inprop, *inosc, physparams) : 1;
                float add_w = inprop.added_weights[i]; 
                const int true_bin = inprop.true_bin_indices[i]; 
                
                float systw = 1;
                for(size_t j = 0; j < inshifts.size(); ++j) {
                    systw *= insyst.GetSplineShift(j, inshifts[j], true_bin);
                }

                float finalw = oscw * systw * add_w;

                myspectrum.Fill(inprop.bin_indices[i], finalw);
            }
        }
        return myspectrum;

    }

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const PROsc *inosc, const std::map<std::string, float> &inshifts, const std::vector<float> &physparams, bool binned){

        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float le = inprop.histLE[i];
                float systw = 1;
                for(const auto &[name, shift]: inshifts) {
                    systw *= insyst.GetSplineShift(name, shift, i);
                }
                if(physparams.size() != 0) {
                    for(size_t j = 0; j < inosc->model_functions.size(); ++j) {
                        float oscw = GetOscWeight(j, le, *inosc, physparams);
                        for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                            myspectrum.Fill(k, systw * oscw * inosc->hists[j](i, k));
                        }
                    }
                } else {
                    for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                        myspectrum.Fill(k, systw * inprop.hist(i, k));
                    }
                }
            }
        } else {
            for(size_t i = 0; i<inprop.truth.size(); ++i){

                float oscw  = physparams.size() != 0 ? GetOscWeight(i, inprop, *inosc, physparams) : 1;
                float add_w = inprop.added_weights[i]; 
                const int true_bin = inprop.true_bin_indices[i]; 
                
                float systw = 1;
                for(const auto &[name, shift]: inshifts) {
                    systw *= insyst.GetSplineShift(name, shift, true_bin);
                }

                float finalw = oscw * systw * add_w;

                myspectrum.Fill(inprop.bin_indices[i], finalw);
            }
        }
        return myspectrum;

    }

    PROspec FillRecoSpectra(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, const std::map<std::string, float> &inshifts, bool binned) {

        PROspec myspectrum(inconfig.m_num_bins_total);

        if(binned) {
            for(long int i = 0; i < inprop.hist.rows(); ++i) {
                float systw = 1;
                for(const auto &[name, shift]: inshifts) {
                    systw *= insyst.GetSplineShift(name, shift, i);
                }
                for(size_t k = 0; k < myspectrum.GetNbins(); ++k) {
                    myspectrum.Fill(k, systw * inprop.hist(i, k));
                }
            }
        } else {
            for(size_t i = 0; i<inprop.truth.size(); ++i){
                float add_w = inprop.added_weights[i]; 
                const int true_bin = inprop.true_bin_indices[i]; 
                float systw = 1;
                for(const auto &[name, shift]: inshifts) {
                    systw *= insyst.GetSplineShift(name, shift, true_bin);
                }
                float finalw = systw * add_w;
                myspectrum.Fill(inprop.bin_indices[i], finalw);
            }
        }
        return myspectrum;
    }

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst) {
        Eigen::VectorXd spec = Eigen::VectorXd::Constant(inconfig.m_num_bins_total, 0);
        Eigen::VectorXd cvspec = Eigen::VectorXd::Constant(inconfig.m_num_bins_total, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::random_device rd{};
        static std::mt19937 rng{rd()};
        std::normal_distribution<float> d;
        std::vector<float> throws;
        Eigen::VectorXd throwC = Eigen::VectorXd::Constant(inconfig.m_num_bins_total, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(size_t i = 0; i < inconfig.m_num_bins_total; i++)
            throwC(i) = d(rng);


        for(long int i = 0; i < inprop.hist.rows(); ++i) {
            float systw = 1;
            for(size_t j = 0; j < throws.size(); ++j) {
                systw *= insyst.GetSplineShift(j, throws[j], i);
            }
            for(size_t k = 0; k < inconfig.m_num_bins_total; ++k) {
                spec(k) += systw * inprop.hist(i, k);
                cvspec(k) += inprop.hist(i, k);
            }
        }

        // TODO: We probably just want to do this once and save it somewhere.
        // But where? PROpeller doesn't know about systs and PROsyst doesn't
        // know about cvspec.
        Eigen::MatrixXd diag = cvspec.asDiagonal();
        Eigen::MatrixXd full_cov = diag * insyst.fractional_covariance * diag;
        Eigen::LLT<Eigen::MatrixXd> llt(full_cov);

        Eigen::VectorXd final_spec = spec + llt.matrixL() * throwC;
        
        return PROspec(final_spec, final_spec.array().sqrt());
    }

    float GetOscWeight(int rule, float le, const PROsc &inosc, const std::vector<float> &inphysparams) {
        // The model functions take L and E separately, so give E as 1 and L as L/E
        return inosc.model_functions[rule](std::pow(10, inphysparams[0]), std::pow(10, inphysparams[1]), 1, le);
    }

    float GetOscWeight(int ev_idx, const PROpeller &inprop, const PROsc &inosc, const std::vector<float> &inphysparams) {

        //get subchannel here from pdg. this will be added when we agree on convention.
        //for now everything is numu disappearance 3+1. 
        // inphysparams[0] is log(delta-msq)
        // inphysparams[1] is sinsq2thmumu

//        float prob2 = inosc.Pmumu(std::pow(10, inphysparams[0]), inphysparams[1], inprop.truth[ev_idx], inprop.baseline[ev_idx]);
        float prob = inosc.model_functions[inprop.model_rule[ev_idx]](std::pow(10, inphysparams[0]), std::pow(10, inphysparams[1]), inprop.truth[ev_idx], inprop.baseline[ev_idx] );
        return prob;
    }

};
