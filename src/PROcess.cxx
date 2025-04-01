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

    PROspec FillSystRandomThrow(const PROconfig &inconfig, const PROpeller &inprop, const PROsyst &insyst, uint32_t seed) {
        Eigen::VectorXf spec = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);
        Eigen::VectorXf cvspec = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);

        // TODO: We should think about centralizing rng in a thread-safe/thread-aware way
        static std::mt19937 rng{seed};
        std::normal_distribution<float> d;
        std::vector<float> throws;
        //Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(inconfig.m_num_bins_total_collapsed, 0);
        for(size_t i = 0; i < insyst.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(size_t i = 0; i < inconfig.m_num_bins_total_collapsed; i++)
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

        if(insyst.GetNCovar() == 0) {
            Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec);
            return PROspec(final_spec, final_spec.array().sqrt());
        }

        // TODO: We probably just want to do this once and save it somewhere.
        // But where? PROpeller doesn't know about systs and PROsyst doesn't
        // know about cvspec.
        Eigen::MatrixXf diag = cvspec.asDiagonal();
        Eigen::MatrixXf full_cov = diag * insyst.fractional_covariance * diag;
        Eigen::MatrixXf coll = CollapseMatrix(inconfig, full_cov);
        Eigen::LDLT<Eigen::MatrixXf> ldlt(coll);
        Eigen::MatrixXf L = ldlt.matrixL(); 
        Eigen::VectorXf D_sqrt = ldlt.vectorD().array().sqrt();  
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(ldlt.transpositionsP());

        if (ldlt.info() != Eigen::Success) {
            log<LOG_ERROR>(L"%1% | Eigen LLT has failed!") % __func__ ;
            if (!coll.isApprox(coll.transpose())) {
                log<LOG_ERROR>(L"%1% | Matrix is not symmetric!") % __func__ ;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(coll);
            if (eigensolver.eigenvalues().minCoeff() <= 0) {
                log<LOG_ERROR>(L"%1% | Matrix is not positive semi definite, minCoeff is %2% ") % __func__ % eigensolver.eigenvalues().minCoeff();
            }
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", "\n", "", "", "", "");
            std::ostringstream oss;
            oss << coll.format(fmt);
            log<LOG_ERROR>(L"%1% | Matrix is %2% ") % __func__ % oss.str().c_str();
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf final_spec = CollapseMatrix(inconfig, spec) + P*L*D_sqrt.asDiagonal() * throwC;

        //std::vector<float> stdVec(final_spec.data(), final_spec.data() + final_spec.size());
        //log<LOG_INFO>(L"%1% | final_spec is %2% ") % __func__ % stdVec;


        return PROspec(final_spec, final_spec.array().sqrt());
    }
};
