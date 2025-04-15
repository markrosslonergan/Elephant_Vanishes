#ifndef PROPLOT_H
#define PROPLOT_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iomanip>
// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"
#include "PROspec.h"

// Root includes
#include "TAttLine.h"
#include "TAttMarker.h"
#include "THStack.h"
#include "TStyle.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TRatioPlot.h"
#include "TPaveText.h"
#include "TTree.h"

namespace PROfit{


    //some helper functions for PROplot
    std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int other_index = -1);
    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);
    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);
    std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale = false, int other_index = -1);
    template<class T, class P>
    std::unique_ptr<TGraphAsymmErrors> getMCMCErrorBand(Metropolis<T, P> met, size_t burnin, size_t iterations, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, Eigen::MatrixXf &post_covar, bool scale = false);

    enum class PlotOptions {
        Default = 0,
        CVasStack = 1 << 0,
        AreaNormalized = 1 << 1,
        BinWidthScaled = 1 << 2,
        DataMCRatio = 1 << 3,
        DataPostfitRatio = 1 << 4,
    };

    PlotOptions operator|(PlotOptions a, PlotOptions b) {
        return static_cast<PlotOptions>(static_cast<int>(a) | static_cast<int>(b));
    }

    PlotOptions operator|=(PlotOptions &a, PlotOptions b) {
        return a = a | b;
    }

    PlotOptions operator&(PlotOptions a, PlotOptions b) {
        return static_cast<PlotOptions>(static_cast<int>(a) & static_cast<int>(b));
    }

    PlotOptions operator&=(PlotOptions &a, PlotOptions b) {
        return a = a & b;
    }

    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, TPaveText *text, PlotOptions opt = PlotOptions::Default, int other_index = -1);


};

#endif
