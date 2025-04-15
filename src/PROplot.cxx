#include "PROplot.h"

namespace PROfit{


    std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec &spec, const PROconfig& inconfig, bool scale, int other_index) {
        std::map<std::string, std::unique_ptr<TH1D>> hists;  

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                        const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                        int rcolor = color == "NONE" ? kRed - 7 : inconfig.HexToROOTColor(color);
                        std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(spec.toTH1D(inconfig, global_subchannel_index, other_index));
                        htmp->SetLineWidth(1);
                        htmp->SetLineColor(kBlack);
                        htmp->SetFillColor(rcolor);
                        if(scale) htmp->Scale(1,"width");
                        hists[subchannel_name] = std::move(htmp);

                        log<LOG_DEBUG>(L"%1% || Printot %2% %3% %4% %5% %6% : Integral %7% ") % __func__ % global_channel_index % global_subchannel_index % subchannel_name.c_str() % sc % ic % hists[subchannel_name]->Integral();
                        ++global_subchannel_index;
                    }//end subchan
                    ++global_channel_index;
                }//end chan
            }//end det
        }//end mode
        return hists;
    }

    std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv) {
        std::map<std::string, std::unique_ptr<TH2D>> ret;
        Eigen::MatrixXf fractional_cov = syst.fractional_covariance;
        Eigen::MatrixXf diag = cv.Spec().array().matrix().asDiagonal(); 
        Eigen::MatrixXf full_covariance =  diag*fractional_cov*diag;
        Eigen::MatrixXf collapsed_full_covariance =  CollapseMatrix(config,full_covariance);  
        Eigen::VectorXf collapsed_cv = CollapseMatrix(config, cv.Spec());
        Eigen::MatrixXf collapsed_cv_inv_diag = collapsed_cv.asDiagonal().inverse();
        Eigen::MatrixXf collapsed_frac_cov = collapsed_cv_inv_diag * collapsed_full_covariance * collapsed_cv_inv_diag;

        std::unique_ptr<TH2D> cov_hist = std::make_unique<TH2D>("cov", "Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
        std::unique_ptr<TH2D> collapsed_cov_hist = std::make_unique<TH2D>("ccov", "Collapsed Fractional Covariance Matrix;Bin # ;Bin #", config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed, config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed);

        std::unique_ptr<TH2D> cor_hist = std::make_unique<TH2D>("cor", "Correlation Matrix;Bin # ;Bin #", config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
        std::unique_ptr<TH2D> collapsed_cor_hist = std::make_unique<TH2D>("ccor", "Collapsed Correlation Matrix;Bin # ;Bin #", config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed, config.m_num_bins_total_collapsed, 0, config.m_num_bins_total_collapsed);

        for(size_t i = 0; i < config.m_num_bins_total; ++i)
            for(size_t j = 0; j < config.m_num_bins_total; ++j){
                cov_hist->SetBinContent(i+1,j+1,fractional_cov(i,j));
                cor_hist->SetBinContent(i+1,j+1,fractional_cov(i,j)/(sqrt(fractional_cov(i,i))*sqrt(fractional_cov(j,j))));
            }

        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i)
            for(size_t j = 0; j < config.m_num_bins_total_collapsed; ++j){
                collapsed_cov_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j));
                collapsed_cor_hist->SetBinContent(i+1,j+1,collapsed_frac_cov(i,j)/(sqrt(collapsed_frac_cov(i,i))*sqrt(collapsed_frac_cov(j,j))));
            }

        ret["total_frac_cov"] = std::move(cov_hist);
        ret["collapsed_total_frac_cov"] = std::move(collapsed_cov_hist);
        ret["total_cor"] = std::move(cor_hist);
        ret["collapsed_total_cor"] = std::move(collapsed_cor_hist);

        for(const auto &name: syst.covar_names) {
            const Eigen::MatrixXf &covar = syst.GrabMatrix(name);
            const Eigen::MatrixXf &corr = syst.GrabCorrMatrix(name);

            std::unique_ptr<TH2D> cov_h = std::make_unique<TH2D>(("cov"+name).c_str(), (name+" Fractional Covariance;Bin # ;Bin #").c_str(), config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
            std::unique_ptr<TH2D> corr_h = std::make_unique<TH2D>(("cor"+name).c_str(), (name+" Correlation;Bin # ;Bin #").c_str(), config.m_num_bins_total, 0, config.m_num_bins_total, config.m_num_bins_total, 0, config.m_num_bins_total);
            for(size_t i = 0; i < config.m_num_bins_total; ++i){
                for(size_t j = 0; j < config.m_num_bins_total; ++j){
                    cov_h->SetBinContent(i+1,j+1,covar(i,j));
                    corr_h->SetBinContent(i+1,j+1,corr(i,j));
                }
            }

            ret[name+"_cov"] = std::move(cov_h);
            ret[name+"_corr"] = std::move(corr_h);
        }

        return ret;
    }

    std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> 
        getSplineGraphs(const PROsyst &systs, const PROconfig &config) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs;

            for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                const std::string &name = systs.spline_names[i];
                const PROsyst::Spline &spline = systs.GrabSpline(name);
                //using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;
                std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>> bin_graphs;
                size_t nbins = 
                    systs.spline_binnings[i] == -2 ? config.m_num_truebins_total :
                    systs.spline_binnings[i] == -1 ? config.m_num_bins_total
                    : config.m_num_other_bins_total[systs.spline_binnings[i]];
                bin_graphs.reserve(nbins);

                for(size_t j = 0; j < nbins; ++j) {
                    const std::vector<std::pair<float, std::array<float, 4>>> &spline_for_bin = spline[j];
                    std::unique_ptr<TGraph> curve = std::make_unique<TGraph>();
                    std::unique_ptr<TGraph> fixed_pts = std::make_unique<TGraph>();
                    for(size_t k = 0; k < spline_for_bin.size(); ++k) {
                        //const auto &[lo, coeffs] = spline_for_bin[k];
                        float lo = spline_for_bin[k].first;
                        std::array<float, 4> coeffs = spline_for_bin[k].second;
                        float hi = k < spline_for_bin.size() - 1 ? spline_for_bin[k+1].first : systs.spline_hi[i];
                        auto fn = [coeffs](float shift){
                            return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
                        };
                        fixed_pts->SetPoint(fixed_pts->GetN(), lo, fn(0)); 
                        if(k == spline_for_bin.size() - 1)
                            fixed_pts->SetPoint(fixed_pts->GetN(), hi, fn(hi - lo));
                        float width = (hi - lo) / 20;
                        for(size_t l = 0; l < 20; ++l)
                            curve->SetPoint(curve->GetN(), lo + l * width, fn(l * width));
                    }
                    bin_graphs.push_back(std::make_pair(std::move(fixed_pts), std::move(curve)));
                }
                spline_graphs[name] = std::move(bin_graphs);
            }

            return spline_graphs;
        }

    std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale, int other_index) {
        //TODO: Only works with 1 mode/detector/channel
        Eigen::VectorXf cv = other_index < 0 ? CollapseMatrix(config, FillCVSpectrum(config, prop, true).Spec()) :
            CollapseMatrix(config, FillOtherCVSpectrum(config, prop, other_index).Spec(), other_index);
        std::vector<float> edges = other_index < 0 ? config.GetChannelBinEdges(0) : config.GetChannelOtherBinEdges(0, other_index);
        log<LOG_DEBUG>(L"%1% || For other var %2% the cv is %3% and the edges are %4%")
            % __func__ % other_index % cv % edges;
        std::vector<float> centers;
        size_t nerrorsample = 5000;
        for(size_t i = 0; i < edges.size() - 1; ++i)
            centers.push_back((edges[i+1] + edges[i])/2);
        std::vector<Eigen::VectorXf> specs;
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        for(size_t i = 0; i < nerrorsample; ++i)
            specs.push_back(FillSystRandomThrow(config, prop, syst, dseed(PROseed::global_rng), other_index).Spec());
        //specs.push_back(CollapseMatrix(config, FillSystRandomThrow(config, prop, syst).Spec()));
        TH1D tmphist("th", "", cv.size(), edges.data());
        for(int i = 0; i < cv.size(); ++i)
            tmphist.SetBinContent(i+1, cv(i));
        if(scale) tmphist.Scale(1, "width");
        //std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(cv.size(), centers.data(), cv.data());
        std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(&tmphist);
        for(int i = 0; i < cv.size(); ++i) {
            std::vector<float> binconts(nerrorsample);
            for(size_t j = 0; j < nerrorsample; ++j) {
                binconts[j] = specs[j](i);
            }
            float scale_factor = tmphist.GetBinContent(i+1)/cv(i);
            if(std::isnan(scale_factor)) scale_factor = 1;
            std::sort(binconts.begin(), binconts.end());
            float ehi = std::abs((binconts[5*840] - cv(i))*scale_factor);
            float elo = std::abs((cv(i) - binconts[5*160])*scale_factor);
            ret->SetPointEYhigh(i, ehi);
            ret->SetPointEYlow(i, elo);

            log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% %7%") % __func__ % i % cv(i) % ehi % elo % scale_factor % tmphist.GetBinContent(i+1);
        }
        return ret;
    }

    template<class T, class P>
        std::unique_ptr<TGraphAsymmErrors> getMCMCErrorBand(Metropolis<T,P> mh, size_t burnin, size_t iterations, const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, Eigen::MatrixXf &post_covar, bool scale) {
            for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                posteriors.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);

            Eigen::VectorXf cv = FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true).Spec();
            Eigen::MatrixXf L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
            std::normal_distribution<float> nd;
            Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_bins_total_collapsed, 0);

            int nspline = metric.GetSysts().GetNSplines();
            int nphys = metric.GetModel().nparams;
            Eigen::VectorXf splines_bf = best_fit.segment(nphys, nspline);
            post_covar = Eigen::MatrixXf::Constant(nspline, nspline, 0);
            size_t accepted = 0;
            std::vector<Eigen::VectorXf> specs;
            const auto action = [&](const Eigen::VectorXf &value) {
                accepted += 1;
                for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i)
                    throws(i) = nd(PROseed::global_rng);
                specs.push_back(CollapseMatrix(config, FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true).Spec())+L*throws);
                for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
                    posteriors[i].Fill(value(i+nphys));
                Eigen::VectorXf splines = value.segment(nphys, nspline);
                Eigen::VectorXf diff = splines-splines_bf;
                post_covar += diff * diff.transpose();
            };
            mh.run(burnin, iterations, action);
            post_covar /= accepted;

            //TODO: Only works with 1 mode/detector/channel
            cv = CollapseMatrix(config, cv);
            std::vector<float> edges = config.GetChannelBinEdges(0);
            std::vector<float> centers;
            for(size_t i = 0; i < edges.size() - 1; ++i)
                centers.push_back((edges[i+1] + edges[i])/2);
            TH1D tmphist("th", "", cv.size(), edges.data());
            for(int i = 0; i < cv.size(); ++i)
                tmphist.SetBinContent(i+1, cv(i));
            if(scale) tmphist.Scale(1, "width");
            std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(&tmphist);
            for(int i = 0; i < cv.size(); ++i) {
                std::vector<float> binconts(specs.size());
                for(size_t j = 0; j < specs.size(); ++j) {
                    binconts[j] = specs[j](i);
                }
                float scale_factor = tmphist.GetBinContent(i+1)/cv(i);
                if(std::isnan(scale_factor)) scale_factor = 1;
                std::sort(binconts.begin(), binconts.end());
                float ehi = std::abs((binconts[0.84*specs.size()] - cv(i))*scale_factor);
                float elo = std::abs((cv(i) - binconts[0.16*specs.size()])*scale_factor);
                ret->SetPointEYhigh(i, ehi);
                ret->SetPointEYlow(i, elo);
                log<LOG_DEBUG>(L"%1% || ErrorBand bin %2% %3% %4% %5% %6% %7%") % __func__ % i % cv(i) % ehi % elo % scale_factor % tmphist.GetBinContent(i+1);
            }
            return ret;
        }

    void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, TPaveText *text, PlotOptions opt, int other_index) {
        TCanvas c;
        c.Print((filename+"[").c_str());

        std::map<std::string, std::unique_ptr<TH1D>> cvhists;
        if(cv) cvhists = getCVHists(*cv, config, (bool)(opt & PlotOptions::BinWidthScaled), other_index);

        Eigen::VectorXf bf_spec;
        if(best_fit) {
            bf_spec = other_index < 0 ? CollapseMatrix(config, best_fit->Spec()) : CollapseMatrix(config, best_fit->Spec(), other_index);
        }

        std::string ytitle = bool(opt&PlotOptions::AreaNormalized)
            ? "Area Normalized"
            : bool(opt&PlotOptions::BinWidthScaled) 
            ? "Events/GeV" 
            : "Events";

        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t mode = 0; mode < config.m_num_modes; ++mode) {
            for(size_t det = 0; det < config.m_num_detectors; ++det) {
                for(size_t channel = 0; channel < config.m_num_channels; ++channel) {
                    size_t channel_nbins = other_index < 0 ? config.m_channel_num_bins[channel] : config.m_channel_num_other_bins[channel][other_index];
                    std::vector<float> edges = other_index < 0 ? config.GetChannelBinEdges(0) : config.GetChannelOtherBinEdges(0, other_index);
                    std::string xtitle = other_index < 0 ? config.m_channel_units[channel] : config.m_channel_other_units[channel][other_index];
                    std::string hist_title = config.m_channel_plotnames[channel]+";"+xtitle+";"+ytitle;
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    TH1D cv_hist(std::to_string(global_channel_index).c_str(), hist_title.c_str(), channel_nbins, edges.data());
                    cv_hist.SetLineWidth(3);
                    cv_hist.SetLineColor(kBlue);
                    cv_hist.SetFillStyle(0);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        cv_hist.SetBinContent(bin+1, 0);
                    }

                    // Set up TPads for ratios, unused if ratio option not chosen
                    TPad p1("p1", "p1", 0, 0.25, 1, 1);
                    p1.SetBottomMargin(0);

                    TPad p2("p2", "p2", 0, 0, 1, 0.25);
                    p2.SetTopMargin(0);
                    p2.SetBottomMargin(0.3);

                    THStack *cvstack = NULL;
                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) cvstack = new THStack(std::to_string(global_channel_index).c_str(), hist_title.c_str());
                        for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            if(bool(opt&PlotOptions::CVasStack)) {
                                cvstack->Add(cvhists[subchannel_name].get());
                                leg->AddEntry(cvhists[subchannel_name].get(), config.m_subchannel_plotnames[channel][subchannel].c_str() ,"f");
                            }
                            cv_hist.Add(cvhists[subchannel_name].get());
                            ++global_subchannel_index;
                        }
                        if(bool(opt&PlotOptions::AreaNormalized)) {
                            float integral = cv_hist.Integral();
                            cv_hist.Scale(1 / integral);
                            if(bool(opt&PlotOptions::CVasStack)) {
                                TList *stlists = (TList*)cvstack->GetHists();
                                for(const auto&& obj: *stlists){
                                    ((TH1*)obj)->Scale(1/integral);
                                }
                            }
                        }
                    }

                    TGraphAsymmErrors *channel_errband = NULL;
                    if(errband) {
                        channel_errband = new TGraphAsymmErrors(&cv_hist);
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized)) {
                                scale = channel_errband->GetPointY(bin) / (*errband)->GetPointY(bin+channel_start);
                            }
                            channel_errband->SetPointEYhigh(bin, scale*(*errband)->GetErrorYhigh(bin+channel_start));
                            channel_errband->SetPointEYlow(bin, scale*(*errband)->GetErrorYlow(bin+channel_start));
                        }
                        channel_errband->SetFillColor(kGray+2);
                        //channel_errband->SetFillColorAlpha(kGray, 0.35);
                        channel_errband->SetFillStyle(3002);
                        channel_errband->SetLineColor(kGray+2);
                        channel_errband->SetLineWidth(1);
                        leg->AddEntry(channel_errband, "#pm 1#sigma", "f");
                    }

                    TH1D bf_hist(("bf"+std::to_string(global_channel_index)).c_str(), hist_title.c_str(), channel_nbins, edges.data());
                    if(best_fit) {
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            bf_hist.SetBinContent(bin+1, bf_spec(bin+channel_start));
                        }
                        bf_hist.SetLineColor(kGreen);
                        bf_hist.SetLineWidth(3);
                        leg->AddEntry(&bf_hist, "Best Fit", "l");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            bf_hist.Scale(1.0/bf_hist.Integral());
                    }

                    TGraphAsymmErrors *post_channel_errband = NULL;
                    if(posterrband) {
                        post_channel_errband = new TGraphAsymmErrors(&bf_hist);
                        int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                        for(size_t bin = 0; bin < channel_nbins; ++bin) {
                            float scale = 1.0;
                            if(bool(opt&PlotOptions::AreaNormalized)) {
                                scale = post_channel_errband->GetPointY(bin) / (*posterrband)->GetPointY(bin+channel_start);
                            }
                            post_channel_errband->SetPointEYhigh(bin, scale*(*posterrband)->GetErrorYhigh(bin+channel_start));
                            post_channel_errband->SetPointEYlow(bin, scale*(*posterrband)->GetErrorYlow(bin+channel_start));
                        }
                        post_channel_errband->SetFillColor(kRed);
                        post_channel_errband->SetFillStyle(3354);
                        post_channel_errband->SetLineColor(kRed);
                        post_channel_errband->SetLineWidth(1);
                        leg->AddEntry(post_channel_errband, "post-fit #pm 1#sigma", "f");
                    }

                    TH1D data_hist;
                    if(data) {
                        data_hist = data->toTH1D(config, global_channel_index, other_index);
                        data_hist.SetLineColor(kBlack);
                        data_hist.SetLineWidth(2);
                        data_hist.SetMarkerStyle(kFullCircle);
                        data_hist.SetMarkerColor(kBlack);
                        data_hist.SetMarkerSize(1);
                        leg->AddEntry(&data_hist, "Data", "pe");
                        if(bool(opt&PlotOptions::BinWidthScaled))
                            data_hist.Scale(1, "width");
                        if(bool(opt&PlotOptions::AreaNormalized))
                            data_hist.Scale(1.0/data_hist.Integral());
                    }

                    /*******************/
                    /* Draw everything */
                    /*******************/

                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio))
                        p1.cd();

                    if(cv) {
                        if(bool(opt&PlotOptions::CVasStack)) {
                            cvstack->SetMaximum(1.2*cvstack->GetMaximum());
                            cvstack->Draw("hist");
                        } else {
                            cv_hist.SetMaximum(1.2*cv_hist.GetMaximum());
                            leg->AddEntry(&cv_hist, "CV", "l");
                            cv_hist.Draw("hist");
                        }
                    }

                    if(errband) channel_errband->Draw("2 same");

                    if(best_fit) {
                        if(cv) bf_hist.Draw("hist same");
                        else bf_hist.Draw("hist");
                    }

                    if(posterrband) post_channel_errband->Draw("2 same");

                    if(data) {
                        if(cv || best_fit) data_hist.Draw("PE1 same");
                        else data_hist.Draw("E1P");
                    }

                    if(text) {
                        text->Draw("same");
                    }

                    leg->Draw("same");

                    TH1D *ratio, *one;
                    TGraphAsymmErrors *ratio_err;
                    if(bool(opt&PlotOptions::DataMCRatio) || bool(opt&PlotOptions::DataPostfitRatio)) {
                        p2.cd();

                        std::string y_title = bool(opt&PlotOptions::DataMCRatio) ? "data/MC" : "data/Best Fit";
                        ratio = new TH1D(("rat"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        one = new TH1D(("one"+std::to_string(global_channel_index)).c_str(), (";"+xtitle+";"+y_title).c_str(), channel_nbins, edges.data());
                        ratio_err = new TGraphAsymmErrors(); 
                        *ratio_err = bool(opt&PlotOptions::DataMCRatio)
                            ? *channel_errband
                            : *post_channel_errband;

                        one->GetYaxis()->SetTitleSize(0.1);
                        one->GetYaxis()->SetLabelSize(0.1);
                        one->GetXaxis()->SetTitleSize(0.1);
                        one->GetXaxis()->SetLabelSize(0.1);
                        one->GetYaxis()->SetTitleOffset(0.5);

                        for(size_t i = 0; i < channel_nbins; ++i) {
                            float numerator = data_hist.GetBinContent(i+1);
                            float denonminator = 
                                bool(opt&PlotOptions::DataMCRatio)
                                ? cv_hist.GetBinContent(i+1)
                                : bf_hist.GetBinContent(i+1);
                            float rat = numerator/denonminator;
                            if(isnan(rat)) rat = 1;
                            ratio->SetBinError(i+1, 1.0 / sqrt(numerator));
                            ratio->SetBinContent(i+1, rat);
                            one->SetBinContent(i+1, 1.0);
                            ratio_err->SetPointEYhigh(i, ratio_err->GetErrorYhigh(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointEYlow(i, ratio_err->GetErrorYlow(i)/ratio_err->GetPointY(i));
                            ratio_err->SetPointY(i, 1.0);
                        }


                        one->SetMaximum(1.5);
                        one->SetMinimum(0.5);
                        one->SetLineColor(kBlack);
                        one->SetLineStyle(kDashed);
                        one->Draw("hist");

                        ratio->SetLineColor(kBlack);
                        ratio->SetLineWidth(2);
                        ratio->SetMarkerStyle(kFullCircle);
                        ratio->SetMarkerColor(kBlack);
                        ratio->SetMarkerSize(1);

                        //ratio_err->SetFillColor(kRed);
                        //ratio_err->SetFillStyle(3345);
                        ratio_err->Draw("2 same");

                        ratio->Draw("PE1 same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();
                    }

                    c.Print(filename.c_str());

                    ++global_channel_index;
                }
            }
        }

        c.Print((filename+"]").c_str());
    }



};
