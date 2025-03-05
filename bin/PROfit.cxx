#include "PROconfig.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROcess.h"
#include "PROsurf.h"
#include "PROfitter.h"
#include "PROmodel.h"
#include "PROtocall.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include "TStyle.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TRatioPlot.h"

#include <Eigen/Eigen>

#include <filesystem>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace PROfit;

log_level_t GLOBAL_LEVEL = LOG_DEBUG;

//some helper functions for PROplot
std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false);
std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);
std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);
std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale = false);

int main(int argc, char* argv[])
{
    gStyle->SetOptStat(0);
    CLI::App app{"PROfit, a PROfessional, PROductive fitting and oscillation framework. Together let's minimize PROfit!"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    std::string analysis_tag = "PROfit";
    std::string chi2 = "PROchi";
    bool eventbyevent=false;
    bool shapeonly = false;//not coded yet
    bool rateonly = false;//not coded yet
    bool force = false;
    size_t nthread = 1;
    size_t maxevents;

    bool with_splines = false, binwidth_scale = false;

    std::vector<float> osc_params;
    std::map<std::string, float> injected_systs;
    std::vector<std::string> syst_list, systs_excluded;

    float xlo, xhi, ylo, yhi;
    std::array<float, 2> xlims, ylims;
    std::vector<int> grid_size;
    bool statonly = false, logx=true, logy=true;
    std::string xlabel, ylabel;

    std::string reweights_file;
    std::vector<std::string> mockreweights;
    std::vector<TH2D*> weighthists;

    //Global Arguments for all PROfit enables subcommands.
    app.add_option("-x,--xml", xmlname, "Input PROfit XML configuration file.")->required();
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(GLOBAL_LEVEL);
    app.add_option("-t,--tag", analysis_tag, "Analysis Tag used for output identification.")->default_str("PROfit");
    app.add_option("-n, --nthread",   nthread, "Number of threads to parallelize over.")->default_val(1);
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-c, --chi2", chi2, "Which chi2 function to use. Options are PROchi or PROCNP")->default_str("PROchi");
    app.add_option("--inject", osc_params, "Physics parameters to inject as true signal.")->expected(-1);// HOW TO
    app.add_option("--inject-systs", injected_systs, "Systematic shifts to inject. Map of name and shift value in sigmas. Only spline systs are supported right now.");
    app.add_option("--syst-list", syst_list, "Override list of systematics to use (note: all systs must be in the xml).");
    app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 
    app.add_option("-f, --rwfile", reweights_file, "File containing histograms for reweighting");
    app.add_option("-r, --mockrw",   mockreweights, "Vector of reweights to use for mock data");
    app.add_flag("--scale-by-width", binwidth_scale, "Scale histgrams by 1/(bin width).");
    app.add_flag("--event-by-event", eventbyevent, "Do you want to weight event-by-event?");
    app.add_flag("--statonly", statonly, "Run a stats only surface instead of fitting systematics");
    app.add_flag("--shapeonly", shapeonly, "Run a shape only analysis");
    app.add_flag("--rateonly", rateonly, "Run a rate only analysis");
    app.add_flag("--force",force,"Force loading binary data even if hash is incorrect (Be Careful!)");


    //PROcess, into binary data [Do this once first!]
    CLI::App *process_command = app.add_subcommand("process", "PROcess the MC and systematics in root files into binary data for future rapid loading.");

    //PROsurf, make a 2D surface scan of physics parameters
    CLI::App *surface_command = app.add_subcommand("surface", "Make a 2D surface scan of two physics parameters, profiling over all others.");
    surface_command->add_option("-g, --grid", grid_size, "Set grid size. If one dimension passed, grid assumed to be square, else rectangular")->expected(0, 2)->default_val(40);
    CLI::Option *xlim_opt = surface_command->add_option("--xlims", xlims, "Limits for x-axis");
    CLI::Option *ylim_opt = surface_command->add_option("--ylims", ylims, "Limits for y-axis");
    surface_command->add_option("--xlo", xlo, "Lower limit for x-axis")->excludes(xlim_opt)->default_val(1e-4);
    surface_command->add_option("--xhi", xhi, "Upper limit for x-axis")->excludes(xlim_opt)->default_val(1);
    surface_command->add_option("--ylo", ylo, "Lower limit for y-axis")->excludes(ylim_opt)->default_val(1e-2);
    surface_command->add_option("--yhi", yhi, "Upper limit for y-axis")->excludes(ylim_opt)->default_val(1e2);
    surface_command->add_option("--xlabel", xlabel, "X-axis label")->default_val("sin^{2}2#theta_{#mu#mu}");
    surface_command->add_option("--ylabel", ylabel, "Y-axis label")->default_val("#Deltam^{2}_{41}");
    surface_command->add_flag("--logx,!--linx", logx, "Specify if x-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--logy,!--liny", logy, "Specify if y-axis is logarithmic or linear (default log)");

    //PROfile, make N profile'd chi^2 for each physics and nuisence parameters
    CLI::App *profile_command = app.add_subcommand("profile", "Make a 1D profiled chi2 for each physics and nuisence parameter.");

    //PROplot, plot things
    CLI::App *proplot_command = app.add_subcommand("plot", "Make plots of CV, or injected point with error bars and covariance.");
    proplot_command->add_flag("--with-splines", with_splines, "Include graphs of splines in output.");

    //PROtest, test things
    CLI::App *protest_command = app.add_subcommand("protest", "Testing ground for rapid quick tests.");



    CLI11_PARSE(app, argc, argv);

    log<LOG_INFO>(L"%1% || PROfit commandline input arguments. xml: %2%, tag: %3%, nthread: %4% ") % __func__ % xmlname.c_str() % analysis_tag.c_str() % nthread ;

    //Initilize configuration from the XML;
    PROconfig config(xmlname);

    //Inititilize PROpeller to keep MC
    PROpeller prop;

    //Initilize objects for systematics storage
    std::vector<SystStruct> systsstructs;

    //input/output logic
    std::string propBinName = analysis_tag+"_prop.bin";
    std::string systBinName = analysis_tag+"_syst.bin";

    if((*process_command) || (!std::filesystem::exists(systBinName) || !std::filesystem::exists(propBinName))  ){
        log<LOG_INFO>(L"%1% || Processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();
        //Process the CAF files to grab and fill all SystStructs and PROpeller
        PROcess_CAFAna(config, systsstructs, prop);
        prop.save(propBinName);    
        saveSystStructVector(systsstructs,systBinName);
        log<LOG_INFO>(L"%1% || Done processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();

        if(*process_command)return 0;
    }else{
        log<LOG_INFO>(L"%1% || Loading PROpeller and PROsysts from precalc binary input: %2%") % __func__ % propBinName.c_str();
        prop.load(propBinName);
        loadSystStructVector(systsstructs, systBinName);

        log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) are here. ") % __func__ %  config.hash % prop.hash % systsstructs[0].hash;
        if(config.hash!=prop.hash && config.hash!=systsstructs.front().hash){
            if(force){
                log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) not compatable! ") % __func__ %  config.hash % prop.hash % systsstructs.front().hash;
                log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
            }else{
                log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) not compatable! ") % __func__ %  config.hash % prop.hash % systsstructs.front().hash;
                return 1;
            }
        }
    }

    //Build a PROsyst to sort and analyze all systematics
    PROsyst systs(systsstructs);
    std::unique_ptr<PROmodel> model = get_model_from_string(config.m_model_tag, prop);

    if(syst_list.size()) {
        systs = systs.subset(syst_list);
    } else if(systs_excluded.size()) {
        systs = systs.excluding(systs_excluded);
    }


    Eigen::VectorXf pparams = Eigen::VectorXf::Constant(model->nparams + systs.GetNSplines(), 0);
    if(osc_params.size()) {
        if(osc_params.size() != model->nparams) {
            log<LOG_ERROR>(L"%1% || Incorrect number of physics parameters provided. Expected %2%, found %3%.")
                % __func__ % model->nparams % osc_params.size();
            exit(EXIT_FAILURE);
        }
        for(size_t i = 0; i < osc_params.size(); ++i) {
            pparams(i) = std::log10(osc_params[i]);
        }
    }

    Eigen::VectorXf allparams = Eigen::VectorXf::Constant(model->nparams + systs.GetNSplines(), 0);
    for(int i = 0; i < pparams.size(); ++i) allparams(i) = pparams(i);
    for(const auto& [name, shift]: injected_systs) {
        log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;
        auto it = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        if(it == systs.spline_names.end()) {
            log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
            continue;
        }
        int idx = std::distance(systs.spline_names.begin(), it);
        allparams(idx) = shift;
    }

    //Create CV or injected data spectrum for all subsequent steps
    //this now will inject osc param, splines and reweight all at once
    PROspec data = osc_params.size() || injected_systs.size() ? FillRecoSpectra(config, prop, systs, *model, allparams, !eventbyevent) :  FillCVSpectrum(config, prop, !eventbyevent);

    //Only for reweighting tests
    if (!mockreweights.empty()) {
        log<LOG_INFO>(L"%1% || Will use reweighted MC (with any requested oscillations) as data for this study") % __func__  ;
        log<LOG_INFO>(L"%1% || Any parameter shifts requested will be ignored (fix later?)") % __func__  ;
        auto file = std::make_unique<TFile>(reweights_file.c_str());
        log<LOG_DEBUG>(L"%1% || Set file to : %2% ") % __func__ % reweights_file.c_str();
        log<LOG_DEBUG>(L"%1% || Size of reweights vector : %2% ") % __func__ % mockreweights.size() ;
        for (size_t i=0; i < mockreweights.size(); ++i) {
            log<LOG_DEBUG>(L"%1% || Mock reweight i : %2% ") % __func__ % mockreweights[i].c_str() ;
            TH2D* rwhist = (TH2D*)file->Get(mockreweights[i].c_str());
            weighthists.push_back(rwhist);
            log<LOG_DEBUG>(L"%1% || Read in weight hist ") % __func__ ;      
        }
        data = FillWeightedSpectrumFromHist(config,prop,weighthists,*model, allparams,!eventbyevent);
    }

    Eigen::VectorXf data_vec = CollapseMatrix(config, data.Spec());
    Eigen::VectorXf err_vec_sq = data.Error().array().square();
    Eigen::VectorXf err_vec = CollapseMatrix(config, err_vec_sq).array().sqrt();
    data = PROspec(data_vec, err_vec);


    //Some global minimizer params
    LBFGSpp::LBFGSBParam<float> param;  
    param.epsilon = 1e-6;
    param.max_iterations = 100;
    param.max_linesearch = 250;
    param.delta = 1e-6;

    //Metric Time
    PROmetric *metric;
    if(chi2 == "PROchi") {
        metric = new PROchi("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
    } else if(chi2 == "PROCNP") {
        metric = new PROCNP("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
    } else {
        log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
        abort();
    }


    //***********************************************************************
    //***********************************************************************
    //******************** PROfile PROfile PROfile **************************
    //***********************************************************************
    //***********************************************************************

    if(*profile_command){


        size_t nparams = 2 + systs.GetNSplines();
        Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
        lb(0) = -2; lb(1) = -std::numeric_limits<float>::infinity();
        Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
        ub(0) = 2; ub(1) = 0;
        for(size_t i = 2; i < nparams; ++i) {
            lb(i) = systs.spline_lo[i-2];
            ub(i) = systs.spline_hi[i-2];
        }
        PROfitter fitter(ub, lb, param);

        float chi2 = fitter.Fit(*metric); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();

        //std::string hname = "#chi^{2}/ndf = " + to_string(chi2) + "/" + to_string(config.m_num_bins_total_collapsed);
        std::string hname = "";

        Eigen::VectorXf subvector1 = best_fit.segment(0, 2);
        std::vector<float> fitparams(subvector1.data(), subvector1.data() + subvector1.size());

        Eigen::VectorXf subvector2 = best_fit.segment(2, systs.GetNSplines());
        std::vector<float> shifts(subvector2.data(), subvector2.data() + subvector2.size());

        Eigen::VectorXf post_fit = CollapseMatrix(config, FillRecoSpectra(config, prop, systs, *model, best_fit, true).Spec());
        TH1D post_hist("ph", hname.c_str(), config.m_num_bins_total_collapsed, config.m_channel_bin_edges[0].data());
        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i) {
            post_hist.SetBinContent(i+1, post_fit(i));
        }

        PROfile(config, prop, systs, *model, data, *metric , analysis_tag+"_PROfile", true, nthread, best_fit);

        delete metric;
        return 0;

        //***********************************************************************
        //***********************************************************************
        //******************** PROsurf PROsurf PROsurf **************************
        //***********************************************************************
        //***********************************************************************
    }else if(*surface_command){

        if (grid_size.empty()) {
            grid_size = {40, 40};
        }
        if (grid_size.size() == 1) {
            grid_size.push_back(grid_size[0]); //make it square
        }

        if(*xlim_opt) {
            xlo = xlims[0];
            xhi = xlims[1];
        }
        if(*ylim_opt) {
            ylo = ylims[0];
            yhi = ylims[1];
        }

        //Define grid and Surface
        size_t nbinsx = grid_size[0], nbinsy = grid_size[1];
        PROsurf surface(*metric, 1, 0, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

        if(statonly)
            surface.FillSurfaceStat(config, analysis_tag+"_statonly_surface.txt");
        else
            surface.FillSurface(analysis_tag+"_surface.txt",nthread);

        std::vector<float> binedges_x, binedges_y;
        for(size_t i = 0; i < surface.nbinsx+1; i++)
            binedges_x.push_back(logx ? std::pow(10, surface.edges_x(i)) : surface.edges_x(i));
        for(size_t i = 0; i < surface.nbinsy+1; i++)
            binedges_y.push_back(logy ? std::pow(10, surface.edges_y(i)) : surface.edges_y(i));

        TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

        for(size_t i = 0; i < surface.nbinsx; i++) {
            for(size_t j = 0; j < surface.nbinsy; j++) {
                surf.SetBinContent(i+1, j+1, surface.surface(i, j));
            }
        }

        log<LOG_INFO>(L"%1% || Saving surface to %2% as TH2D named \"surf.\"") % __func__ % analysis_tag.c_str();
        TFile fout((analysis_tag+"_surf.root").c_str(), "RECREATE");
        surf.Write();

        TCanvas c;
        if(logy)
            c.SetLogy();
        if(logx)
            c.SetLogx();
        c.SetLogz();
        surf.Draw("colz");
        c.Print((analysis_tag+"_surface.pdf").c_str());
        delete metric;
        fout.Close();
        return 0;

        //***********************************************************************
        //***********************************************************************
        //******************** PROplot PROplot PROplot **************************
        //***********************************************************************
        //***********************************************************************
    }else if(*proplot_command){

        TCanvas c;


        c.Print((analysis_tag +"_PROplot_CV.pdf"+ "[").c_str(), "pdf");
        PROspec spec = FillCVSpectrum(config, prop, !eventbyevent);

        std::map<std::string, std::unique_ptr<TH1D>> cv_hists = getCVHists(spec, config, binwidth_scale);
        size_t global_subchannel_index = 0;
        size_t global_channel_index = 0;
        for(size_t im = 0; im < config.m_num_modes; im++){
            for(size_t id =0; id < config.m_num_detectors; id++){
                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                    std::unique_ptr<THStack> s = std::make_unique<THStack>((std::to_string(global_channel_index)).c_str(),(std::to_string(global_channel_index)).c_str());
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                        s->Add(cv_hists[subchannel_name].get());
                        leg->AddEntry(cv_hists[subchannel_name].get(), config.m_subchannel_plotnames[ic][sc].c_str() ,"f");
                        ++global_subchannel_index;
                    }

                    s->Draw("hist");
                    leg->Draw();

                    s->SetTitle((config.m_mode_names[im]  +" "+ config.m_detector_names[id]+" "+ config.m_channel_names[ic]).c_str());
                    s->GetXaxis()->SetTitle(config.m_channel_units[ic].c_str());
                    if(binwidth_scale)
                        s->GetYaxis()->SetTitle("Events/GeV");
                    else
                        s->GetYaxis()->SetTitle("Events");

                    c.Print((analysis_tag+"_PROplot_CV.pdf").c_str(), "pdf");
                }
            }
        }
        c.Print((analysis_tag+"_PROplot_CV.pdf" + "]").c_str(), "pdf");

        if(osc_params.size()) {

            c.Print((analysis_tag +"_PROplot_Osc.pdf"+ "[").c_str(), "pdf");

            PROspec osc_spec = FillRecoSpectra(config, prop, systs, *model, pparams, !eventbyevent);
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCVHists(osc_spec, config, binwidth_scale);
            size_t global_subchannel_index = 0;
            size_t global_channel_index = 0;
            for(size_t im = 0; im < config.m_num_modes; im++){
                for(size_t id =0; id < config.m_num_detectors; id++){
                    for(size_t ic = 0; ic < config.m_num_channels; ic++){
                        TH1D* osc_hist = NULL;
                        TH1D* cv_hist = NULL;
                        for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++){
                            const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                            const auto &h = cv_hists[subchannel_name];
                            const auto &o = osc_hists[subchannel_name];
                            if(sc == 0) {
                                cv_hist = (TH1D*)h->Clone();
                                osc_hist = (TH1D*)o->Clone();
                            } else {
                                cv_hist->Add(&*h);
                                osc_hist->Add(&*o);
                            }
                            ++global_subchannel_index;
                        }
                        if(binwidth_scale )
                            cv_hist->GetYaxis()->SetTitle("Events/GeV");
                        else
                            cv_hist->GetYaxis()->SetTitle("Events");
                        cv_hist->SetTitle((config.m_mode_names[im]  +" "+ config.m_detector_names[id]+" "+ config.m_channel_names[ic]).c_str());
                        cv_hist->GetXaxis()->SetTitle("");
                        cv_hist->SetLineColor(kBlack);
                        cv_hist->SetFillColor(kWhite);
                        cv_hist->SetFillStyle(0);
                        osc_hist->SetLineColor(kBlue);
                        osc_hist->SetFillColor(kWhite);
                        osc_hist->SetFillStyle(0);
                        cv_hist->SetLineWidth(3);
                        osc_hist->SetLineWidth(3);
                        TH1D *rat = (TH1D*)osc_hist->Clone();
                        rat->Divide(cv_hist);
                        rat->SetTitle("");
                        rat->GetYaxis()->SetTitle("Ratio");
                        TH1D *one = (TH1D*)rat->Clone();
                        one->Divide(one);
                        one->SetLineColor(kBlack);
                        one->GetYaxis()->SetTitle("Ratio");

                        std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                        leg->SetFillStyle(0);
                        leg->SetLineWidth(0);
                        leg->AddEntry(cv_hist, "No Oscillations", "l");
                        std::string oscstr = "";//"#splitline{Oscilations:}{";
                        for(int j=0;j<model->nparams;j++){
                            oscstr+=model->pretty_param_names[j]+ " : "+ to_string_prec(osc_params[j],2) + (j==0 ? ", " : "" );
                        }
                        //oscstr+="}";

                        leg->AddEntry(osc_hist, oscstr.c_str(), "l");

                        TPad p1("p1", "p1", 0, 0.25, 1, 1);
                        p1.SetBottomMargin(0);
                        p1.cd();
                        cv_hist->Draw("hist");
                        osc_hist->Draw("hist same");
                        leg->Draw("same");

                        TPad p2("p2", "p2", 0, 0, 1, 0.25);
                        p2.SetTopMargin(0);
                        p2.SetBottomMargin(0.3);
                        p2.cd();
                        one->GetYaxis()->SetTitleSize(0.1);
                        one->GetYaxis()->SetLabelSize(0.1);
                        one->GetXaxis()->SetTitleSize(0.1);
                        one->GetXaxis()->SetLabelSize(0.1);
                        one->GetYaxis()->SetTitleOffset(0.5);
                        one->Draw("hist");
                        one->SetMaximum(rat->GetMaximum()*1.2);
                        one->SetMinimum(rat->GetMinimum()*0.8);
                        rat->Draw("hist same");

                        c.cd();
                        p1.Draw();
                        p2.Draw();

                        c.Print((analysis_tag+"_PROplot_Osc.pdf").c_str(), "pdf");

                        delete cv_hist;
                        delete osc_hist;
                    }
                }
            }
            c.Print((analysis_tag+"_PROplot_Osc.pdf" + "]").c_str(), "pdf");
        }



        //Now some covariances
        std::map<std::string, std::unique_ptr<TH2D>> matrices = covarianceTH2D(systs, config, spec);
        c.Print((analysis_tag+"_PROplot_Covar.pdf" + "[").c_str(), "pdf");
        for(const auto &[name, mat]: matrices) {
            mat->Draw("colz");
            c.Print((analysis_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
        }
        c.Print((analysis_tag+"_PROplot_Covar.pdf" + "]").c_str(), "pdf");


        //errorband
        //
        c.Print((analysis_tag+"_PROplot_ErrorBand.pdf" + "[").c_str(), "pdf");
        global_subchannel_index = 0;
        global_channel_index = 0;
        for(size_t im = 0; im < config.m_num_modes; im++){
            for(size_t id =0; id < config.m_num_detectors; id++){
                for(size_t ic = 0; ic < config.m_num_channels; ic++){
                    std::unique_ptr<THStack> s = std::make_unique<THStack>((std::to_string(global_channel_index)).c_str(),(std::to_string(global_channel_index)).c_str());
                    std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                    leg->SetFillStyle(0);
                    leg->SetLineWidth(0);
                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; sc++){
                        const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                        s->Add(cv_hists[subchannel_name].get());
                        leg->AddEntry(cv_hists[subchannel_name].get(), config.m_subchannel_plotnames[ic][sc].c_str() ,"f");
                        ++global_subchannel_index;
                    }
                    std::unique_ptr<TGraphAsymmErrors> err_band = getErrorBand(config, prop, systs, binwidth_scale );
                    err_band->SetLineColor(kRed+1);                        
                    err_band->GetYaxis()->SetTitle("Events/GeV");
                    err_band->Draw("AP");
                    err_band->GetXaxis()->SetRangeUser(0.3, 3.0);


                    s->Draw("hist SAME");
                    leg->Draw("SAME");
                    err_band->SetTitle((config.m_mode_names[im]  +" "+ config.m_detector_names[id]+" "+ config.m_channel_names[ic]).c_str());
                    err_band->GetXaxis()->SetTitle(config.m_channel_units[ic].c_str());
                    TH1* dummy = new TH1F("", "", 1, 0, 1);
                    dummy->SetLineColor(kRed+1);
                    leg->AddEntry(dummy->Clone(), "Syst", "l");
                    err_band->Draw("SAME P");
                    c.Print((analysis_tag+"_PROplot_ErrorBand.pdf").c_str(), "pdf");
                }
            }
        }
        c.Print((analysis_tag+"_PROplot_ErrorBand.pdf" + "]").c_str(), "pdf");



        if (!mockreweights.empty()) {

            //stupid hack, must be a better way to do this
            //Set up options:
            std::vector<const char*> xlabel(4);
            xlabel[0] = "Reconstructed Neutrino Energy";
            xlabel[1] = "True Leading Proton Momentum";
            xlabel[2] = "True Leading Proton Cos(Theta)";
            xlabel[3] = "Check what variable you are plotting!";
            int xi;
            if (xmlname.find("standard") != std::string::npos) {
                xi = 0;
            }
            else if (xmlname.find("pmom") != std::string::npos) {
                xi = 1;
            }
            else if (xmlname.find("costh") != std::string::npos) {  
                xi = 2;
            }
            else {
                xi = 3;
            }

            TH1D hcv = spec.toTH1D(config,0);
            TH1D hmock = data.toTH1D(config,0);
            hcv.Scale(1, "width");
            hmock.Scale(1, "width");
            hcv.GetYaxis()->SetTitle("Events/GeV");
            hmock.GetYaxis()->SetTitle("Events/GeV");
            hcv.GetXaxis()->SetTitle(xlabel[xi]);
            hmock.GetXaxis()->SetTitle(xlabel[xi]);
            hcv.SetTitle("");
            hmock.SetTitle("");

            TCanvas *c2 = new TCanvas((analysis_tag+"_spec_cv").c_str(), (analysis_tag+"_spec_cv").c_str(), 800, 800);
            hcv.SetLineColor(kBlack);
            hmock.SetLineColor(5);
            hmock.SetFillColor(5);
            TRatioPlot * rp = new TRatioPlot(&hmock,&hcv);
            rp->Draw();
            rp->GetLowerRefGraph()->SetMarkerStyle(21);
            TGraphAsymmErrors *lowerGraph = dynamic_cast<TGraphAsymmErrors*>(rp->GetLowerRefGraph());
            if (lowerGraph) {
                int nPoints = lowerGraph->GetN();
                for (int i = 0; i < nPoints; i++) {
                    lowerGraph->SetPointError(i, 0, 0, 0, 0); // Set both x and y errors to zero
                }
            }
            std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.35,0.7,0.89,0.89);
            leg->SetFillStyle(0);
            leg->SetLineWidth(0);
            leg->AddEntry(&hcv,"CV","l");
            leg->AddEntry(&hmock,"Mock data: ", "f");
            TObject *null = new TObject(); 
            int i=0;

            for(const auto& [name, shift]: injected_systs) {
                char ns[6];
                snprintf(ns, sizeof(ns),"%.2f", shift);
                leg->AddEntry(null, (name+": "+ns+ " sigma").c_str(),"");
                i++;
            }

            for (const auto& m : mockreweights) {
                leg->AddEntry(null, m.c_str(),"");
                i++;
            }
            for (const auto& m : osc_params) {
                leg->AddEntry(null, ("param: "+std::to_string(m)).c_str(),"");
                i++;
            }

            leg->Draw();
            c2->SaveAs((analysis_tag+"_ReWeight_spec.pdf").c_str());


        }





        if(with_splines) {
            c.Print((analysis_tag+"_PROplot_Spline.pdf" + "[").c_str(), "pdf");


            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(systs, config);
            c.Clear();
            c.Divide(4,4);
            for(const auto &[syst_name, syst_bins]: spline_graphs) {
                int bin = 0;
                bool unprinted = true;
                for(const auto &[fixed_pts, curve]: syst_bins) {
                    unprinted = true;
                    c.cd(bin%16+1);
                    fixed_pts->SetMarkerColor(kBlack);
                    fixed_pts->SetMarkerStyle(kFullCircle);
                    fixed_pts->GetXaxis()->SetTitle("#sigma");
                    fixed_pts->GetYaxis()->SetTitle("Weight");
                    fixed_pts->SetTitle((syst_name+" - True Bin "+std::to_string(bin)).c_str());
                    fixed_pts->Draw("PA");
                    curve->Draw("C same");
                    ++bin;
                    if(bin % 16 == 0) {
                        c.Print((analysis_tag+"_PROplot_spline.pdf").c_str(), "pdf");
                        unprinted = false;
                    }
                }
                if(unprinted)
                    c.Print((analysis_tag+"_PROplot_spline.pdf").c_str(), "pdf");
            }

            c.Print((analysis_tag+"_PROplot_Spline.pdf" + "]").c_str(), "pdf");
        }

        //now onto root files
        TFile fout((analysis_tag+"_PROplot.root").c_str(), "RECREATE");

        fout.mkdir("CV_hists");
        fout.cd("CV_hists");
        for(const auto &[name, hist]: cv_hists) {
            hist->Write(name.c_str());
        }

        if((osc_params.size())) {
            PROspec osc_spec = FillRecoSpectra(config, prop, systs, *model, pparams, !eventbyevent);
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCVHists(osc_spec, config, binwidth_scale);
            fout.mkdir("Osc_hists");
            fout.cd("Osc_hists");
            for(const auto &[name, hist]: osc_hists) {
                hist->Write(name.c_str());
            }
        }

        fout.mkdir("Covariance");
        fout.cd("Covariance");
        for(const auto &[name, mat]: matrices)
            mat->Write(name.c_str());

        std::unique_ptr<TGraphAsymmErrors> err_band = getErrorBand(config, prop, systs, binwidth_scale );
        fout.mkdir("ErrorBand");
        fout.cd("ErrorBand");
        err_band->Write("err_band");

        if((with_splines)) {
            std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> spline_graphs = getSplineGraphs(systs, config);
            fout.mkdir("Splines");
            fout.cd("Splines");
            for(const auto &[name, syst_splines]: spline_graphs) {
                size_t bin = 0;
                for(const auto &[fixed_pts, curve]: syst_splines) {
                    fixed_pts->Write((name+"_fixedpts_"+std::to_string(bin)).c_str());
                    curve->Write((name+"_curve_"+std::to_string(bin)).c_str());
                    bin++;
                }
            }
        }

        fout.Close();
        delete metric;
        return 0;




        //***********************************************************************
        //***********************************************************************
        //******************** TEST AREA TEST AREA     **************************
        //***********************************************************************
        //***********************************************************************
    }else if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest. Place anything here, a playground for testing things .") % __func__;

        //***************************** END *********************************
    }else{
        log<LOG_WARNING>(L"%1% || Please pass a subcommand to tell PROfit to do something! see --help for ideas.") % __func__;
        return 1;
    }


    return 0;
}



//******************************** Functions to help plotting, move to a src later ************************************

std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec &spec, const PROconfig& inconfig, bool scale) {
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
                    std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(spec.toTH1D(inconfig, global_subchannel_index));
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
        bin_graphs.reserve(config.m_num_truebins_total);

        for(size_t j = 0; j < config.m_num_truebins_total; ++j) {
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

std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale) {
    //TODO: Only works with 1 mode/detector/channel
    Eigen::VectorXf cv = CollapseMatrix(config, FillCVSpectrum(config, prop, true).Spec());
    std::vector<float> edges = config.GetChannelBinEdges(0);
    std::vector<float> centers;
    for(size_t i = 0; i < edges.size() - 1; ++i)
        centers.push_back((edges[i+1] + edges[i])/2);
    std::vector<Eigen::VectorXf> specs;
    for(size_t i = 0; i < 1000; ++i)
        specs.push_back(FillSystRandomThrow(config, prop, syst).Spec());
    //specs.push_back(CollapseMatrix(config, FillSystRandomThrow(config, prop, syst).Spec()));
    TH1D tmphist("th", "", cv.size(), edges.data());
    for(size_t i = 0; i < cv.size(); ++i)
        tmphist.SetBinContent(i+1, cv(i));
    if(scale) tmphist.Scale(1, "width");
    //std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(cv.size(), centers.data(), cv.data());
    std::unique_ptr<TGraphAsymmErrors> ret = std::make_unique<TGraphAsymmErrors>(&tmphist);
    for(size_t i = 0; i < cv.size(); ++i) {
        std::array<float, 1000> binconts;
        for(size_t j = 0; j < 1000; ++j) {
            binconts[j] = specs[j](i);
        }
        float scale_factor = tmphist.GetBinContent(i+1)/cv(i);
        std::sort(binconts.begin(), binconts.end());
        float ehi = std::abs((binconts[840] - cv(i))*scale_factor);
        float elo = std::abs((cv(i) - binconts[160])*scale_factor);
        ret->SetPointEYhigh(i, ehi);
        ret->SetPointEYlow(i, elo);
    }
    return ret;
}

