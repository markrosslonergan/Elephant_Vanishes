#include "PROconfig.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROCNP.h"
#include "PROpoisson.h"
#include "PROcess.h"
#include "PROsurf.h"
#include "PROfitter.h"
#include "PROmodel.h"
#include "PROMCMC.h"
#include "PROtocall.h"
#include "PROseed.h"

#include "CLI11.h"
#include "LBFGSB.h"

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

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <LBFGSpp/Param.h>
#include <cmath>
#include <filesystem>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PROfit;

log_level_t GLOBAL_LEVEL = LOG_INFO;
std::wostream *OSTREAM = &wcout;

struct fc_out{
    float chi2_syst, chi2_osc, dmsq, sinsq2tmm;
    Eigen::VectorXf best_fit_syst, best_fit_osc, syst_throw;
};

struct fc_args {
    const size_t todo;
    std::vector<float>* dchi2s;
    std::vector<fc_out>* out;
    const PROconfig config;
    const PROpeller prop;
    const PROsyst systs;
    std::string chi2;
    const Eigen::VectorXf phy_params;
    const Eigen::MatrixXf L;
    LBFGSpp::LBFGSBParam<float> param;
    const int thread;
    const bool binned;
};

void fc_worker(fc_args args) {
    log<LOG_INFO>(L"%1% || FC for point %2%") % __func__ % args.phy_params;
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::unique_ptr<PROmodel> model = get_model_from_string(args.config.m_model_tag, args.prop);
    std::unique_ptr<PROmodel> null_model = std::make_unique<NullModel>(args.prop);

    PROchi::EvalStrategy strat = args.binned ? PROchi::BinnedChi2 : PROchi::EventByEvent;
    Eigen::VectorXf throws = Eigen::VectorXf::Constant(model->nparams + args.systs.GetNSplines(), 0);
    for(size_t i = 0; i < model->nparams; ++i) throws(i) = args.phy_params(i);
    size_t nparams = model->nparams + args.systs.GetNSplines();
    Eigen::VectorXf lb_osc = Eigen::VectorXf::Constant(nparams, -3.0);
    Eigen::VectorXf ub_osc = Eigen::VectorXf::Constant(nparams, 3.0);
    Eigen::VectorXf lb = Eigen::VectorXf::Constant(args.systs.GetNSplines(), -3.0);
    Eigen::VectorXf ub = Eigen::VectorXf::Constant(args.systs.GetNSplines(), 3.0);
    size_t nphys = model->nparams;
    //set physics to correct values
    for(size_t j=0; j<nphys; j++){
        ub_osc(j) = model->ub(j);
        lb_osc(j) = model->lb(j); 
    }
    //upper lower bounds for splines
    for(size_t j = nphys; j < nparams; ++j) {
        lb_osc(j) = args.systs.spline_lo[j-nphys];
        ub_osc(j) = args.systs.spline_hi[j-nphys];
        lb(j-nphys) = args.systs.spline_lo[j-nphys];
        ub(j-nphys) = args.systs.spline_hi[j-nphys];
    }
    for(size_t u = 0; u < args.todo; ++u) {
        log<LOG_INFO>(L"%1% | Thread #%2% Throw #%3%") % __func__ % args.thread % u;
        std::normal_distribution<float> d;
        Eigen::VectorXf throwC = Eigen::VectorXf::Constant(args.config.m_num_bins_total_collapsed, 0);
        for(size_t i = 0; i < args.systs.GetNSplines(); i++)
            throws(i+nphys) = d(rng);
        for(size_t i = 0; i < args.config.m_num_bins_total_collapsed; i++)
            throwC(i) = d(rng);
        PROspec shifted = FillRecoSpectra(args.config, args.prop, args.systs, *model, throws, strat);
        PROspec newSpec = PROspec::PoissonVariation(PROspec(CollapseMatrix(args.config, shifted.Spec()) + args.L * throwC, CollapseMatrix(args.config, shifted.Error())));
        PROdata data(newSpec.Spec(), newSpec.Error());
        //Metric Time
        PROmetric *metric, *null_metric;
        if(args.chi2 == "PROchi") {
            metric = new PROchi("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            null_metric = new PROchi("", args.config, args.prop, &args.systs, *null_model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        } else if(args.chi2 == "PROCNP") {
            metric = new PROCNP("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            null_metric = new PROCNP("", args.config, args.prop, &args.systs, *null_model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        } else if(args.chi2 == "Poisson") {
            metric = new PROpoisson("", args.config, args.prop, &args.systs, *model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
            null_metric = new PROpoisson("", args.config, args.prop, &args.systs, *null_model, data, !args.binned ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        } else {
            log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % args.chi2.c_str();
            abort();
        }

        // No oscillations
        std::uniform_int_distribution<uint32_t> dseed(0, std::numeric_limits<uint32_t>::max());
        PROfitter fitter(ub, lb, args.param, dseed(rng));
        float chi2_syst = fitter.Fit(*null_metric);

        // With oscillations
        PROfitter fitter_osc(ub_osc, lb_osc, args.param, dseed(rng));
        float chi2_osc = fitter_osc.Fit(*metric); 

        Eigen::VectorXf t = Eigen::VectorXf::Map(throws.data(), throws.size());

        args.out->push_back({
                chi2_syst, chi2_osc, 
                std::pow(10.0f, fitter_osc.best_fit(0)), std::pow(10.0f, fitter_osc.best_fit(1)), 
                fitter.best_fit, fitter_osc.best_fit.segment(2, nparams-2), t
        });

        args.dchi2s->push_back(std::abs(chi2_syst - chi2_osc ));
        delete metric;
        delete null_metric;
    }
}

//some helper functions for PROplot
std::map<std::string, std::unique_ptr<TH1D>> getCVHists(const PROspec & spec, const PROconfig& inconfig, bool scale = false, int other_index = -1);
std::map<std::string, std::unique_ptr<TH2D>> covarianceTH2D(const PROsyst &syst, const PROconfig &config, const PROspec &cv);
std::map<std::string, std::vector<std::pair<std::unique_ptr<TGraph>,std::unique_ptr<TGraph>>>> getSplineGraphs(const PROsyst &systs, const PROconfig &config);
std::unique_ptr<TGraphAsymmErrors> getErrorBand(const PROconfig &config, const PROpeller &prop, const PROsyst &syst, bool scale = false, int other_index = -1);
std::unique_ptr<TGraphAsymmErrors> getPostFitErrorBand(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, bool scale = false);
void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, bool plot_cv_stack, TPaveText *text, bool binwidth_scale = false, int other_index = -1);

int main(int argc, char* argv[])
{
    gStyle->SetOptStat(0);
    CLI::App app{"PROfit: a PROfessional, PROductive fitting and oscillation framework. Together let's minimize PROfit!"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    std::string data_xml = "";
    std::string analysis_tag = "PROfit";
    std::string output_tag = "v1";
    std::string chi2 = "PROchi";
    bool eventbyevent=false;
    bool shapeonly = false;
    bool rateonly = false;
    bool force = false;
    bool noxrootd = false;
    size_t nthread = 1;
    std::map<std::string, float> fit_options;
    size_t maxevents;
    int global_seed = -1;
    std::string log_file = "";

    bool with_splines = false, binwidth_scale = false;

    std::vector<float> osc_params;
    std::map<std::string, float> injected_systs;
    std::vector<std::string> syst_list, systs_excluded;

    bool systs_only_profile = false;

    float xlo, xhi, ylo, yhi;
    std::array<float, 2> xlims, ylims;
    std::vector<int> grid_size;
    bool statonly = false, logx=true, logy=true;
    std::string xlabel, ylabel;
    std::string xvar = "sinsq2thmm", yvar = "dmsq";
    bool run_brazil = false;
    bool statonly_brazil = false;
    bool single_brazil = false;
    bool only_brazil = false;
    std::vector<std::string> brazil_throws;

    std::string reweights_file;
    std::vector<std::string> mockreweights;
    std::vector<TH2D*> weighthists;

    size_t nuniv;

    //Global Arguments for all PROfit enables subcommands.
    app.add_option("-x,--xml", xmlname, "Input PROfit XML configuration file.")->required();
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4]->[Error,Warning,Info,Debug].")->default_val(GLOBAL_LEVEL);
    app.add_option("-t,--tag", analysis_tag, "Analysis Tag used for output identification.")->default_str("PROfit");
    app.add_option("-o,--output",output_tag,"Additional output filename quantifier")->default_str("v1");
    app.add_option("-n, --nthread",   nthread, "Number of threads to parallelize over.")->default_val(1);
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-c, --chi2", chi2, "Which chi2 function to use. Options are PROchi or PROCNP")->default_str("PROchi");
    app.add_option("-d, --data", data_xml, "Load from a seperate data xml/data file instead of signal injection. Only used with plot subcommand.")->default_str("");
    app.add_option("-i, --inject", osc_params, "Physics parameters to inject as true signal.")->expected(-1);// HOW TO
    app.add_option("-s, --seed", global_seed, "A global seed for PROseed rng. Default to -1 for hardware rng seed.")->default_val(-1);
    app.add_option("--inject-systs", injected_systs, "Systematic shifts to inject. Map of name and shift value in sigmas. Only spline systs are supported right now.");
    app.add_option("--syst-list", syst_list, "Override list of systematics to use (note: all systs must be in the xml).");
    app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 
    app.add_option("--fit-options", fit_options, "Parameters for LBFGSB.");
    app.add_option("-f, --rwfile", reweights_file, "File containing histograms for reweighting");
    app.add_option("-r, --mockrw",   mockreweights, "Vector of reweights to use for mock data");
    app.add_option("--log", log_file, "File to save log to. Warning: Will overwrite this file.");
    app.add_flag("--scale-by-width", binwidth_scale, "Scale histgrams by 1/(bin width).");
    app.add_flag("--event-by-event", eventbyevent, "Do you want to weight event-by-event?");
    app.add_flag("--statonly", statonly, "Run a stats only surface instead of fitting systematics");
    app.add_flag("--force",force,"Force loading binary data even if hash is incorrect (Be Careful!)");
    app.add_flag("--no-xrootd",noxrootd,"Do not use XRootD, which is enabled by default");
    auto* shape_flag = app.add_flag("--shapeonly", shapeonly, "Run a shape only analysis");
    auto* rate_flag = app.add_flag("--rateonly", rateonly, "Run a rate only analysis");
    shape_flag->excludes(rate_flag);   //PROcess, into binary data [Do this once first!]
    CLI::App *process_command = app.add_subcommand("process", "PROcess the MC and systematics in root files into binary data for future rapid loading.");

    //PROsurf, make a 2D surface scan of physics parameters
    CLI::App *surface_command = app.add_subcommand("surface", "Make a 2D surface scan of two physics parameters, profiling over all others.");
    surface_command->add_option("-g, --grid", grid_size, "Set grid size. If one dimension passed, grid assumed to be square, else rectangular")->expected(0, 2)->default_val(40);
    surface_command->add_option("--xvar", xvar, "Name of variable to put on x-axis")->default_val("sinsq2thmm");
    surface_command->add_option("--yvar", yvar, "Name of variable to put on x-axis")->default_val("dmsq");
    CLI::Option *xlim_opt = surface_command->add_option("--xlims", xlims, "Limits for x-axis");
    CLI::Option *ylim_opt = surface_command->add_option("--ylims", ylims, "Limits for y-axis");
    surface_command->add_option("--xlo", xlo, "Lower limit for x-axis")->excludes(xlim_opt)->default_val(1e-4);
    surface_command->add_option("--xhi", xhi, "Upper limit for x-axis")->excludes(xlim_opt)->default_val(1);
    surface_command->add_option("--ylo", ylo, "Lower limit for y-axis")->excludes(ylim_opt)->default_val(1e-2);
    surface_command->add_option("--yhi", yhi, "Upper limit for y-axis")->excludes(ylim_opt)->default_val(1e2);
    surface_command->add_option("--xlabel", xlabel, "X-axis label");
    surface_command->add_option("--ylabel", ylabel, "Y-axis label");
    surface_command->add_flag("--logx,!--linx", logx, "Specify if x-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--logy,!--liny", logy, "Specify if y-axis is logarithmic or linear (default log)");
    surface_command->add_flag("--brazil-band", run_brazil, "Run 1000 throws of stats+systs and draw 1 sigma and 2 sigma Brazil bands");
    surface_command->add_flag("--stat-throws", statonly_brazil, "Only do stat throws for the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--single-throw", single_brazil, "Only run a single iteration of the Brazil band")->needs("--brazil-band");
    surface_command->add_flag("--only-throw", only_brazil, "Only run Brazil band throws and not the nominal surface")->needs("--brazil-band");
    surface_command->add_option("--from-many", brazil_throws, "Make Brazil band from many provided throws")->needs("--brazil-band");

    //PROfile, make N profile'd chi^2 for each physics and nuisence parameters
    CLI::App *profile_command = app.add_subcommand("profile", "Make a 1D profiled chi2 for each physics and nuisence parameter.");
    profile_command->add_flag("--syst-only", systs_only_profile, "Profile over nuisance parameters only");

    //PROplot, plot things
    CLI::App *proplot_command = app.add_subcommand("plot", "Make plots of CV, or injected point with error bars and covariance.");
    proplot_command->add_flag("--with-splines", with_splines, "Include graphs of splines in output.");

    //PROfc, Feldmand-Cousins
    CLI::App *profc_command = app.add_subcommand("fc", "Run Feldman-Cousins for this injected signal");
    profc_command->add_option("-u,--universes", nuniv, "Number of Feldman Cousins universes to throw")->default_val(1000);

    //PROtest, test things
    CLI::App *protest_command = app.add_subcommand("protest", "Testing ground for rapid quick tests.");


    //Parse inputs. 
    CLI11_PARSE(app, argc, argv);

    std::wofstream log_out;
    if(log_file != "") {
        log_out.open(log_file);
        OSTREAM = &log_out;
    }

    log<LOG_INFO>(L" %1% ") % getIcon().c_str()  ;
    std::string final_output_tag =analysis_tag +"_"+output_tag;
    log<LOG_INFO>(L"%1% || PROfit commandline input arguments. xml: %2%, tag: %3%, output %4%, nthread: %5% ") % __func__ % xmlname.c_str() % analysis_tag.c_str() % output_tag.c_str() % nthread ;

    //Initilize configuration from the XML;
    PROconfig config(xmlname, rateonly);

    //Inititilize PROpeller to keep MC
    PROpeller prop;

    //Initilize objects for systematics storage
    std::vector<std::vector<SystStruct>> systsstructs;

    //input/output logic
    std::string propBinName = analysis_tag+"_prop.bin";
    std::string systBinName = analysis_tag+"_syst.bin";

    if((*process_command) || (!std::filesystem::exists(systBinName) || !std::filesystem::exists(propBinName))  ){
        log<LOG_INFO>(L"%1% || Processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();
        //Process the CAF files to grab and fill all SystStructs and PROpeller
        PROcess_CAFAna(config, systsstructs, prop,noxrootd);
        prop.save(propBinName);    
        saveSystStructVector(systsstructs,systBinName);
        log<LOG_INFO>(L"%1% || Done processing PROpeller and PROsysts from XML defined root files, and saving to binary output also: %2%") % __func__ % propBinName.c_str();

    }else{
        log<LOG_INFO>(L"%1% || Loading PROpeller and PROsysts from precalc binary input: %2%") % __func__ % propBinName.c_str();
        prop.load(propBinName);
        loadSystStructVector(systsstructs, systBinName);

        log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) are here. ") % __func__ %  config.hash % prop.hash % systsstructs[0][0].hash;
        if(config.hash!=prop.hash && config.hash!=systsstructs.front().front().hash){
            if(force){
                log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) not compatable! ") % __func__ %  config.hash % prop.hash % systsstructs.front().front().hash;
                log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
            }else{
                log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded PROpeller (%3%) or PROsyst hash(%4%) not compatable! ") % __func__ %  config.hash % prop.hash % systsstructs.front().front().hash;
                return 1;
            }
        }
    }


    //Build a PROsyst to sort and analyze all systematics
    PROsyst systs(prop, config, systsstructs.front(), shapeonly);
    std::vector<PROsyst> other_systs;
    for(size_t i = 0; i < config.m_num_other_vars; ++i)
        other_systs.emplace_back(prop, config, systsstructs.at(i+1), shapeonly, i);
    std::unique_ptr<PROmodel> model = get_model_from_string(config.m_model_tag, prop);
    std::unique_ptr<PROmodel> null_model = std::make_unique<NullModel>(prop);

    //Some eystematics might be ignored for this
    if(syst_list.size()) {
        systs = systs.subset(syst_list);
        for(PROsyst &syst: other_systs)
            syst = syst.subset(syst_list);
    } else if(systs_excluded.size()) {
        systs = systs.excluding(systs_excluded);
        for(PROsyst &syst: other_systs)
            syst = syst.excluding(systs_excluded);
    }

    //Pysics parameter input
        Eigen::VectorXf pparams = Eigen::VectorXf::Constant(model->nparams + systs.GetNSplines(), 0);
        if(osc_params.size()) {
            if(osc_params.size() != model->nparams) {
                log<LOG_ERROR>(L"%1% || Incorrect number of physics parameters provided. Expected %2%, found %3%.")
                    % __func__ % model->nparams % osc_params.size();
                exit(EXIT_FAILURE);
            }
            for(size_t i = 0; i < osc_params.size(); ++i) {
                pparams(i) = std::log10(osc_params[i]);
                //if(std::isinf(pparams(i))) pparams(i) = -10;
            }
        } else {
            for(size_t i = 0; i < model->nparams; ++i) {
                pparams(i) = model->default_val(i); 
            }
        }

        //Spline injection studies
        Eigen::VectorXf allparams = Eigen::VectorXf::Constant(model->nparams + systs.GetNSplines(), 0);
        Eigen::VectorXf systparams = Eigen::VectorXf::Constant(systs.GetNSplines(), 0);
        for(size_t i = 0; i < model->nparams; ++i) allparams(i) = pparams(i);
        for(const auto& [name, shift]: injected_systs) {
            log<LOG_INFO>(L"%1% || Injected syst: %2% shifted by %3%") % __func__ % name.c_str() % shift;
            auto it = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
            if(it == systs.spline_names.end()) {
                log<LOG_ERROR>(L"%1% || Error: Unrecognized spline %2%. Ignoring this injected shift.") % __func__ % name.c_str();
                continue;
            }
            int idx = std::distance(systs.spline_names.begin(), it);
            allparams(idx+model->nparams) = shift;
            systparams(idx) = shift;
        }

    //Some logic for EITHER injecting fake/mock data of oscillated signal/syst shifts OR using real data
    PROdata data;
    std::vector<PROdata> other_data;
    if(!data_xml.empty()){
        PROconfig dataconfig(data_xml);
        std::string dataBinName = analysis_tag+"_data.bin";
        for(size_t i = 0; i < dataconfig.m_num_channels; ++i) {
            size_t nsubch = dataconfig.m_num_subchannels[i];
            if(nsubch != 1) {
                log<LOG_ERROR>(L"%1% || Data xml required to have exactly 1 subchannel per channel. Found %2% for channel %3%")
                    % __func__ % nsubch % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
            std::string &subchname = dataconfig.m_subchannel_names[i][0];
            if(subchname != "data") {
                log<LOG_ERROR>(L"%1% || Data subchannel required to be called \"data.\" Found name %2% for channel %3%")
                    % __func__ % subchname.c_str() % i;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }
        }
        if(!PROconfig::SameChannels(config, dataconfig)) {
            log<LOG_ERROR>(L"%1% || Require data and MC to have same channels. A difference was found, check messages above.")
                % __func__;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        if((*process_command) || (!std::filesystem::exists(dataBinName))  ){
            log<LOG_INFO>(L"%1% || Processing Data Spectrum and saving to binary output also: %2%") % __func__ % dataBinName.c_str();

            //Process the CAF files to grab and fill spectrum directly
            std::vector<PROdata> alldata = CreatePROdata(dataconfig);
            PROdata::saveVector(dataconfig, alldata, dataBinName);
            data = alldata[0];
            //data.save(dataconfig,dataBinName);
            for(size_t io = 0; io < dataconfig.m_num_other_vars; ++io)
                other_data.push_back(alldata[io+1]);

            log<LOG_INFO>(L"%1% || Done processing Data from XML defined root files, and saving to binary output also: %2%") % __func__ % dataBinName.c_str();
        }else{
            log<LOG_INFO>(L"%1% || Loading Data from precalc binary input: %2%") % __func__ % dataBinName.c_str();
            //data.load(dataBinName);
            std::vector<PROdata> alldata;
            PROdata::loadVector(alldata, dataBinName);
            data = alldata[0];
            //data.save(dataconfig,dataBinName);
            for(size_t io = 0; io < dataconfig.m_num_other_vars; ++io)
                other_data.push_back(alldata[io+1]);

            log<LOG_INFO>(L"%1% || Done loading. Config hash (%2%) and binary loaded Data (%3%) hash are here. ") % __func__ %  dataconfig.hash % data.hash;
            if(dataconfig.hash!=data.hash){
                if(force){
                    log<LOG_WARNING>(L"%1% || WARNING config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    log<LOG_WARNING>(L"%1% || WARNING But we are forcing ahead, be SUPER clear and happy you understand what your doing.  ") % __func__;
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR config hash (%2%) and binary loaded data (%3%) hash not compatable! ") % __func__ %  dataconfig.hash % data.hash ;
                    return 1;
                }
            }
        }

    if(*profile_command || *surface_command || *protest_command){
                    log<LOG_ERROR>(L"%1% || ERROR --data can only be used with plot subcommand! ") % __func__  ;
                    return 1;
    }


    }//if no data, use injected or fake data;
    else{
        //Create CV or injected data spectrum for all subsequent steps
        //this now will inject osc param, splines and reweight all at once
        PROspec data_spec = osc_params.size() || injected_systs.size() ? FillRecoSpectra(config, prop, systs, *model, allparams, !eventbyevent) :  FillCVSpectrum(config, prop, !eventbyevent);

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
            data_spec = FillWeightedSpectrumFromHist(config, prop, weighthists, *model, allparams, !eventbyevent);
        }
        Eigen::VectorXf data_vec = CollapseMatrix(config, data_spec.Spec());
        Eigen::VectorXf err_vec_sq = data_spec.Error().array().square();
        Eigen::VectorXf err_vec = CollapseMatrix(config, err_vec_sq).array().sqrt();
        //data = PROdata(data_vec, err_vec);
        data = PROdata(data_vec, data_vec.array().sqrt());

        for(size_t io = 0; io < config.m_num_other_vars; ++io) {
            PROspec data_spec = osc_params.size() || injected_systs.size() 
                ? FillOtherRecoSpectra(config, prop, systs, *model, allparams, io)
                : FillOtherCVSpectrum(config, prop, io);

            Eigen::VectorXf data_vec = CollapseMatrix(config, data_spec.Spec(), io);
            Eigen::VectorXf err_vec_sq = data_spec.Error().array().square();
            Eigen::VectorXf err_vec = CollapseMatrix(config, err_vec_sq, io).array().sqrt();
            other_data.push_back(PROdata(data_vec, err_vec));
        }
    }

    //Seed time
    PROseed myseed(nthread, global_seed);
    
    //Some global minimizer params
    LBFGSpp::LBFGSBParam<float> param;  
    param.epsilon = 1e-6;
    param.max_iterations = 100;
    param.max_linesearch = 250;
    param.delta = 1e-6;
    for(const auto &[param_name, value]: fit_options) {
        if(param_name == "epsilon") {
            param.epsilon = value;
        } else if(param_name == "delta") {
            param.delta = value;
        } else if(param_name == "m") {
            param.m = value;
            if(value < 3) {
                log<LOG_WARNING>(L"%1% || Number of corrections to approximate inverse Hessian in"
                                 L" L-BFGS-B is recommended to be at least 3, provided value is %2%."
                                 L" Note: this is controlled via --fit-options m.")
                    % __func__ % value;
            }
        } else if(param_name == "epsilon_rel") {
            param.epsilon_rel = value;
        } else if(param_name == "past") {
            param.past = value;
            if(value == 0) {
                log<LOG_WARNING>(L"%1% || L-BFGS-B 'past' parameter set to 0. This will disable delta convergence test")
                    % __func__;
            }
        } else if(param_name == "max_iterations") {
            param.max_iterations = value;
        } else if(param_name == "max_submin") {
            param.max_submin = value;
        } else if(param_name == "max_linesearch") {
            param.max_linesearch = value;
        } else if(param_name == "min_step") {
            param.min_step = value;
            log<LOG_WARNING>(L"%1% || Modifying the minimum step size in the line search to be %2%."
                             L" This is not usually needed according to the LBFGSpp documentation.")
                % __func__ % value;
        } else if(param_name == "max_step") {
            param.max_step = value;
            log<LOG_WARNING>(L"%1% || Modifying the maximum step size in the line search to be %2%."
                             L" This is not usually needed according to the LBFGSpp documentation.")
                % __func__ % value;
        } else if(param_name == "ftol") {
            param.ftol = value;
        } else if(param_name == "wolfe") {
            param.wolfe = value;
        } else {
            log<LOG_WARNING>(L"%1% || Unrecognized LBFGSB parameter %2%. Will ignore.") 
                % __func__ % param_name.c_str();
        }
    }
    try {
        param.check_param();
    } catch(std::invalid_argument &except) {
        log<LOG_ERROR>(L"%1% || Invalid L-BFGS-B parameters: %2%") % __func__ % except.what();
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    //Metric Time
    PROmetric *metric, *null_metric;
    if(chi2 == "PROchi") {
        metric = new PROchi("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        null_metric = new PROchi("", config, prop, &systs, *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
    } else if(chi2 == "PROCNP") {
        metric = new PROCNP("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        null_metric = new PROCNP("", config, prop, &systs, *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
    } else if(chi2 == "Poisson") {
        metric = new PROpoisson("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
        null_metric = new PROpoisson("", config, prop, &systs, *null_model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
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

        PROmetric *metric_to_use = systs_only_profile ? null_metric : metric;
        size_t nparams = metric_to_use->GetModel().nparams + metric_to_use->GetSysts().GetNSplines();
        size_t nphys = metric_to_use->GetModel().nparams;
        Eigen::VectorXf lb = Eigen::VectorXf::Constant(nparams, -3.0);
        Eigen::VectorXf ub = Eigen::VectorXf::Constant(nparams, 3.0);
        for(size_t i = 0; i < nphys; ++i) {
            lb(i) = metric_to_use->GetModel().lb(i);
            ub(i) = metric_to_use->GetModel().ub(i);
        }
        for(size_t i = nphys; i < nparams; ++i) {
            lb(i) = metric_to_use->GetSysts().spline_lo[i-nphys];
            ub(i) = metric_to_use->GetSysts().spline_hi[i-nphys];
        }
        PROfitter fitter(ub, lb, param);

        float chi2 = fitter.Fit(*metric_to_use); 
        Eigen::VectorXf best_fit = fitter.best_fit;
        Eigen::MatrixXf post_covar = fitter.Covariance();

        std::string hname = "#chi^{2}/ndf = " + to_string(chi2) + "/" + to_string(config.m_num_bins_total_collapsed);
        PROspec cv = FillCVSpectrum(config, prop, true);
        PROspec bf = FillRecoSpectra(config, prop, metric_to_use->GetSysts(), metric_to_use->GetModel(), best_fit, true);
        TH1D post_hist("ph", hname.c_str(), config.m_num_bins_total_collapsed, config.m_channel_bin_edges[0].data());
        TH1D pre_hist("prh", hname.c_str(), config.m_num_bins_total_collapsed, config.m_channel_bin_edges[0].data());
        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i) {
            post_hist.SetBinContent(i+1, bf.Spec()(i));
            pre_hist.SetBinContent(i+1, cv.Spec()(i));
        }
        std::vector<TH1D> posteriors;
        std::unique_ptr<TGraphAsymmErrors> err_band = getErrorBand(config, prop, systs);
        std::unique_ptr<TGraphAsymmErrors> post_err_band = getPostFitErrorBand(config, prop, *metric_to_use, best_fit, posteriors);
        
        TPaveText chi2text(0.59, 0.50, 0.89, 0.59, "NDC");
        chi2text.AddText(hname.c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        plot_channels((final_output_tag+"_PROfile_hists.pdf"), config, cv, bf, data, err_band.get(), post_err_band.get(), false, &chi2text);

        TCanvas c;
        c.Print((final_output_tag+"_postfit_posteriors.pdf[").c_str());
        for(auto &h: posteriors) {
            h.Draw("hist");
            c.Print((final_output_tag+"_postfit_posteriors.pdf").c_str());
        }
        c.Print((final_output_tag+"_postfit_posteriors.pdf]").c_str());

        PROfile profile(config, metric_to_use->GetSysts(), metric_to_use->GetModel(), *metric_to_use, myseed, param, 
                final_output_tag+"_PROfile", !systs_only_profile, nthread, best_fit,
                systs_only_profile ? systparams : allparams);
        TFile fout((final_output_tag+"_PROfile.root").c_str(), "RECREATE");
        profile.onesig.Write("one_sigma_errs");
        pre_hist.Write("cv");
        err_band->Write("prefit_errband");
        post_err_band->Write("postfit_errband");
        post_hist.Write("best_fit");

        //***********************************************************************
        //***********************************************************************
        //******************** PROsurf PROsurf PROsurf **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*surface_command){

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
        size_t xaxis_idx = 1, yaxis_idx = 0;
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), xvar); loc != model->param_names.end()) {
            xaxis_idx = std::distance(model->param_names.begin(), loc);
        } else if(const auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), xvar); loc != systs.spline_names.end()) {
            xaxis_idx = std::distance(systs.spline_names.begin(), loc);
        }
        if(const auto loc = std::find(model->param_names.begin(), model->param_names.end(), yvar); loc != model->param_names.end()) {
            yaxis_idx = std::distance(model->param_names.begin(), loc);
        } else if(const auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), yvar); loc != systs.spline_names.end()) {
            yaxis_idx = std::distance(systs.spline_names.begin(), loc);
        }
        size_t nbinsx = grid_size[0], nbinsy = grid_size[1];
        PROsurf surface(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

        if(!only_brazil) {
            if(statonly)
                surface.FillSurfaceStat(config, param, final_output_tag+"_statonly_surface.txt");
            else
                surface.FillSurface(param, final_output_tag+"_surface.txt",myseed,nthread);
        }

        std::vector<float> binedges_x, binedges_y;
        for(size_t i = 0; i < surface.nbinsx+1; i++)
            binedges_x.push_back(logx ? std::pow(10, surface.edges_x(i)) : surface.edges_x(i));
        for(size_t i = 0; i < surface.nbinsy+1; i++)
            binedges_y.push_back(logy ? std::pow(10, surface.edges_y(i)) : surface.edges_y(i));

        if(xlabel == "") 
            xlabel = xaxis_idx < model->nparams ? model->pretty_param_names[xaxis_idx] : 
                config.m_mcgen_variation_plotname_map[systs.spline_names[xaxis_idx]];
        if(ylabel == "") 
            ylabel = yaxis_idx < model->nparams ? model->pretty_param_names[yaxis_idx] : 
                config.m_mcgen_variation_plotname_map[systs.spline_names[yaxis_idx]];
        TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

        for(size_t i = 0; i < surface.nbinsx; i++) {
            for(size_t j = 0; j < surface.nbinsy; j++) {
                surf.SetBinContent(i+1, j+1, surface.surface(i, j));
            }
        }

        log<LOG_INFO>(L"%1% || Saving surface to %2% as TH2D named \"surf.\"") % __func__ % final_output_tag.c_str();
        TFile fout((final_output_tag+"_surf.root").c_str(), "RECREATE");
        if(!only_brazil) {
            surf.Write();
            float chisq;
            int xbin, ybin;
            std::map<std::string, float> best_fit;
            TTree tree("tree", "BestFitTree");
            tree.Branch("chi2", &chisq); 
            tree.Branch("xbin", &xbin); 
            tree.Branch("ybin", &ybin); 
            tree.Branch("best_fit", &best_fit); 

            for(const auto &res: surface.results) {
                chisq = res.chi2;
                xbin = res.binx;
                ybin = res.biny;
                // If all fit points fail
                if(!res.best_fit.size()) { tree.Fill(); continue; }
                for(size_t i = 0; i < model->nparams; ++i) {
                    best_fit[model->param_names[i]] = res.best_fit(i);
                }
                for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                    best_fit[systs.spline_names[i]] = res.best_fit(i + model->nparams);
                }
                tree.Fill();
            }
            // TODO: Should we save the spectra as TH1s?

            tree.Write();

            TCanvas c;
            if(logy)
                c.SetLogy();
            if(logx)
                c.SetLogx();
            c.SetLogz();
            surf.Draw("colz");
            c.Print((final_output_tag+"_surface.pdf").c_str());
        }

        std::vector<PROsurf> brazil_band_surfaces;
        if(run_brazil && brazil_throws.size() == 0) {
            std::mt19937 rng(global_seed);
            std::normal_distribution<float> d;
            size_t nphys = metric->GetModel().nparams;
            PROspec cv = FillCVSpectrum(config, prop, true);
            PROspec collapsed_cv = PROspec(CollapseMatrix(config, cv.Spec()), CollapseMatrix(config, cv.Error()));
            Eigen::MatrixXf L = metric->GetSysts().DecomposeFractionalCovariance(config, cv.Spec());
            for(size_t i = 0; i < 1000; ++i) {
                Eigen::VectorXf throwp = pparams;
                Eigen::VectorXf throwC = Eigen::VectorXf::Constant(config.m_num_bins_total_collapsed, 0);
                for(size_t i = 0; i < metric->GetSysts().GetNSplines(); i++)
                    throwp(i+nphys) = d(rng);
                for(size_t i = 0; i < config.m_num_bins_total_collapsed; i++)
                    throwC(i) = d(rng);
                PROspec shifted = FillRecoSpectra(config, prop, metric->GetSysts(), metric->GetModel(), throwp, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                PROspec newSpec = statonly_brazil ? PROspec::PoissonVariation(collapsed_cv) :
                    PROspec::PoissonVariation(PROspec(CollapseMatrix(config, shifted.Spec()) + L * throwC, CollapseMatrix(config, shifted.Error())));
                PROdata data(newSpec.Spec(), newSpec.Error());
                PROmetric *metric;
                if(chi2 == "PROchi") {
                    metric = new PROchi("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "PROCNP") {
                    metric = new PROCNP("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else if(chi2 == "Poisson") {
                    metric = new PROpoisson("", config, prop, &systs, *model, data, eventbyevent ? PROmetric::EventByEvent : PROmetric::BinnedChi2);
                } else {
                    log<LOG_ERROR>(L"%1% || Unrecognized chi2 function %2%") % __func__ % chi2.c_str();
                    abort();
                }

                brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                        nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

                if(statonly)
                    brazil_band_surfaces.back().FillSurfaceStat(config, param, "");
                else
                    brazil_band_surfaces.back().FillSurface(param, "", myseed, nthread);

                TH2D surf("surf", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

                for(size_t i = 0; i < surface.nbinsx; i++) {
                    for(size_t j = 0; j < surface.nbinsy; j++) {
                        surf.SetBinContent(i+1, j+1, brazil_band_surfaces.back().surface(i, j));
                    }
                }
                surf.Write(("brazil_throw_surf_"+std::to_string(i)).c_str());

                // WARNING: Metric reference stored in surface. DO NOT USE IT AFTER THIS POINT.
                delete metric;
                if(single_brazil) break;
            }
        } else if(run_brazil) { // if brazil_thows.size() > 0
                for(const std::string &in: brazil_throws) {
                    brazil_band_surfaces.emplace_back(*metric, xaxis_idx, yaxis_idx, nbinsx, logx ? PROsurf::LogAxis : PROsurf::LinAxis, xlo, xhi,
                            nbinsy, logy ? PROsurf::LogAxis : PROsurf::LinAxis, ylo, yhi);

                    TFile fin(in.c_str());
                    // TODO: Check that axes and labels are the same
                    TH2D *surf = fin.Get<TH2D>("surf");
                    if(!surf) {
                        log<LOG_ERROR>(L"%1% || Could not find a TH2D called 'surf' in the file %2%. Terminating.")
                            % __func__ % in.c_str();
                        return EXIT_FAILURE;
                    }
                    for(size_t i = 0; i < surface.nbinsx; ++i) {
                        for(size_t j = 0; j < surface.nbinsy; ++j) {
                            brazil_band_surfaces.back().surface(i,j) = surf->GetBinContent(i,j);
                        }
                    }
                }
        }

        if(run_brazil && !single_brazil) {
            TH2D surf16("surf16", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf84("surf84", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf98("surf98", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf02("surf02", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());
            TH2D surf50("surf50", (";"+xlabel+";"+ylabel).c_str(), surface.nbinsx, binedges_x.data(), surface.nbinsy, binedges_y.data());

            for(size_t i = 0; i < surface.nbinsx; ++i) {
                for(size_t j = 0; j < surface.nbinsy; ++j) {
                    std::vector<float> values;
                    for(const auto &bbsurf: brazil_band_surfaces)
                        values.push_back(bbsurf.surface(i,j));
                    std::sort(values.begin(), values.end());
                    surf02.SetBinContent(i+1, j+1, values[(size_t)(0.023 * values.size())]);
                    surf16.SetBinContent(i+1, j+1, values[(size_t)(0.159 * values.size())]);
                    surf50.SetBinContent(i+1, j+1, values[(size_t)(0.500 * values.size())]);
                    surf84.SetBinContent(i+1, j+1, values[(size_t)(0.841 * values.size())]);
                    surf98.SetBinContent(i+1, j+1, values[(size_t)(0.977 * values.size())]);
                }
            }
            
            fout.cd();
            surf02.Write();
            surf16.Write();
            surf50.Write();
            surf84.Write();
            surf98.Write();
        }

        //***********************************************************************
        //***********************************************************************
        //******************** PROplot PROplot PROplot **************************
        //***********************************************************************
        //***********************************************************************
    }
    if(*proplot_command){
        PROspec spec = FillCVSpectrum(config, prop, !eventbyevent);
        plot_channels(final_output_tag+"_PROplot_CV.pdf", config, spec, {}, {}, {}, {}, true, NULL, binwidth_scale);
        std::vector<PROspec> other_cvs;
        for(size_t io = 0; io < config.m_num_other_vars; ++io) {
            other_cvs.push_back(FillOtherCVSpectrum(config, prop, io));
            plot_channels(final_output_tag+"_other_"+std::to_string(io)+"_PROplot_CV.pdf", config, other_cvs.back(), {}, {}, {}, {}, true, NULL, binwidth_scale, io);
        }

        std::map<std::string, std::unique_ptr<TH1D>> cv_hists = getCVHists(spec, config, binwidth_scale);
        std::vector<std::map<std::string, std::unique_ptr<TH1D>>> other_hists;
        for(size_t io = 0; io < config.m_num_other_vars; ++io) {
            other_hists.push_back(getCVHists(other_cvs[io], config, binwidth_scale, io));
        }

        TCanvas c;
        if(osc_params.size()) {

            c.Print((final_output_tag +"_PROplot_Osc.pdf"+ "[").c_str(), "pdf");

            PROspec osc_spec = FillRecoSpectra(config, prop, systs, *model, pparams, !eventbyevent);
            std::map<std::string, std::unique_ptr<TH1D>> osc_hists = getCVHists(osc_spec, config, binwidth_scale);
            size_t global_subchannel_index = 0;
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
                        for(size_t j=0;j<model->nparams;j++){
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

                        c.Print((final_output_tag+"_PROplot_Osc.pdf").c_str(), "pdf");

                        delete cv_hist;
                        delete osc_hist;
                    }
                }
            }
            c.Print((final_output_tag+"_PROplot_Osc.pdf" + "]").c_str(), "pdf");
        }



        //Now some covariances
        std::map<std::string, std::unique_ptr<TH2D>> matrices;

        if(systs.GetNCovar()>0){
            matrices = covarianceTH2D(systs, config, spec);
            c.Print((final_output_tag+"_PROplot_Covar.pdf" + "[").c_str(), "pdf");
            for(const auto &[name, mat]: matrices) {
                mat->Draw("colz");
                c.Print((final_output_tag+"_PROplot_Covar.pdf").c_str(), "pdf");
            }
            c.Print((final_output_tag+"_PROplot_Covar.pdf" + "]").c_str(), "pdf");
        }

        //errorband
        //
        //TODO: Multiple channels
        int global_channel_index = 0;
        double chival = metric->getSingleChannelChi(global_channel_index);
        log<LOG_INFO>(L"%1% || On channel %2% the datamc chi^2/ndof is %3%/%4% .") % __func__ % global_channel_index % chival % config.m_channel_num_bins[global_channel_index];
        TPaveText chi2text(0.59, 0.50, 0.89, 0.59, "NDC");
        chi2text.AddText(("#chi^{2}/ndf = "+to_string_prec(chival,2)+"/"+std::to_string(config.m_channel_num_bins[global_channel_index])).c_str());
        chi2text.SetFillColor(0);
        chi2text.SetBorderSize(0);
        chi2text.SetTextAlign(12);
        std::unique_ptr<TGraphAsymmErrors> err_band = getErrorBand(config, prop, systs, binwidth_scale );
        plot_channels(final_output_tag+"_PROplot_ErrorBand.pdf", config, spec, {}, data, err_band.get(), {}, true, &chi2text, binwidth_scale);
        std::vector<std::unique_ptr<TGraphAsymmErrors>> other_err_bands;
        for(size_t io = 0; io < config.m_num_other_vars; ++io) {
            other_err_bands.push_back(getErrorBand(config, prop, other_systs[io], binwidth_scale, io));
            plot_channels(final_output_tag+"_PROplot_other_"+std::to_string(io)+"_ErrorBand.pdf", config, other_cvs[io], {}, other_data[io], 
                    other_err_bands.back().get(), {}, true, NULL, binwidth_scale, io);
        }

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

            TH1D hcv = spec.toTH1D_Collapsed(config,0);
            TH1D hmock = data.toTH1D(config,0);
            if(binwidth_scale){
                hcv.Scale(1, "width");
                hmock.Scale(1, "width");
            }
            hcv.GetYaxis()->SetTitle("Events/GeV");
            hmock.GetYaxis()->SetTitle("Events/GeV");
            hcv.GetXaxis()->SetTitle(xlabel[xi]);
            hmock.GetXaxis()->SetTitle(xlabel[xi]);
            hcv.SetTitle("");
            hmock.SetTitle("");

            TCanvas *c2 = new TCanvas((final_output_tag+"_spec_cv").c_str(), (final_output_tag+"_spec_cv").c_str(), 800, 800);
            hmock.SetLineColor(kBlack);
            hcv.SetLineColor(5);
            hcv.SetFillColor(5);
            TRatioPlot * rp = new TRatioPlot(&hcv,&hmock);
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
            leg->AddEntry(&hcv,"CV","f");
            leg->AddEntry(&hmock,"Mock data: ", "l");
            TObject *null = new TObject(); 

            for(const auto& [name, shift]: injected_systs) {
                char ns[6];
                snprintf(ns, sizeof(ns),"%.2f", shift);
                leg->AddEntry(null, (name+": "+ns+ " sigma").c_str(),"");
            }

            for (const auto& m : mockreweights) {
                leg->AddEntry(null, m.c_str(),"");
            }
            for (const auto& m : osc_params) {
                leg->AddEntry(null, ("param: "+std::to_string(m)).c_str(),"");
            }

            leg->Draw();
            c2->SaveAs((final_output_tag+"_ReWeight_spec.pdf").c_str());
        }

        if(with_splines) {
            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "[").c_str(), "pdf");

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
                        c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
                        unprinted = false;
                    }
                }
                if(unprinted)
                    c.Print((final_output_tag+"_PROplot_Spline.pdf").c_str(), "pdf");
            }

            c.Print((final_output_tag+"_PROplot_Spline.pdf" + "]").c_str(), "pdf");
        }

        //now onto root files
        TFile fout((final_output_tag+"_PROplot.root").c_str(), "RECREATE");

        fout.mkdir("CV_hists");
        fout.cd("CV_hists");
        for(const auto &[name, hist]: cv_hists) {
            hist->Write(name.c_str());
        }
        int io = 0;
        for(const auto &other: other_hists) {
            for(const auto &[name, hist]: other) {
                hist->Write(("other_"+std::to_string(io)+name).c_str());
            }
            io++;
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

        fout.mkdir("ErrorBand");
        fout.cd("ErrorBand");
        err_band->Write("err_band");
        io = 0;
        for(const auto &band: other_err_bands)
            band->Write(("other_"+std::to_string(io++)+"_err_band").c_str());


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
    }

    //***********************************************************************
    //***********************************************************************
    //********************     Feldman-Cousins    ***************************
    //***********************************************************************
    //***********************************************************************

    if(*profc_command) {
        size_t FCthreads = nthread > nuniv ? nuniv : nthread;
        Eigen::MatrixXf diag = FillCVSpectrum(config, prop, !eventbyevent).Spec().array().matrix().asDiagonal();
        Eigen::MatrixXf full_cov = diag * systs.fractional_covariance * diag;
        Eigen::LLT<Eigen::MatrixXf> llt(CollapseMatrix(config, full_cov));

        std::vector<std::vector<float>> dchi2s;
        dchi2s.reserve(FCthreads);
        std::vector<std::vector<fc_out>> outs;
        outs.reserve(FCthreads);
        std::vector<std::thread> threads;
        size_t todo = nuniv/FCthreads;
        size_t addone = FCthreads - nuniv%FCthreads;
        for(size_t i = 0; i < nthread; i++) {
            dchi2s.emplace_back();
            outs.emplace_back();
            fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), config, prop, systs, chi2, pparams, llt.matrixL(), param, (int)i, !eventbyevent};
            threads.emplace_back(fc_worker, args);
        }
        for(auto&& t: threads) {
            t.join();
        }

        {
            TFile fout((final_output_tag+"_FC.root").c_str(), "RECREATE");
            fout.cd();
            float chi2_osc, chi2_syst, best_dmsq, best_sinsq2t;
            std::map<std::string, float> best_systs_osc, best_systs, syst_throw;
            TTree tree("tree", "tree");
            tree.Branch("chi2_osc", &chi2_osc); 
            tree.Branch("chi2_syst", &chi2_syst); 
            tree.Branch("best_dmsq", &best_dmsq); 
            tree.Branch("best_sinsq2t", &best_sinsq2t); 
            tree.Branch("best_systs_osc", &best_systs_osc); 
            tree.Branch("best_systs", &best_systs); 
            tree.Branch("syst_throw", &syst_throw);

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    chi2_osc = fco.chi2_osc;
                    chi2_syst = fco.chi2_syst;
                    best_dmsq = fco.dmsq;
                    best_sinsq2t = fco.sinsq2tmm;
                    for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                        best_systs_osc[systs.spline_names[i]] = fco.best_fit_osc(i);
                        best_systs[systs.spline_names[i]] = fco.best_fit_syst(i);
                        syst_throw[systs.spline_names[i]] = fco.syst_throw(i);
                    }
                    tree.Fill();
                }
            }

            tree.Write();
        }
        {
            ofstream fcout(final_output_tag+"_FC.csv");
            fcout << "chi2_osc,chi2_syst,best_dmsq,best_sinsq2t";
            for(const std::string &name: systs.spline_names) {
                fcout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
            }
            fcout << "\r\n";

            for(const auto &out: outs) {
                for(const auto &fco: out) {
                    fcout << fco.chi2_osc << "," << fco.chi2_syst << "," << fco.dmsq << "," << fco.sinsq2tmm;
                    for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                        fcout << fco.best_fit_osc(i) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                    }
                    fcout << "\r\n";
                }
            }
        }
        std::vector<float> flattened_dchi2s;
        for(const auto& v: dchi2s) for(const auto& dchi2: v) flattened_dchi2s.push_back(dchi2);
        std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
        log<LOG_INFO>(L"%1% || 90%% Feldman-Cousins delta chi2 after throwing %2% universes is %3%") 
            % __func__ % nuniv % flattened_dchi2s[0.9*flattened_dchi2s.size()];
    }


        //***********************************************************************
        //***********************************************************************
        //******************** TEST AREA TEST AREA     **************************
        //***********************************************************************
        //***********************************************************************
    if(*protest_command){
        log<LOG_INFO>(L"%1% || PROtest. Place anything here, a playground for testing things .") % __func__;

        //***************************** END *********************************
    }

    delete metric;

    return 0;
}



//******************************** Functions to help plotting, move to a src later ************************************

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
    for(size_t i = 0; i < nerrorsample; ++i)
        specs.push_back(FillSystRandomThrow(config, prop, syst, other_index).Spec());
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

std::unique_ptr<TGraphAsymmErrors> getPostFitErrorBand(const PROconfig &config, const PROpeller &prop, PROmetric &metric, const Eigen::VectorXf &best_fit, std::vector<TH1D> &posteriors, bool scale) {
    // Fix physics parameters
    std::vector<int> fixed_pars;
    for(size_t i = 0; i < metric.GetModel().nparams; ++i) fixed_pars.push_back(i);

    Metropolis mh(simple_target{metric}, simple_proposal(metric, 0.2, fixed_pars), best_fit);

    for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
        posteriors.emplace_back("", (";"+config.m_mcgen_variation_plotname_map.at(metric.GetSysts().spline_names[i])).c_str(), 60, -3, 3);

    Eigen::VectorXf cv = FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), best_fit, true).Spec();
    Eigen::MatrixXf L = metric.GetSysts().DecomposeFractionalCovariance(config, cv);
    std::random_device rd{};
    std::mt19937 rng(rd());
    std::normal_distribution<float> nd;
    Eigen::VectorXf throws = Eigen::VectorXf::Constant(config.m_num_bins_total_collapsed, 0);

    std::vector<Eigen::VectorXf> specs;
    const auto action = [&](const Eigen::VectorXf &value) {
        int nphys = metric.GetModel().nparams;
        for(size_t i = 0; i < config.m_num_bins_total_collapsed; ++i)
            throws(i) = nd(rng);
        specs.push_back(CollapseMatrix(config, FillRecoSpectra(config, prop, metric.GetSysts(), metric.GetModel(), value, true).Spec())+L*throws);
        for(size_t i = 0; i < metric.GetSysts().GetNSplines(); ++i)
            posteriors[i].Fill(value(i+nphys));
    };
    mh.run(10'000, 50'000, action);

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

void plot_channels(const std::string &filename, const PROconfig &config, std::optional<PROspec> cv, std::optional<PROspec> best_fit, std::optional<PROdata> data, std::optional<TGraphAsymmErrors*> errband, std::optional<TGraphAsymmErrors*> posterrband, bool plot_cv_stack, TPaveText *text, bool binwidth_scale, int other_index) {
    TCanvas c;
    c.Print((filename+"[").c_str());

    std::map<std::string, std::unique_ptr<TH1D>> cvhists;
    if(cv) cvhists = getCVHists(*cv, config, binwidth_scale, other_index);

    Eigen::VectorXf bf_spec;
    if(best_fit) {
        bf_spec = other_index < 0 ? CollapseMatrix(config, best_fit->Spec()) : CollapseMatrix(config, best_fit->Spec(), other_index);
    }

    std::string ytitle = binwidth_scale ? "Events/GeV" : "Events";

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
                if(binwidth_scale) cv_hist.Scale(1, "width");
                if(cv) {
                    THStack *cvstack = NULL;
                    if(plot_cv_stack) cvstack = new THStack(std::to_string(global_channel_index).c_str(), hist_title.c_str());
                    for(size_t subchannel = 0; subchannel < config.m_num_subchannels[channel]; ++subchannel){
                        const std::string& subchannel_name  = config.m_fullnames[global_subchannel_index];
                        if(plot_cv_stack) {
                            cvstack->Add(cvhists[subchannel_name].get());
                            leg->AddEntry(cvhists[subchannel_name].get(), config.m_subchannel_plotnames[channel][subchannel].c_str() ,"f");
                        }
                        cv_hist.Add(cvhists[subchannel_name].get());
                        ++global_subchannel_index;
                    }
                    if(plot_cv_stack) {
                        cvstack->SetMaximum(1.2*cvstack->GetMaximum());
                        cvstack->Draw("hist");
                    } else {
                        cv_hist.SetMaximum(1.2*cv_hist.GetMaximum());
                        leg->AddEntry(&cv_hist, "CV");
                        cv_hist.Draw("hist");
                    }
                }

                TGraphAsymmErrors *channel_errband = NULL;
                if(errband) {
                    channel_errband = new TGraphAsymmErrors(&cv_hist);
                    int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        channel_errband->SetPointEYhigh(bin, (*errband)->GetErrorYhigh(bin+channel_start));
                        channel_errband->SetPointEYlow(bin, (*errband)->GetErrorYlow(bin+channel_start));
                    }
                    channel_errband->SetFillColor(kRed);
                    channel_errband->SetFillStyle(3345);
                    leg->AddEntry(channel_errband, "#pm 1#sigma");
                    channel_errband->Draw("2 same");
                }

                TH1D bf_hist(("bf"+std::to_string(global_channel_index)).c_str(), hist_title.c_str(), channel_nbins, edges.data());
                if(best_fit) {
                    int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        bf_hist.SetBinContent(bin+1, bf_spec(bin+channel_start));
                    }
                    bf_hist.SetLineColor(kGreen);
                    bf_hist.SetLineWidth(3);
                    leg->AddEntry(&bf_hist, "Best Fit");
                    if(cv) bf_hist.Draw("hist same");
                    else bf_hist.Draw("hist");
                }

                TGraphAsymmErrors *post_channel_errband = NULL;
                if(posterrband) {
                    post_channel_errband = new TGraphAsymmErrors(&bf_hist);
                    int channel_start = other_index < 0 ? config.GetCollapsedGlobalBinStart(global_channel_index) : config.GetCollapsedGlobalOtherBinStart(global_channel_index, other_index);
                    for(size_t bin = 0; bin < channel_nbins; ++bin) {
                        post_channel_errband->SetPointEYhigh(bin, (*posterrband)->GetErrorYhigh(bin+channel_start));
                        post_channel_errband->SetPointEYlow(bin, (*posterrband)->GetErrorYlow(bin+channel_start));
                    }
                    post_channel_errband->SetFillColor(kCyan);
                    post_channel_errband->SetFillStyle(3354);
                    leg->AddEntry(post_channel_errband, "post-fit #pm 1#sigma");
                    post_channel_errband->Draw("2 same");
                }

                TH1D data_hist;
                if(data) {
                    data_hist = data->toTH1D(config, global_channel_index, other_index);
                    data_hist.SetLineColor(kBlack);
                    data_hist.SetLineWidth(2);
                    data_hist.SetMarkerStyle(kFullCircle);
                    data_hist.SetMarkerColor(kBlack);
                    data_hist.SetMarkerSize(1);
                    leg->AddEntry(&data_hist, "Data");
                    if(cv || best_fit) data_hist.Draw("PE1 same");
                    else data_hist.Draw("E1P");
                }

                if(text) {
                    text->Draw("same");
                }
                
                leg->Draw("same");

                c.Print(filename.c_str());

                ++global_channel_index;
            }
        }
    }

    c.Print((filename+"]").c_str());
}

