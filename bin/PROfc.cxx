#include "PROconfig.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROcess.h"
#include "PROfitter.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Eigen>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>

#include "TTree.h"

using namespace PROfit;

log_level_t GLOBAL_LEVEL = LOG_ERROR;

struct fc_out{
    double chi2_syst, chi2_osc, dmsq, sinsq2tmm;
    Eigen::VectorXd best_fit_syst, best_fit_osc, syst_throw;
};

struct fc_args {
    const size_t todo;
    std::vector<double>* dchi2s;
    std::vector<fc_out>* out;
    const PROconfig config;
    const PROpeller prop;
    const PROsyst systs;
    const std::vector<float> phy_params;
    const Eigen::MatrixXd L;
    const int thread;
    const bool binned;
};

void fc_worker(fc_args args) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    PROsc osc(args.prop);
    PROchi::EvalStrategy strat = args.binned ? PROchi::BinnedChi2 : PROchi::EventByEvent;
    for(size_t u = 0; u < args.todo; ++u) {
        log<LOG_INFO>(L"%1% | Thread #%2% Throw #%3%") % __func__ % args.thread % u;
        std::normal_distribution<float> d;
        std::vector<float> throws;
        Eigen::VectorXd throwC = Eigen::VectorXd::Constant(args.config.m_num_bins_total, 0);
        for(size_t i = 0; i < args.systs.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(size_t i = 0; i < args.config.m_num_bins_total; i++)
            throwC(i) = d(rng);
        PROspec shifted = FillRecoSpectra(args.config, args.prop, args.systs, &osc, throws, args.phy_params, strat);
        std::cout << "Num total bins " << args.config.m_num_bins_total << " Num spectrum bins " << shifted.Spec().size() << std::endl;
        PROspec newSpec = PROspec::PoissonVariation(PROspec(shifted.Spec() + args.L * throwC, shifted.Error()));

        // No oscillations
        LBFGSpp::LBFGSBParam<double> param;  
        param.epsilon = 1e-6;
        param.max_iterations = 100;
        param.max_linesearch = 50;
        param.delta = 1e-6;

        size_t nparams = args.systs.GetNSplines();
        Eigen::VectorXd lb = Eigen::VectorXd::Constant(nparams, -3.0);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(nparams, 3.0);
        PROfitter fitter(ub, lb, param);

        PROchi chi("3plus1",&args.config,&args.prop,&args.systs,&osc, newSpec, nparams, args.systs.GetNSplines(), strat);
        double chi2_syst = fitter.Fit(chi);

        // With oscillations
        LBFGSpp::LBFGSBParam<double> param_osc;  
        param_osc.epsilon = 1e-6;
        param_osc.max_iterations = 100;
        param_osc.max_linesearch = 250;
        param_osc.delta = 1e-6;

        nparams = 2 + args.systs.GetNSplines();
        Eigen::VectorXd lb_osc = Eigen::VectorXd::Constant(nparams, -3.0);
        lb_osc(0) = -2; lb_osc(1) = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd ub_osc = Eigen::VectorXd::Constant(nparams, 3.0);
        ub_osc(0) = 2; ub_osc(1) = 0;
        PROfitter fitter_osc(ub_osc, lb_osc, param);

        PROchi chi_osc("3plus1",&args.config,&args.prop,&args.systs,&osc, newSpec, nparams, args.systs.GetNSplines(), strat);
        double chi2_osc = fitter_osc.Fit(chi_osc); 

        Eigen::VectorXd t = Eigen::VectorXd::Constant(throws.size(), 0);
        for(size_t i = 0; i < throws.size(); i++) t(i) = throws[i];

        args.out->push_back({
                chi2_syst, chi2_osc, 
                std::pow(10, fitter_osc.best_fit(0)), std::pow(10, fitter_osc.best_fit(1)), 
                fitter.best_fit, fitter_osc.best_fit.segment(2, nparams-2), t
        });

        args.dchi2s->push_back(std::abs(chi2_syst - chi2_osc ));
    }
}

int main(int argc, char* argv[])
{
    CLI::App app{"Test for PROfit"}; 

    std::string xmlname, filename;
    int maxevents;
    size_t  nthread, nfit;
    std::array<float, 2> injected_pt{0, 0};
    std::vector<std::string> syst_list, systs_excluded;
    bool eventbyevent=false;

    app.add_option("-x, --xml",       xmlname,        "Input PROfit XML config.")->required();
    app.add_option("-n, --nfit",      nfit,           "Number of fits.")->required();
    app.add_option("-m, --max",       maxevents,      "Max number of events to run over.")->default_val(50000);
    app.add_option("-v, --verbosity", GLOBAL_LEVEL,   "Verbosity Level [1-4].")->default_val(LOG_ERROR);
    app.add_option("-t, --nthread",   nthread,        "Number of fits.")->default_val(1);
    app.add_option("-o, --outfile",   filename,       "If you want chisq to be dumped to text file, provide name")->default_val("");
    app.add_option("--inject",        injected_pt,    "Physics parameters to inject as true signal.")->default_str("0 0");
    app.add_option("--syst-list",     syst_list,      "Override list of systematics to use (note: all systs must be in the xml).");
    app.add_option("--exclude-systs", systs_excluded, "List of systematics to exclude.")->excludes("--syst-list"); 

    app.add_flag("--event-by-event",  eventbyevent,   "Do you want to weight event-by-event?");

    CLI11_PARSE(app, argc, argv);

    bool savetoroot = filename.size() > 5 && filename.substr(filename.size() - 5) == ".root";

    if(nthread > nfit) nthread = nfit;

    //Initilize configuration from the XML;
    PROconfig myConf(xmlname);

    PROpeller myprop;
    std::vector<SystStruct> systsstructs;
    PROcess_CAFAna(myConf, systsstructs, myprop);

    PROsyst systs(systsstructs);
    PROsc osc(myprop);

    std::vector<float> pparams = {std::log10(injected_pt[0]), std::log10(injected_pt[1])};
    log<LOG_INFO>(L"%1% | Injected point: sinsq2t = %2% and dmsq = %3%\n") % __func__ % injected_pt[0] % injected_pt[1];

    if(syst_list.size()) {
      systs = systs.subset(syst_list);
    } else if(systs_excluded.size()) {
      systs = systs.excluding(systs_excluded);
    }

    Eigen::MatrixXd diag = FillCVSpectrum(myConf, myprop, !eventbyevent).Spec().array().matrix().asDiagonal();
    Eigen::MatrixXd full_cov = diag * systs.fractional_covariance * diag;
    Eigen::LLT<Eigen::MatrixXd> llt(full_cov);
    
    std::vector<std::vector<double>> dchi2s;
    dchi2s.reserve(nthread);
    std::vector<std::vector<fc_out>> outs;
    outs.reserve(nthread);
    std::vector<std::thread> threads;
    size_t todo = nfit/nthread;
    size_t addone = nthread - nfit%nthread;
    for(size_t i = 0; i < nthread; i++) {
        dchi2s.emplace_back();
        outs.emplace_back();
        fc_args args{todo + (i >= addone), &dchi2s.back(), &outs.back(), myConf, myprop, systs, pparams, llt.matrixL(), (int)i, !eventbyevent};
        threads.emplace_back(fc_worker, args);
    }
    for(auto&& t: threads) {
        t.join();
    }

    if(savetoroot) {
        TFile fout(filename.c_str(), "RECREATE");
        fout.cd();
        double chi2_osc, chi2_syst, best_dmsq, best_sinsq2t;
        std::map<std::string, double> best_systs_osc, best_systs, syst_throw;
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
    } else if(filename != "") {
        ofstream fout(filename);
        fout << "chi2_osc,chi2_syst,best_dmsq,best_sinsq2t";
        for(const std::string &name: systs.spline_names) {
            fout << ",best_" << name << "_osc,best_" << name << "," << name << "_throw";
        }
        fout << "\r\n";

        for(const auto &out: outs) {
            for(const auto &fco: out) {
                fout << fco.chi2_osc << "," << fco.chi2_syst << "," << fco.dmsq << "," << fco.sinsq2tmm;
                for(size_t i = 0; i < systs.GetNSplines(); ++i) {
                    fout << fco.best_fit_osc(i) << "," << fco.best_fit_syst(i) << "," << fco.syst_throw(i);
                }
                fout << "\r\n";
            }
        }
    }

    std::vector<double> flattened_dchi2s;
    for(const auto& v: dchi2s) for(const auto& dchi2: v) flattened_dchi2s.push_back(dchi2);
    std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());
    std::cout << "90% delta chi2: " << flattened_dchi2s[0.9*flattened_dchi2s.size()] << std::endl;

    return 0;
}

