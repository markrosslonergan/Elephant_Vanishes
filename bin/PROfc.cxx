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

#include "TH2D.h"
#include "TStyle.h"

using namespace PROfit;

//log_level_t GLOBAL_LEVEL = LOG_DEBUG;
log_level_t GLOBAL_LEVEL = LOG_ERROR;

struct fc_out{
    double chi2_syst, chi2_osc, dmsq, sinsq2tmm;
    Eigen::VectorXd best_fit_syst, best_fit_osc, syst_throw;
};

struct fc_args {
    size_t todo;
    std::vector<double>* dchi2s;
    std::vector<fc_out>* out;
    const PROconfig config;
    const PROpeller prop;
    PROsyst systs;
    Eigen::VectorXd cv, err;
    Eigen::MatrixXd L;
    int thread;
};

void fc_worker(fc_args args) {
    auto dchi2s = args.dchi2s;
    auto out = args.out;
    auto &config = args.config;
    auto &prop = args.prop;
    auto &systs = args.systs;
    auto &cv = args.cv;
    //auto &err = args.err;
    auto &L = args.L;
    std::random_device rd{};
    std::mt19937 rng{rd()};
    PROsc osc(prop);
    while(dchi2s->size() < args.todo) {
        std::cout << "Thread #" << args.thread << " Size " << dchi2s->size() << "\n";
        std::normal_distribution<float> d;
        std::vector<float> throws;
        Eigen::VectorXd throwC = Eigen::VectorXd::Constant(cv.size(), 0);
        for(size_t i = 0; i < systs.GetNSplines(); i++)
            throws.push_back(d(rng));
        for(size_t i = 0; i < (size_t)cv.size(); i++)
            throwC(i) = d(rng);
        //PROspec shifted = systs.GetSplineShiftedSpectrum(config, prop, throws);
        std::vector<float> phy;
        phy.push_back(std::log10(4.0f));
        phy.push_back(std::log10(0.2f));
        PROspec shifted = FillRecoSpectra(config, prop, systs, &osc, throws, phy);
        PROspec newSpec = PROspec::PoissonVariation(PROspec(shifted.Spec() + L * throwC, shifted.Error()));

        // No oscillations
        LBFGSpp::LBFGSBParam<double> param;  
        param.epsilon = 1e-6;
        param.max_iterations = 100;
        param.max_linesearch = 50;
        param.delta = 1e-6;

        size_t nparams = systs.GetNSplines();
        Eigen::VectorXd lb = Eigen::VectorXd::Constant(nparams, -3.0);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(nparams, 3.0);
        PROfitter fitter(ub, lb, param);

        PROchi chi("3plus1",&config,&prop,&systs,&osc, newSpec, nparams, systs.GetNSplines(), PROchi::BinnedChi2);
        double chi2_syst = fitter.Fit(chi);

        // With oscillations
        LBFGSpp::LBFGSBParam<double> param_osc;  
        param_osc.epsilon = 1e-6;
        param_osc.max_iterations = 100;
        param_osc.max_linesearch = 250;
        param_osc.delta = 1e-6;

        nparams = 2 + systs.GetNSplines();
        Eigen::VectorXd lb_osc = Eigen::VectorXd::Constant(nparams, -3.0);
        lb_osc(0) = -2; lb_osc(1) = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd ub_osc = Eigen::VectorXd::Constant(nparams, 3.0);
        ub_osc(0) = 2; ub_osc(1) = 0;
        PROfitter fitter_osc(ub_osc, lb_osc, param);

        PROchi chi_osc("3plus1",&config,&prop,&systs,&osc, newSpec, nparams, systs.GetNSplines(), PROchi::BinnedChi2);
        double chi2_osc = fitter_osc.Fit(chi_osc); 

        Eigen::VectorXd t = Eigen::VectorXd::Constant(throws.size(), 0);
        for(size_t i = 0; i < throws.size(); i++) t(i) = throws[i];

        out->push_back({
                chi2_syst, chi2_osc, 
                std::pow(10, fitter_osc.best_fit(0)), std::pow(10, fitter_osc.best_fit(1)), 
                fitter.best_fit, fitter_osc.best_fit.segment(2, nparams-2), t
        });

        dchi2s->push_back(std::abs(chi2_syst - chi2_osc ));
    }
}

int main(int argc, char* argv[])
{
    gStyle->SetOptStat(0);
    CLI::App app{"Test for PROfit"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    int maxevents = 100;
    size_t nfit = 1, nthread = 1;

    //doubles
    app.add_option("-x,--xml", xmlname, "Input PROfit XML config.");
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");
    app.add_option("-n,--nfit",nfit, "Number of fits.");
    app.add_option("-t,--nthread",nthread, "Number of threads.");

    CLI11_PARSE(app, argc, argv);

    if(nthread > nfit) nthread = nfit;

    //Initilize configuration from the XML;
    PROconfig myConf(xmlname);

    //Inititilize PROpeller to keep MC
    PROpeller myprop;
    
    //Initilize objects for systematics storage
    std::vector<SystStruct> systsstructs;

    //Process the CAF files to grab and fill all SystStructs and PROpeller
    PROcess_CAFAna(myConf, systsstructs, myprop);

    //Build a PROsyst to sort and analyze all systematics
    PROsyst systs(systsstructs);

    //Define the model (currently 3+1 SBL)
    PROsc osc(myprop);

    std::random_device rd{};
    std::mt19937 gen{rd()};

    Eigen::VectorXd data = systsstructs.back().CV().Spec();
    Eigen::VectorXd err = systsstructs.back().CV().Error();
    Eigen::MatrixXd diag = data.array().matrix().asDiagonal();
    Eigen::MatrixXd full_cov = diag * systs.fractional_covariance * diag;
    //Eigen::MatrixXd stat_cov = CollapseMatrix(myConf, diag);
    Eigen::LLT<Eigen::MatrixXd> llt(CollapseMatrix(myConf, full_cov));//+ stat_cov);
    
    std::vector<std::vector<double>> dchi2s;
    dchi2s.reserve(nthread);
    std::vector<std::vector<fc_out>> outs;
    outs.reserve(nthread);
    std::vector<std::thread> threads;
    for(size_t i = 0; i < nthread; i++) {
        dchi2s.emplace_back();
        outs.emplace_back();
        // TODO: nfit/nthread is not correct if nthread does not evenly divide nfit
        fc_args args{nfit/nthread, &dchi2s.back(), &outs.back(), myConf, myprop, systs, data, err, llt.matrixL(), (int)i};
        threads.emplace_back(fc_worker, args);
    }
    for(auto&& t: threads) {
        t.join();
    }

    TH1D hdchi2s("hdchi2s", ";#Delta#chi^{2}", 50, 0, 50);
    std::vector<double> dmsqedges;
    for(size_t i = 0; i < 41; ++i)
        dmsqedges.push_back(std::pow(10.0, -2.0 + i * 4.0 / 40));
    TH1D hdmsqs("hdmsqs", ";#Deltam^{2}", 40, dmsqedges.data());
    TH1D hss2t("hss2t", ";sin^{2}2#theta_{#mu#mu}", 50, 0, 1);
    TH2D hdmsqvss2t("hdmsqvss2t", ";sin^{2}2#theta_{#mu#mu};#Deltam^{2}", 50, 0, 1, 40, dmsqedges.data());

    std::vector<double> flattened_dchi2s;
    for(const auto& v: dchi2s) {
        for(const auto& dchi2: v) {
            flattened_dchi2s.push_back(dchi2);
            hdchi2s.Fill(dchi2);
        }
    }
    for(const auto& out: outs) {
      for(const auto& fco: out) {
        hdmsqs.Fill(fco.dmsq);
        hss2t.Fill(fco.sinsq2tmm);
        hdmsqvss2t.Fill(fco.sinsq2tmm, fco.dmsq);
      }
    }

    TCanvas c;
    c.SetLogy();
    hdchi2s.Draw("hist");
    c.Print("dchi2s.pdf");
    hss2t.Draw("hist");
    c.Print("sinsq2tmm.pdf");
    c.SetLogz();
    hdmsqvss2t.Draw("colz");
    c.Print("dmsq_v_sinsq2tmm.pdf");
    c.SetLogx();
    hdmsqs.Draw("hist");
    c.Print("dmsqs.pdf");

    std::sort(flattened_dchi2s.begin(), flattened_dchi2s.end());

    std::cout << "90% delta chi2: " << flattened_dchi2s[0.9*flattened_dchi2s.size()] << std::endl;

    return 0;
}

