#include "PROconfig.h"
#include "PROlog.h"
#include "PROspec.h"
#include "PROsyst.h"
#include "PROtocall.h"
#include "PROcreate.h"
#include "PROpeller.h"
#include "PROchi.h"
#include "PROcess.h"
#include "PROsurf.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>

#include "TH2D.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include "TRatioPlot.h"
#include "TGraphAsymmErrors.h"

using namespace PROfit;

//Run a single fit at a given point in parameter space (nominally no osc) with mock data
//log_level_t GLOBAL_LEVEL = LOG_DEBUG;
log_level_t GLOBAL_LEVEL = LOG_ERROR;

int main(int argc, char* argv[])
{
  gStyle->SetOptStat(0);
  CLI::App app{"Test for PROfit"}; 
  
  // Define options
  std::string xmlname = "NULL.xml"; 
  int maxevents = 50000;
  size_t  nthread = 1;
  //Define a filename to save chisq values in
  std::string filename;
  bool binned=false;
  std::vector<int> grid_size;
  std::vector<std::string> mockparams;
  std::vector<float> mockshifts;
  std::vector<float> physics_params;
  
  app.add_option("-x, --xml",       xmlname, "Input PROfit XML config.");
  app.add_option("-m, --max",       maxevents, "Max number of events to run over.");
  app.add_option("-v, --verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");
  app.add_option("-t, --nthread",   nthread, "Number of fits.");
  app.add_option("-o, --outfile",   filename, "If you want chisq to be dumped to text file, provide name");
  app.add_option("-s, --mocksys",   mockparams, "Vector of systematics parameter names to vary for mock data")->expected(-1);
  app.add_option("-u, --mockvals",  mockshifts, "Vector of size of shifts. Default +1")->expected(-1);
  app.add_option("-p, --pparams",   physics_params, "deltam^2, sin^22thetamumu, default no osc")->expected(-1);

  app.add_flag(  "-b, --binned",    binned, "Do you want to weight event-by-event?");

  CLI11_PARSE(app, argc, argv);


  //Initilize configuration from the XML;
  PROconfig config(xmlname);

  //Inititilize PROpeller to keep MC
  PROpeller prop;

  //Initilize objects for systematics storage
  std::vector<SystStruct> systsstructs;

  //Process the CAF files to grab and fill all SystStructs and PROpeller
  PROcess_CAFAna(config, systsstructs, prop);

  //Build a PROsyst to sort and analyze all systematics
  PROsyst systs(systsstructs);

  for (size_t i = 0; i < mockparams.size(); ++i) {
    log<LOG_INFO>(L"%1% || Mock data parameters %2%") % __func__  % mockparams[i].c_str();
  }
  PROspec data, cv;
  cv = systsstructs.back().CV();
  if (mockparams.empty()) {
    log<LOG_INFO>(L"%1% || Will use CV MC as data for this study") % __func__  ;
    data = systsstructs.back().CV();
  }
  else{
    if (mockshifts.size() != mockparams.size()) {
      log<LOG_INFO>(L"%1% || No shifts provided, setting all parameter shifts to 1 sigma") % __func__  ;
      for (size_t i = 0; i < mockparams.size(); ++i) {
	mockshifts.push_back(1.0);
      }
    }
    data = systs.GetSplineShiftedSpectrum(config,prop,mockparams,mockshifts);
  }

  //cv.plotSpectrum(config, filename+"_spec_cv");
  //cv.Print();
  //data.plotSpectrum(config, filename+"_spec_mockdata");
  //data.Print();

  TH1D hcv = cv.toTH1D(config,0);
  TH1D hmock = data.toTH1D(config,0);
  hcv.Scale(1, "width");
  hmock.Scale(1, "width");
  hcv.GetYaxis()->SetTitle("Events/GeV");
  hmock.GetYaxis()->SetTitle("Events/GeV");
  hcv.SetTitle("");
  hmock.SetTitle("");

  TCanvas *c = new TCanvas((filename+"_spec_cv").c_str(), (filename+"_spec_cv").c_str(), 800, 800);
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
  for (const auto& m : mockparams) {
    char ns[6];
    snprintf(ns, sizeof(ns),"%.2f", mockshifts[i]);
    leg->AddEntry(null, (m+": "+ns+ " sigma").c_str(),"");
    i++;
  }
  leg->Draw();
  c->SaveAs((filename+"_spec.pdf").c_str());

  //I think setting these here is not doing anything - want to be able to define them...
  if (physics_params.empty()) {
    physics_params.push_back(0.0);
    physics_params.push_back(0.0);
  }

  //Define the model (currently 3+1 SBL)
  PROsc osc(prop);

  PROfile(config,prop,systs,osc,data,filename+"_PROfile");

  return 0;
}
