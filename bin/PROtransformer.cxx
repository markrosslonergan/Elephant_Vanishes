#include "PROconfig.h"
#include "PROspec.h"
#include "PROcreate.h"
#include "PROtocall.h"
#include "sbnanaobj/StandardRecord/SRGlobal.h"
#include "sbnanaobj/StandardRecord/SRWeightPSet.h"
#include <bits/stdc++.h>
#include <chrono>

#include "CLI11.h"
#include "LBFGSB.h"

#include "TFile.h"
#include "TTree.h"

#include <memory>
#include <string>
#include <cstdlib>
#include <highfive/H5File.hpp>
#include <ctime>

using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;

int main(int argc, char* argv[]){
 
  CLI::App app{"Test for PROfit"};
  // Define options
  std::string infilename = "NULL.flat.caf.root";
  std::string outfilename = "null.h5";
  std::string xmlname = "xml.xml";
  app.add_option("-x,--xml", xmlname, "Input PROfit XML config.");
  app.add_option("-o,--outfile", outfilename, "H5 file to output to.");
  app.add_option("-i, --infile", infilename, "root file input");
  app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");
 
  CLI11_PARSE(app, argc, argv);

  auto start = chrono::high_resolution_clock::now();
  auto flat_caf = std::make_unique<TFile>(infilename.c_str(),"read");
  auto end  = chrono::high_resolution_clock::now();
  double time_taken =
      chrono::duration_cast<chrono::nanoseconds>(end - start).count();
 
  time_taken *= 1e-9;

  std::cout<<"It took this many seconds to open caf: "<<time_taken<<std::endl;
  TTree *mytree = (TTree*)flat_caf->Get("nutree");
  
  double recoE = 0;
  double trueE =0;
  double baseline=0;
  
  mytree->SetBranchAddress("trueE", &trueE);
  mytree->SetBranchAddress("recoE", &recoE);
  mytree->SetBranchAddress("baseline", &baseline);

  std::cout << "set branch addresses" << std::endl;
  //std::vector<float> v_trueE;
  std::vector<float> v_recoE;
  //std::vector<float> v_baseline;
  std::vector<float> L_over_E;
  start = chrono::high_resolution_clock::now();

  for(int i = 0 ; i != mytree->GetEntries(); ++i){
      mytree->GetEntry(i);
      //v_trueE.push_back(trueE);
      v_recoE.push_back(recoE);
      //v_baseline.push_back(baseline);
      L_over_E.push_back(baseline/recoE);
  }

  end  = chrono::high_resolution_clock::now();
  time_taken =
      chrono::duration_cast<chrono::nanoseconds>(end - start).count();
 
  time_taken *= 1e-9;
  std::cout<<"It took this many seconds to access caf: "<<time_taken<<std::endl;
  HighFive::File myfile(outfilename, HighFive::File::Truncate);
  //myfile.createDataSet("nugroup/trueE", trueE);
  myfile.createDataSet("nugroup/recoE", v_recoE);
  //myfile.createDataSet("nugroup/baseline", baseline);
  myfile.createDataSet("nugroup/LE", L_over_E);
  //auto dset1 = myfile.getDataSet("/nugroup/trueE/")
  //auto trueE = dset1.read<std::vector<float>>
  //}
  std::cout<<"starting the config piece"<<std::endl;
  PROconfig myConf(xmlname); 
  std::vector<long int> digitized_reco;
  for(int i = 0 ; i <= v_recoE.size(); ++i){
      long int this_bin = FindGlobalBin(myConf, v_recoE[i], "nu_SBND_numu_cc");
      digitized_reco.push_back(this_bin);
      }
 
  start = chrono::high_resolution_clock::now();
  int nbins   = myConf.m_num_bins_total;
  PROspec   mySpec(nbins);
  std::vector<float> powfvec;
  for(int i = 0 ; i <= v_recoE.size(); ++i){
      mySpec.Fill(digitized_reco[i], 1.0);
      float x = sinf(L_over_E[i]);
      powfvec.push_back(x*x);
      } 
  end  = chrono::high_resolution_clock::now();
  time_taken =
      chrono::duration_cast<chrono::nanoseconds>(end - start).count();
  time_taken *= 1e-9;
  std::cout<<"It took this many seconds to fill spec: "<<time_taken<<std::endl;
 
  return 0;
    
}
