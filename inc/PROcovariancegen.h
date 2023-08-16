#ifndef PROCOVARIANCEGEN_H_
#define PROCOVARIANCEGEN_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>


// ROOT
#include "TH1D.h"

// EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>

// PROfit
#include "PROconfig.h"
#include "PROspec.h"
#include "PROcreate.h"

//Delete when ready
#include "TMatrixDEigen.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixD.h"
#include "TMatrixT.h"
#include "TDirectory.h"
#include "TVectorD.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TNtuple.h"
#include "TLine.h"

#include "TText.h"
#include "TROOT.h"
#include "TRint.h"
#include "TGraph.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "THnSparse.h"
#include "TStopwatch.h"
#include "TTreeFormula.h"
#include "TCanvas.h"

#include <sys/stat.h> 

//#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
    typedef float eweight_type;
#else
    typedef double eweight_type;
#endif


namespace PROfit{

    /* Function: given a syst struct with cv and variation spectra, build fractional covariance matrix for the systematics, and return it. 
     */
    Eigen::MatrixXd GenerateCovarMatrix(const SystStruct& sys_obj);

int generateFracCovarianceFromXML(const PROconfig &inconfig, Eigen::MatrixXd &out_frac_covar);


};
#endif
