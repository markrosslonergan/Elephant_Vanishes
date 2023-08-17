#include "PROconfig.h"
#include "PROspec.h"
#include "PROcreate.h"
#include "PROsyst.h"
#include "sbnanaobj/StandardRecord/SRGlobal.h"
#include "sbnanaobj/StandardRecord/SRWeightPSet.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include "TFile.h"

#include <memory>
#include <string>

using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;

int main(int argc, char* argv[])
{

    CLI::App app{"Test for PROfit"}; 

    // Define options
    std::string xmlname = "NULL.INPUT.xml"; 

    app.add_option("-x,--xml", xmlname, "XML file to read configuration from.");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");

    CLI11_PARSE(app, argc, argv);

    std::cout << "read from xml: " << xmlname << std::endl;
    PROconfig config(xmlname);
    PROspec cv_spec = CreatePROspecCV(config);

    //fill in sysetamtic variations 
    std::vector<SystStruct> syst_vector;
    PROcess_SBNfit(config, syst_vector);
    std::cout << syst_vector.size() << std::endl;

    //generate covariance matrix 
    Eigen::MatrixXd fractional_matrix = PROsyst::GenerateFracCovarMatrix(syst_vector[0]);
    std::cout << " Formed fractional covariance matrix by PROfit: " << std::endl; 
    std::cout << fractional_matrix << std::endl;

    //check is matrix is psd
    bool res1 =  PROsyst::isPositiveSemiDefinite(fractional_matrix), res2 = PROsyst::isPositiveSemiDefinite_WithTolerance(fractional_matrix);
    std::cout << "Matrix is positive semidefinite? " << res1 << " " << res2 << std::endl;


    Eigen::MatrixXd matrix1(3,3), matrix2(2,2), matrix3(2,2);
    matrix1 << 3,2,1,2,3,1,1,2,3;
    matrix2 << 1, 2, 2, 3;
    matrix3 << 1, 2, 2, 4;
    res1 =  PROsyst::isPositiveSemiDefinite(matrix1), res2 = PROsyst::isPositiveSemiDefinite_WithTolerance(matrix1);
    std::cout << "Matrix1 is positive semidefinite? " << res1 << " " << res2 <<  std::endl;
    std::cout << matrix1 << std::endl;
    res1 =  PROsyst::isPositiveSemiDefinite(matrix2), res2 = PROsyst::isPositiveSemiDefinite_WithTolerance(matrix2);
    std::cout << "Matrix2 is positive semidefinite? " << res1 << " " << res2 <<  std::endl;
    std::cout << matrix2 << std::endl;
    res1 =  PROsyst::isPositiveSemiDefinite(matrix3), res2 = PROsyst::isPositiveSemiDefinite_WithTolerance(matrix3);
    std::cout << "Matrix3 is positive semidefinite? " << res1 << " " << res2 <<  std::endl;
    std::cout << matrix3 << std::endl;
   
     return 0;
}
