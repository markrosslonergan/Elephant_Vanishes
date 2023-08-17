#ifndef PROSYST_H_
#define PROSYST_H_

//C++ include 
#include <string>
#include <vector>
#include <unordered_map>

// Our include
#include "PROcreate.h"
#include "PROlog.h"
#include "PROmfa.h"
namespace PROfit {

class PROsyst {
public:
    using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;

    	//constructor
    	PROsyst(){}
    	PROsyst(const std::vector<SystStruct>& systs);



	//----- Spline and Covariance matrix related ---
	//----- Spline and Covariance matrix related ---

	Eigen::MatrixXd SumMatrices() const;
	Eigen::MatrixXd SumMatrices(const std::vector<std::string>& sysnames) const;

    	/* Function: Given a SystStruct, generate fractinal covariance matrix, and correlation matrix, and add matrices to covmat_map and corrtmat_map
 	 * Note: this function is lazy. It wouldn't do anything if it found covariance matrix with the same name already in the map.
 	 */
    	void CreateMatrix(const SystStruct& syst);


        /* Function: given a syst struct with cv and variation spectra, build fractional covariance matrix for the systematics, as well as correlation matrix 
         * Return: {fractional covariance matrix, correlation covariance matrix}
         */
        static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GenerateCovarMatrices(const SystStruct& sys_obj);

        /* Function: given a SystStruct with cv and variation spectra, build fractional covariance matrix for the systematics, and return it
 	 * Note: it assumes the SystStruct is filled 
 	 */
	static Eigen::MatrixXd GenerateFracCovarMatrix(const SystStruct& sys_obj);

	/* Given fractional covariance matrix, calculate the correlation matrix */
	static Eigen::MatrixXd GenerateCorrMatrix(const Eigen::MatrixXd& frac_matrix);

	/* Function: check if matrix has nan, or infinite value */
	static bool isFiniteMatrix(const Eigen::MatrixXd& in_matrix);

	/* Function: if matrix has nan/inf values, change to 0. 
 	 * Note: this modifies the matrix !! 
 	 */
	static void toFiniteMatrix(Eigen::MatrixXd& in_matrix);

        /* Function: check if given matrix is positive semi-definite with tolerance. UST THIS ONE!!*/
	static bool isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXd& in_matrix, double tolerance=1.0e-16);

        /* Function: check if given matrix is positive semi-definite, no tolerance at all (besides precision error from Eigen) */
	static bool isPositiveSemiDefinite(const Eigen::MatrixXd& in_matrix);

   	   	/* function: get weight for bin for a given shift using spline */
   	float getsplineshift(std::string name, float shift, int bin);
    
   	/* Function: Fill spline_coeffs assuming p_cv and p_multi_spec have been filled */
   	void FillSpline(const SystStruct& syst);


  	/* Function: Get cv spectrum shifted using spline */
  	PROspec GetSplineShiftedSpectrum(const PROspec& cv, std::string name, float shift);
    PROspec GetSplineShiftedSpectrum(const PROspec& cv, std::vector<std::string> names, std::vector<float> shifts);


private:
    std::unordered_map<std::string, Spline> splines;
    std::unordered_map<std::string, Eigen::MatrixXd> covmat_map;
    std::unordered_map<std::string, Eigen::MatrixXd> corrmat_map;
    std::<std:string, mfa::MFA<double> *> mfas;
};

};

#endif
