#ifndef PROSYST_H_
#define PROSYST_H_

#include "PROcreate.h"

#include <vector>

namespace PROfit {

class PROsyst {
public:
    using Spline = std::vector<std::vector<std::pair<float, std::array<float, 4>>>>;

    PROsyst(const std::vector<SystStruct>& systs);

   	/* Function: Fill spline_coeffs assuming p_cv and p_multi_spec have been filled */
   	void FillSpline(const SystStruct& syst);

   	/* Function: Get weight for bin for a given shift using spline */
   	float GetSplineShift(std::string name, float shift, int bin);

  	/* Function: Get cv spectrum shifted using spline */
  	PROspec GetSplineShiftedSpectrum(const PROspec& cv, std::string name, float shift);
    PROspec GetSplineShiftedSpectrum(const PROspec& cv, std::vector<std::string> names, std::vector<float> shifts);


private:
    std::map<std::string, Spline> splines;
    std::map<std::string, Eigen::MatrixXd> covmats;
    //std::<std:string, MFA> mfa;
};

};

#endif
