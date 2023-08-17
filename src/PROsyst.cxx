#include "PROsyst.h"
#include "PROcreate.h"

namespace PROfit {

   PROsyst::PROsyst(const std::vector<SystStruct>& systs) {
        for(const auto& syst: systs) {
            if(syst.mode == "multisigma") {
                FillSpline(syst);
            } else if(syst.mode == "multisim") {
		this->CreateMatrix(syst);
		
            }
        }
    }

Eigen::MatrixXd PROsyst::SumMatrices() const{

    Eigen::MatrixXd sum_matrix;
    if(covmat_map.size()){
	int nbins = (covmat_map.begin())->second.rows();
	sum_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
	for(auto& p : covmat_map){
	    sum_matrix += p.second;
	}
    }else{
	log<LOG_ERROR>(L"%1% || There is no covariance available!") % __func__;
	log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;
    }
    return sum_matrix;
}

Eigen::MatrixXd PROsyst::SumMatrices(const std::vector<std::string>& sysnames) const{

   Eigen::MatrixXd sum_matrix;
   if(covmat_map.size()){
	int nbins = (covmat_map.begin())->second.rows();
	sum_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
   }
   else{
        log<LOG_ERROR>(L"%1% || There is no covariance available!") % __func__;
        log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;
        return sum_matrix;
   }


   for(auto& sys : sysnames){
       if(covmat_map.find(sys) == covmat_map.end()){
	    log<LOG_INFO>(L"%1% || No matrix in the map matches with name %2%, Skip") % __func__ % sys.c_str();
	}else{
	    sum_matrix += covmat_map.at(sys);
	}
   }

   return sum_matrix;
}

void PROsyst::CreateMatrix(const SystStruct& syst){

    std::string sysname = syst.GetSysName();

    //generate matrix only if it's not already in the map 
    if(covmat_map.find(sysname) == covmat_map.end()){
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> matrices = PROsyst::GenerateCovarMatrices(syst);
	covmat_map[sysname] = matrices.first;
  	corrmat_map[sysname] = matrices.second;

    }

    return;
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd>  PROsyst::GenerateCovarMatrices(const SystStruct& sys_obj){
    //get fractioal covar
    Eigen::MatrixXd frac_covar_matrix = PROsyst::GenerateFracCovarMatrix(sys_obj);

    //get fractional covariance matrix
    Eigen::MatrixXd corr_covar_matrix = PROsyst::GenerateCorrMatrix(frac_covar_matrix);

    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>({frac_covar_matrix, corr_covar_matrix});
}

Eigen::MatrixXd PROsyst::GenerateFracCovarMatrix(const SystStruct& sys_obj){
    int n_universe = sys_obj.GetNUniverse(); 
    std::string sys_name = sys_obj.GetSysName();
    
    const PROspec& cv_spec = sys_obj.CV();
    int nbins = cv_spec.GetNbins();
    log<LOG_INFO>(L"%1% || Generating covariance matrix.. size: %2% x %3%") % __func__ % nbins % nbins;

    //build full covariance matrix 
    Eigen::MatrixXd full_covar_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
    for(int i = 0; i != n_universe; ++i){
	PROspec spec_diff  = cv_spec - sys_obj.Variation(i);
	full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<double>(n_universe);
    }
 
    //build fractional covariance matrix 
    //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
    Eigen::MatrixXd cv_spec_matrix =  Eigen::MatrixXd::Identity(nbins, nbins);
    for(int i =0; i != nbins; ++i)
	cv_spec_matrix(i, i) = 1.0/cv_spec.GetBinContent(i);

    //second, get fractioal covar
    Eigen::MatrixXd frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;


    //check if it's good
    if(!PROsyst::isPositiveSemiDefinite_WithTolerance(frac_covar_matrix)){
         log<LOG_ERROR>(L"%1% || Fractional Covariance Matrix is not positive semi-definite!") % __func__;
         log<LOG_ERROR>(L"Terminating.");
         exit(EXIT_FAILURE);
    }
  
    //zero out nans 
    PROsyst::toFiniteMatrix(frac_covar_matrix);

    return frac_covar_matrix;
}

Eigen::MatrixXd PROsyst::GenerateCorrMatrix(const Eigen::MatrixXd& frac_matrix){
    int nbins = frac_matrix.rows();
    Eigen::MatrixXd corr_covar_matrix = frac_matrix;

    Eigen::MatrixXd error_reciprocal_matrix = Eigen::MatrixXd::Identity(nbins, nbins);
    for(int i = 0; i != nbins; ++i){
	if(frac_matrix(i,i) != 0)
	    error_reciprocal_matrix(i,i) = 1/sqrt(frac_matrix(i,i));
    }

    corr_covar_matrix = error_reciprocal_matrix * frac_matrix * error_reciprocal_matrix;
    //zero out nans 
    PROsyst::toFiniteMatrix(corr_covar_matrix);
    return corr_covar_matrix;
}


void PROsyst::toFiniteMatrix(Eigen::MatrixXd& in_matrix){
    if(!PROsyst::isFiniteMatrix(in_matrix)){
	log<LOG_DEBUG>(L"%1% || Changing Nan/inf values to 0.0");
	in_matrix = in_matrix.unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
    }
    return;
}

bool PROsyst::isFiniteMatrix(const Eigen::MatrixXd& in_matrix){

    //check for nan and infinite
    if(!in_matrix.allFinite()){
	log<LOG_ERROR>(L"%1% || Matrix has Nan or non-finite values.") % __func__ ;
	return false;
    }
    return true;
}

bool PROsyst::isPositiveSemiDefinite(const Eigen::MatrixXd& in_matrix){

    //first, check if it's symmetric 
    if(!in_matrix.isApprox(in_matrix.transpose(), Eigen::NumTraits<double>::dummy_precision())){
	log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % Eigen::NumTraits<double>::dummy_precision();
	return false;
    }

    //second, check if it's positive semi-definite;
    Eigen::LDLT<Eigen::MatrixXd> llt(in_matrix);
    if((llt.info() == Eigen::NumericalIssue ) || (!llt.isPositive()) )
	return false;

    return true;

}

bool PROsyst::isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXd& in_matrix, double tolerance ){

    //first, check if it's symmetric 
    if(!in_matrix.isApprox(in_matrix.transpose(), Eigen::NumTraits<double>::dummy_precision())){
	log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % Eigen::NumTraits<double>::dummy_precision();
	return false;
    }
   

    //second, check if it's positive semi-definite;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(in_matrix);
    if(eigensolver.info() != Eigen::Success){
	log<LOG_ERROR>(L"%1% || Failing to get eigenvalues..") % __func__ ;
	return false;
    }
   
    Eigen::VectorXd eigenvals = eigensolver.eigenvalues();
    for(auto val : eigenvals ){
	if(val < 0 && fabs(val) > tolerance){
	   log<LOG_ERROR>(L"%1% || Matrix is not PSD. Found negative eigenvalues beyond tolerance (%2%): %3%") % __func__ % tolerance % val;
	   return false;
	}
    }
    return true;

}

    void PROsyst::FillSpline(const SystStruct& syst) {
        std::vector<PROspec> ratios;
        ratios.reserve(syst.p_multi_spec.size());
        for(size_t i = 0; i < syst.p_multi_spec.size(); ++i) {
            ratios.push_back(*syst.p_multi_spec[i] / *syst.p_cv);
            if(syst.knobval[i] == -1) ratios.push_back(*syst.p_cv / *syst.p_cv);
        }
        Spline spline_coeffs;
        spline_coeffs.reserve(syst.p_cv->GetNbins());
        for(long i = 0; i < syst.p_cv->GetNbins(); ++i) {
            std::vector<std::pair<float, std::array<float, 4>>> spline;
            spline.reserve(syst.knobval.size());

            // This comment is copy-pasted from CAFAna:
            // This is cubic interpolation. For each adjacent set of four points we
            // determine coefficients for a cubic which will be the curve between the
            // center two. We constrain the function to match the two center points
            // and to have the right mean gradient at them. This causes this patch to
            // match smoothly with the next one along. The resulting function is
            // continuous and first and second differentiable. At the ends of the
            // range we fit a quadratic instead with only one constraint on the
            // slope. The coordinate conventions are that point y1 sits at x=0 and y2
            // at x=1. The matrices are simply the inverses of writing out the
            // constraints expressed above.

            const float y1 = ratios[0].GetBinContent(i);
            const float y2 = ratios[1].GetBinContent(i);
            const float y3 = ratios[2].GetBinContent(i);
            const Eigen::Vector3f v{y1, y2, (y3-y1)/2};
            const Eigen::Matrix3f m{{ 1, -1,  1},
                {-2,  2, -1},
                { 1,  0,  0}};
            const Eigen::Vector3f res = m * v;
            spline.push_back({syst.knobval[0], {res(2), res(1), res(0), 0}});

            for(unsigned int shiftIdx = 1; shiftIdx < ratios.size()-2; ++shiftIdx){
                const float y0 = ratios[shiftIdx-1].GetBinContent(i);
                const float y1 = ratios[shiftIdx  ].GetBinContent(i);
                const float y2 = ratios[shiftIdx+1].GetBinContent(i);
                const float y3 = ratios[shiftIdx+2].GetBinContent(i);
                const Eigen::Vector4f v{y1, y2, (y2-y0)/2, (y3-y1)/2};
                const Eigen::Matrix4f m{{ 2, -2,  1,  1},
                    {-3,  3, -2, -1},
                    { 0,  0,  1,  0},
                    { 1,  0,  0,  0}};
                const Eigen::Vector4f res = m * v;
                float knobval = syst.knobval[shiftIdx] <  0 ? syst.knobval[shiftIdx] :
                    syst.knobval[shiftIdx] == 1 ? 0 :
                    syst.knobval[shiftIdx - 1];
                spline.push_back({knobval, {res(3), res(2), res(1), res(0)}});
            }

            const float y4 = ratios[ratios.size() - 3].GetBinContent(i);
            const float y5 = ratios[ratios.size() - 2].GetBinContent(i);
            const float y6 = ratios[ratios.size() - 1].GetBinContent(i);
            const Eigen::Vector3f vp{y5, y6, (y6-y4)/2};
            const Eigen::Matrix3f mp{{-1,  1, -1},
                { 0,  0,  1},
                { 1,  0,  0}};
            const Eigen::Vector3f resp = mp * vp;
            spline.push_back({syst.knobval[syst.knobval.size() - 1], {resp(2), resp(1), resp(0), 0}});

            spline_coeffs.push_back(spline);
        }
        splines[syst.systname] = spline_coeffs;
    }

    float PROsyst::GetSplineShift(std::string name, float shift , int bin) {
        if(bin < 0 || bin >= splines[name].size()) return -1;
        const float lowest_knobval = splines[name][0][0].first;
        int shiftBin = (shift < lowest_knobval) ? 0 : (long)(shift - lowest_knobval);
        if(shiftBin > splines[name][0].size() - 1) shiftBin = splines[name][0].size() - 1;
        // We should use the line below if we switch to c++17
        // const long shiftBin = std::clamp((long)(shift - knobval[0]), 0, spline_coeffs[0].size() - 1);
        std::array<float, 4> coeffs = splines[name][bin][shiftBin].second;
        shift -= splines[name][bin][shiftBin].first;
        return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROspec& cv, std::string name, float shift) {
        assert(cv.GetNbins() == splines[name].size());
        PROspec ret(cv.GetNbins());
        for(int i = 0; i < cv.GetNbins(); ++i)
            ret.Fill(i, GetSplineShift(name, shift, i) * cv.GetBinContent(i));
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROspec& cv, std::vector<std::string> names, std::vector<float> shifts) {
        assert(cv.GetNbins() == splines[names[0]].size());
        assert(names.size() == shifts.size());
        PROspec ret(cv.GetNbins());
        for(int i = 0; i < cv.GetNbins(); ++i) {
            float weight = 1;
            for(size_t j = 0; j < names.size(); ++j) {
                weight *= GetSplineShift(names[j], shifts[j], i);
            }
            ret.Fill(i, weight * cv.GetBinContent(i));
        }
        return ret;
    }
};

