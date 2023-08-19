#include "PROsyst.h"
#include "PROconfig.h"
#include "PROcreate.h"
#include "PROtocall.h"

namespace PROfit {

    PROsyst::PROsyst(const std::vector<SystStruct>& systs) {
        for(const auto& syst: systs) {
            if(syst.mode == "multisigma") {
                FillSpline(syst);
            } else if(syst.mode == "multisim") {
                this->CreateMatrix(syst);
            }
        }
        fractional_covariance = this->SumMatrices();
    }


    Eigen::MatrixXd PROsyst::SumMatrices() const{

        Eigen::MatrixXd sum_matrix;
        if(covmat.size()){
            int nbins = (covmat.begin())->rows();
            sum_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
            for(auto& p : covmat){
                sum_matrix += p;
            }
        }else{
            log<LOG_ERROR>(L"%1% || There is no covariance available!") % __func__;
            log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;
        }
        return sum_matrix;
    }

    Eigen::MatrixXd PROsyst::SumMatrices(const std::vector<std::string>& sysnames) const{

        Eigen::MatrixXd sum_matrix;
        if(covmat.size()){
            int nbins = (covmat.begin())->rows();
            sum_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
        }
        else{
            log<LOG_ERROR>(L"%1% || There is no covariance available!") % __func__;
            log<LOG_ERROR>(L"%1% || Returning empty matrix") % __func__;
            return sum_matrix;
        }


        for(auto& sys : sysnames){
            if(syst_map.find(sys) == syst_map.end() || syst_map.at(sys).second != SystType::Covariance){
                log<LOG_INFO>(L"%1% || No matrix in the map matches with name %2%, Skip") % __func__ % sys.c_str();
            }else{
                sum_matrix += covmat.at(syst_map.at(sys).first);
            }
        }

        return sum_matrix;
    }

    void PROsyst::CreateMatrix(const SystStruct& syst){

        std::string sysname = syst.GetSysName();

        //generate matrix only if it's not already in the map 
        if(syst_map.find(sysname) == syst_map.end()){
            std::pair<Eigen::MatrixXd, Eigen::MatrixXd> matrices = PROsyst::GenerateCovarMatrices(syst);
            syst_map[sysname] = {covmat.size(), SystType::Covariance};
            covmat.push_back(matrices.first);
            corrmat.push_back(matrices.second);

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

    Eigen::MatrixXd PROsyst::GenerateFullCovarMatrix(const SystStruct& sys_obj){
        int n_universe = sys_obj.GetNUniverse(); 
        std::string sys_name = sys_obj.GetSysName();

        const PROspec& cv_spec = sys_obj.CV();
        int nbins = cv_spec.GetNbins();
        log<LOG_INFO>(L"%1% || Generating covariance matrix.. size: %2% x %3%") % __func__ % nbins % nbins;

        //build full covariance matrix 
        Eigen::MatrixXd full_covar_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
        for(int i = 0; i != n_universe; ++i){
            PROspec spec_diff  = cv_spec - sys_obj.Variation(i);
        log<LOG_DEBUG>(L"%1% || Check univdrse %2%") % __func__ % i;
            full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<double>(n_universe);
        }

	return full_covar_matrix;
    }

    Eigen::MatrixXd PROsyst::GenerateFracCovarMatrix(const SystStruct& sys_obj){

        //build full covariance matrix 
        Eigen::MatrixXd full_covar_matrix = PROsyst::GenerateFullCovarMatrix(sys_obj);

        //build fractional covariance matrix 
        //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
        const PROspec& cv_spec = sys_obj.CV();
        Eigen::MatrixXd cv_spec_matrix =  Eigen::MatrixXd::Identity(nbins, nbins);
        for(int i =0; i != nbins; ++i){
	double pred = cv_spec.GetBinContent(i);
	   log<LOG_DEBUG>(L"%1% || CV prediction at bin %2% is %3% ") % __func__ % i % pred ; 
            cv_spec_matrix(i, i) = 1.0/cv_spec.GetBinContent(i);
 	}

	log<LOG_DEBUG>(L"%1%  || check %2% ") % __func__ % __LINE__;
        //second, get fractioal covar
        Eigen::MatrixXd frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;

	log<LOG_DEBUG>(L"%1%  || check %2% ") % __func__ % __LINE__;

        //check if it's good
        if(!PROsyst::isPositiveSemiDefinite_WithTolerance(frac_covar_matrix)){
            log<LOG_ERROR>(L"%1% || Fractional Covariance Matrix is not positive semi-definite!") % __func__;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

	log<LOG_DEBUG>(L"%1%  || check %2% ") % __func__ % __LINE__;
        //zero out nans 
        PROsyst::toFiniteMatrix(frac_covar_matrix);

	log<LOG_DEBUG>(L"%1%  || check %2% ") % __func__ % __LINE__;
        return frac_covar_matrix;
    }

    Eigen::MatrixXd PROsyst::GenerateCorrMatrix(const Eigen::MatrixXd& frac_matrix){
        int nbins = frac_matrix.rows();
        Eigen::MatrixXd corr_covar_matrix = frac_matrix;

        Eigen::MatrixXd error_reciprocal_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
        for(int i = 0; i != nbins; ++i){
            if(frac_matrix(i,i) != 0){
		double temp = sqrt(frac_matrix(i,i));
                error_reciprocal_matrix(i,i) = 1.0/temp;
 	    }
	    else
		error_reciprocal_matrix(i,i) = 1.0;
        }


        corr_covar_matrix = error_reciprocal_matrix * corr_covar_matrix * error_reciprocal_matrix;

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
            spline.push_back({syst.knobval[syst.knobval.size() - 2], {resp(2), resp(1), resp(0), 0}});

            spline_coeffs.push_back(spline);
        }
        syst_map[syst.systname] = {splines.size(), SystType::Spline};
        splines.push_back(spline_coeffs);
    }

    float PROsyst::GetSplineShift(int spline_num, float shift , int bin) const {
        if(bin < 0 || bin >= splines[spline_num].size()) return -1;
        const float lowest_knobval = splines[spline_num][0][0].first;
        int shiftBin = (shift < lowest_knobval) ? 0 : (int)(shift - lowest_knobval);
        if(shiftBin > splines[spline_num][0].size() - 1) shiftBin = splines[spline_num][0].size() - 1;
        // We should use the line below if we switch to c++17
        // const long shiftBin = std::clamp((int)(shift - lowest_knobval), 0, splines[spline_num][0].size() - 1);
        std::array<float, 4> coeffs = splines[spline_num][bin][shiftBin].second;
        shift -= splines[spline_num][bin][shiftBin].first;
        return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
    }

    float PROsyst::GetSplineShift(std::string name, float shift, int bin) const {
        return GetSplineShift(syst_map.at(name).first, shift, bin);
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::string name, float shift) {
        PROspec ret(config.m_num_bins_total);
        for(size_t i = 0; i < prop.baseline.size(); ++i) {
            const int subchannel = FindSubchannelIndexFromGlobalBin(config, prop.bin_indices[i]);
            const int true_bin = FindGlobalTrueBin(config, prop.baseline[i] / prop.truth[i], subchannel);
            ret.Fill(prop.bin_indices[i], GetSplineShift(name, shift, true_bin) * prop.added_weights[i]);
        }
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROconfig& config, const PROpeller& prop, std::vector<std::string> names, std::vector<float> shifts) {
        assert(names.size() == shifts.size());
        PROspec ret(config.m_num_bins_total);
        for(size_t i = 0; i < prop.baseline.size(); ++i) {
            const int subchannel = FindSubchannelIndexFromGlobalBin(config, prop.bin_indices[i]);
            const int true_bin = FindGlobalTrueBin(config, prop.baseline[i] / prop.truth[i], subchannel);
            float weight = 1;
            for(size_t j = 0; j < names.size(); ++j) {
                weight *= GetSplineShift(names[j], shifts[j], true_bin);
            }
            ret.Fill(prop.bin_indices[i], weight * prop.added_weights[i]);
        }
        return ret;
    }

    Eigen::MatrixXd PROsyst::GrabMatrix(const std::string& sys) const{
        if(syst_map.find(sys) != syst_map.end())
            return covmat.at(syst_map.at(sys).first);	
        else{
            log<LOG_ERROR>(L"%1% || Systematic you asked for : %2% doesn't have matrix saved yet..") % __func__ % sys.c_str();
            log<LOG_ERROR>(L"%1% || Return empty matrix .") % __func__ ;
            return Eigen::MatrixXd();
        }
    }

    PROsyst::Spline PROsyst::GrabSpline(const std::string& sys) const{
        if(syst_map.find(sys) != syst_map.end())
            return splines.at(syst_map.at(sys).first);	
        else{
            log<LOG_ERROR>(L"%1% || Systematic you asked for : %2% doesn't have spline saved yet..") % __func__ % sys.c_str();
            return {};
        }
    }

    PROsyst::SystType PROsyst::GetSystType(const std::string &syst) {
        return syst_map.at(syst).second;
    }
};

