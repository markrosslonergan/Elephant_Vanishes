#ifndef PROMODEL_H
#define PROMODEL_H

#include "PROpeller.h"

#include <Eigen/Eigen>

#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace PROfit {

class PROmodel {
public:
    size_t nparams;
    std::vector<std::string> param_names;
    std::vector<std::string> pretty_param_names;
    Eigen::VectorXf lb, ub, default_val;
    std::vector<std::function<float(const Eigen::VectorXf&, float)>> model_functions;
    std::vector<Eigen::MatrixXf> hists; //2D hists for binned oscilattions
};

class NullModel : public PROmodel {
public:
    NullModel(const PROpeller &prop) {
        nparams = 0;
        model_functions.push_back([](const Eigen::VectorXf &, float){ return 1.0f; });

        hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
        Eigen::MatrixXf &h = hists.back();
        for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
            int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
            h(tbin, rbin) += prop.added_weights[i];
        }
    }
};

class PROnumudis : public PROmodel {
public:
    PROnumudis(const PROpeller &prop) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),le);});

        for(size_t m = 0; m < model_functions.size(); ++m) {
            hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
            Eigen::MatrixXf &h = hists.back();
            for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
                if(prop.model_rule[i] != (int)m) continue;
                int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
                h(tbin, rbin) += prop.added_weights[i];
            }
        }

        nparams = 2;
        param_names = {"dmsq", "sinsq2thmm"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu#mu}"}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -10, -10;
    };

    /* Function: 3+1 numu->numue disapperance prob in SBL approx */
    float Pmumu(float dmsq, float sinsq2thmumu, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmumu = std::pow(10.0f, sinsq2thmumu);

        if(sinsq2thmumu > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is greater than 1. Setting to 1.")     % __func__ % sinsq2thmumu;
            sinsq2thmumu = 1;
        }
        if(sinsq2thmumu < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmumu is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmumu;
            sinsq2thmumu = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = 1.0f - (sinsq2thmumu*sinterm*sinterm);

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmumu = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmumu % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PROnueapp : public PROmodel {
public:
    PROnueapp(const PROpeller &prop) {
        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0;});
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),le);});

        for(size_t m = 0; m < model_functions.size(); ++m) {
            hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
            Eigen::MatrixXf &h = hists.back();
            for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
                if(prop.model_rule[i] != (int)m) continue;
                int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
                h(tbin, rbin) += prop.added_weights[i];
            }
        }

        nparams = 2;
        param_names = {"dmsq", "sinsq2thme"}; 
        pretty_param_names = {"#Deltam^{2}", "sin^{2}2#theta_{#mu{e}}"}; 
        lb = Eigen::VectorXf(2);
        ub = Eigen::VectorXf(2);
        default_val = Eigen::VectorXf(2);
        lb << -2, -std::numeric_limits<float>::infinity();
        ub << 2, 0;
        default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    };

    float Pmue(float dmsq, float sinsq2thmue, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        sinsq2thmue = std::pow(10.0f, sinsq2thmue);

        if(sinsq2thmue > 1) {
            //log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is greater than 1. Setting to 1.")  % __func__ % sinsq2thmue;
            sinsq2thmue = 1;
        }
        if(sinsq2thmue < 0) {
            log<LOG_ERROR>(L"%1% || sinsq2thmue is %2% which is less than 0. Setting to 0.")
                % __func__ % sinsq2thmue;
            sinsq2thmue = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = sinsq2thmue*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, sinsq2thmue = %4%, L/E = %5%")
                % __func__ % prob % dmsq % sinsq2thmue % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PRO3p1 : public PROmodel {
public:
    PRO3p1(const PROpeller &prop) {

        model_functions.push_back([this]([[maybe_unused]] const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v(0),v(1),v(2),le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v(0),v(1),v(2),le); });

        for(size_t m = 0; m < model_functions.size(); ++m) {
            hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
            Eigen::MatrixXf &h = hists.back();
            for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
                if(prop.model_rule[i] != (int)m) continue;
                int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
                h(tbin, rbin) += prop.added_weights[i];
            }
        }

        nparams = 3;
        param_names = {"dmsq", "Ue4^2", "Um4^2"}; 
        pretty_param_names = {"dmsq", "Ue4^2", "Um4^2"}; 
        lb = Eigen::VectorXf(3);
        ub = Eigen::VectorXf(3);
        default_val = Eigen::VectorXf(3);
        lb << -2, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
        ub << 2, 0, 0;
        default_val << -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity();
    };

    float Pmue(float dmsq, float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);
        Um4sq = std::pow(10.0f, Um4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }
        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.") 
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.27f*dmsq*(le));
        float prob    = 4.0f*Ue4sq*Um4sq*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math."
                           L"dmsq = %3%, Ue4sq = %4%, Um4sq = %5%, L/E = %6%")
                % __func__ % prob % dmsq % Ue4sq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pmumu(float dmsq, [[maybe_unused]]float Ue4sq, float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Um4sq = std::pow(10.0f, Um4sq);

        if(Um4sq > 1) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Um4sq;
            Um4sq = 1;
        }
        if(Um4sq < 0) {
            log<LOG_ERROR>(L"%1% || Um4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Um4sq;
            Um4sq = 0;
        }

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Um4sq*(1.0f-Um4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Um4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Um4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }

    float Pee(float dmsq, float Ue4sq, [[maybe_unused]]float Um4sq, float le) const{
        dmsq = std::pow(10.0f, dmsq);
        Ue4sq = std::pow(10.0f, Ue4sq);

        if(Ue4sq > 1) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is greater than 1. Setting to 1.")
                % __func__ % Ue4sq;
            Ue4sq = 1;
        }
        if(Ue4sq < 0) {
            log<LOG_ERROR>(L"%1% || Ue4sq is %2% which is less than 0. Setting to 0.")
                % __func__ % Ue4sq;
            Ue4sq = 0;
        }

        float sinterm = std::sin(1.27*dmsq*(le));
        float prob    = 1.0f - 4.0f*Ue4sq*(1.0f-Ue4sq)*sinterm*sinterm;

        if(prob<0.0 || prob >1.0){
            log<LOG_ERROR>(L"%1% || Your probability %2% is outside the bounds of math. dmsq = %3%, Ue4sq = %4%, L/E = %5%") % __func__ % prob % dmsq % Ue4sq % le;
            log<LOG_ERROR>(L"%1% || Terminating.") % __func__;
            exit(EXIT_FAILURE);
        }

        return prob;
    }
};

class PROLBL : public PROmodel {
public:
    static constexpr float rho_earth = 3; // g/cc
	static constexpr float Ye_earth = 0.5;
    static constexpr float eVsqkm_to_GeV_over4 = 1e-9 / 1.97327e-7 * 1e3 / 4;
    static constexpr float YerhoE2a = 1.52588e-4;

    PROLBL(const PROpeller &prop) {
        model_functions.push_back([this](const Eigen::VectorXf &v, float) {(void)this; return 1.0; });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pee(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pemu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Petau(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmue(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmumu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Pmutau(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptaue(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptaumu(v,le); });
        model_functions.push_back([this](const Eigen::VectorXf &v, float le) {return this->Ptautau(v,le); });

        for(size_t m = 0; m < model_functions.size(); ++m) {
            hists.emplace_back(Eigen::MatrixXf::Constant(prop.hist.rows(), prop.hist.cols(),0.0));
            Eigen::MatrixXf &h = hists.back();
            for(size_t i = 0; i < prop.bin_indices.size(); ++i) {
                if(prop.model_rule[i] != (int)m) continue;
                int tbin = prop.true_bin_indices[i], rbin = prop.bin_indices[i];
                h(tbin, rbin) += prop.added_weights[i];
            }
        }

        nparams = 6;
        param_names = {"dmsq_21", "dmsq_31", "sinsqt12", "sinsqt13", "sinsqt23", "delta_CP"}; 
        pretty_param_names = {"#Delta m^{2}_{21}", "#Delta m^{2}_{31}", "sin^{2}#theta_{12}",
            "sin^2#theta_{13}", "sin^{2}#theta_{23}", "delta_{CP}"}; 
        lb = Eigen::VectorXf(6);
        ub = Eigen::VectorXf(6);
        lb << 6e-5f, -3e-3f, 0.2f, 0.01f, 0.3f, -M_PI;
        ub << 9e-5f, 3e-3f, 0.4f, 0.04f, 0.7f, M_PI;
    }

    float Pee(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][0];
    }
    
    float Pemu(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][1];
    }
    
    float Petau(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[0][2];
    }
    
    float Pmue(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][0];
    }
    
    float Pmumu(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][1];
    }
    
    float Pmutau(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[1][2];
    }
    
    float Ptaue(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][0];
    }
    
    float Ptaumu(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][1];
    }
    
    float Ptautau(const Eigen::VectorXf &params, float le) {
        double probs_returned[3][3];
        Probability_Matter_LBL(params(2), params(3), params(4), params(5), 
                  params(0), params(1), le, 1.0, rho_earth, Ye_earth, 0, &probs_returned);
        return probs_returned[2][2];
    }
    
    // Taken from NuFast https://github.com/PeterDenton/NuFast/blob/main/c%2B%2B/NuFast.cpp
    void Probability_Matter_LBL(double s12sq, double s13sq, double s23sq, double delta, double Dmsq21, double Dmsq31, double L, double E, double rho, double Ye, int N_Newton, double (*probs_returned)[3][3])
    {
        double c13sq, sind, cosd, Jrr, Jmatter, Dmsqee, Amatter;
        double Ue1sq, Ue2sq, Ue3sq, Um1sq, Um2sq, Um3sq, Ut1sq, Ut2sq, Ut3sq;
        double A, B, C;
        double See, Tee, Smm, Tmm;
        double xmat, lambda2, lambda3, Dlambda21, Dlambda31, Dlambda32;
        double Xp2, Xp3, PiDlambdaInv, tmp;
        double Lover4E, D21, D32;
        double sinD21, sinD31, sinD32;
        double sinsqD21_2, sinsqD31_2, sinsqD32_2, triple_sin;
        double Pme_CPC, Pme_CPV, Pmm, Pee;

        // --------------------------------------------------------------------- //
        // First calculate useful simple functions of the oscillation parameters //
        // --------------------------------------------------------------------- //
        c13sq = 1 - s13sq;

        // Ueisq's
        Ue2sq = c13sq * s12sq;
        Ue3sq = s13sq;

        // Umisq's, Utisq's and Jvac	 
        Um3sq = c13sq * s23sq;
        // Um2sq and Ut2sq are used here as temporary variables, will be properly defined later	 
        Ut2sq = s13sq * s12sq * s23sq;
        Um2sq = (1 - s12sq) * (1 - s23sq);

        Jrr = sqrt(Um2sq * Ut2sq);
        sind = sin(delta);
        cosd = cos(delta);

        Um2sq = Um2sq + Ut2sq - 2 * Jrr * cosd;
        Jmatter = 8 * Jrr * c13sq * sind;
        Amatter = Ye * rho * E * YerhoE2a;
        Dmsqee = Dmsq31 - s12sq * Dmsq21;

        // calculate A, B, C, See, Tee, and part of Tmm
        A = Dmsq21 + Dmsq31; // temporary variable
        See = A - Dmsq21 * Ue2sq - Dmsq31 * Ue3sq;
        Tmm = Dmsq21 * Dmsq31; // using Tmm as a temporary variable	  
        Tee = Tmm * (1 - Ue3sq - Ue2sq);
        C = Amatter * Tee;
        A = A + Amatter;

        // ---------------------------------- //
        // Get lambda3 from lambda+ of MP/DMP //
        // ---------------------------------- //
        xmat = Amatter / Dmsqee;
        tmp = 1 - xmat;
        lambda3 = Dmsq31 + 0.5 * Dmsqee * (xmat - 1 + sqrt(tmp * tmp + 4 * s13sq * xmat));

        // ---------------------------------------------------------------------------- //
        // Newton iterations to improve lambda3 arbitrarily, if needed, (B needed here) //
        // ---------------------------------------------------------------------------- //
        B = Tmm + Amatter * See; // B is only needed for N_Newton >= 1
        for (int i = 0; i < N_Newton; i++)
            lambda3 = (lambda3 * lambda3 * (lambda3 + lambda3 - A) + C) / (lambda3 * (2 * (lambda3 - A) + lambda3) + B); // this strange form prefers additions to multiplications

        // ------------------- //
        // Get  Delta lambda's //
        // ------------------- //
        tmp = A - lambda3;
        Dlambda21 = sqrt(tmp * tmp - 4 * C / lambda3);
        lambda2 = 0.5 * (A - lambda3 + Dlambda21);
        Dlambda32 = lambda3 - lambda2;
        Dlambda31 = Dlambda32 + Dlambda21;

        // ----------------------- //
        // Use Rosetta for Veisq's //
        // ----------------------- //
        // denominators	  
        PiDlambdaInv = 1 / (Dlambda31 * Dlambda32 * Dlambda21);
        Xp3 = PiDlambdaInv * Dlambda21;
        Xp2 = -PiDlambdaInv * Dlambda31;

        // numerators
        Ue3sq = (lambda3 * (lambda3 - See) + Tee) * Xp3;
        Ue2sq = (lambda2 * (lambda2 - See) + Tee) * Xp2;

        Smm = A - Dmsq21 * Um2sq - Dmsq31 * Um3sq;
        Tmm = Tmm * (1 - Um3sq - Um2sq) + Amatter * (See + Smm - A);

        Um3sq = (lambda3 * (lambda3 - Smm) + Tmm) * Xp3;
        Um2sq = (lambda2 * (lambda2 - Smm) + Tmm) * Xp2;

        // ------------- //
        // Use NHS for J //
        // ------------- //
        Jmatter = Jmatter * Dmsq21 * Dmsq31 * (Dmsq31 - Dmsq21) * PiDlambdaInv;

        // ----------------------- //
        // Get all elements of Usq //
        // ----------------------- //
        Ue1sq = 1 - Ue3sq - Ue2sq;
        Um1sq = 1 - Um3sq - Um2sq;

        Ut3sq = 1 - Um3sq - Ue3sq;
        Ut2sq = 1 - Um2sq - Ue2sq;
        Ut1sq = 1 - Um1sq - Ue1sq;

        // ----------------------- //
        // Get the kinematic terms //
        // ----------------------- //
        Lover4E = eVsqkm_to_GeV_over4 * L / E;

        D21 = Dlambda21 * Lover4E;
        D32 = Dlambda32 * Lover4E;
          
        sinD21 = sin(D21);
        sinD31 = sin(D32 + D21);
        sinD32 = sin(D32);

        triple_sin = sinD21 * sinD31 * sinD32;

        sinsqD21_2 = 2 * sinD21 * sinD21;
        sinsqD31_2 = 2 * sinD31 * sinD31;
        sinsqD32_2 = 2 * sinD32 * sinD32;

        // ------------------------------------------------------------------- //
        // Calculate the three necessary probabilities, separating CPC and CPV //
        // ------------------------------------------------------------------- //
        Pme_CPC = (Ut3sq - Um2sq * Ue1sq - Um1sq * Ue2sq) * sinsqD21_2
                + (Ut2sq - Um3sq * Ue1sq - Um1sq * Ue3sq) * sinsqD31_2
                + (Ut1sq - Um3sq * Ue2sq - Um2sq * Ue3sq) * sinsqD32_2;
        Pme_CPV = -Jmatter * triple_sin;

        Pmm = 1 - 2 * (Um2sq * Um1sq * sinsqD21_2
                     + Um3sq * Um1sq * sinsqD31_2
                     + Um3sq * Um2sq * sinsqD32_2);

        Pee = 1 - 2 * (Ue2sq * Ue1sq * sinsqD21_2
                     + Ue3sq * Ue1sq * sinsqD31_2
                     + Ue3sq * Ue2sq * sinsqD32_2);

        // ---------------------------- //
        // Assign all the probabilities //
        // ---------------------------- //
        (*probs_returned)[0][0] = Pee;														// Pee
        (*probs_returned)[0][1] = Pme_CPC - Pme_CPV;										// Pem
        (*probs_returned)[0][2] = 1 - Pee - (*probs_returned)[0][1];  						// Pet

        (*probs_returned)[1][0] = Pme_CPC + Pme_CPV;										// Pme
        (*probs_returned)[1][1] = Pmm;														// Pmm
        (*probs_returned)[1][2] = 1 - (*probs_returned)[1][0] - Pmm;						// Pmt

        (*probs_returned)[2][0] = 1 - Pee - (*probs_returned)[1][0];						// Pte
        (*probs_returned)[2][1] = 1 - (*probs_returned)[0][1] - Pmm;						// Ptm
        (*probs_returned)[2][2] = 1 - (*probs_returned)[0][2] - (*probs_returned)[1][2];	// Ptt
    }
};

// Main interface to different models
static inline
std::unique_ptr<PROmodel> get_model_from_string(const std::string &name, const PROpeller &prop) {
    if(name == "numudis") {
        return std::unique_ptr<PROmodel>(new PROnumudis(prop));
    } else if(name == "nueapp") {
        return std::unique_ptr<PROmodel>(new PROnueapp(prop));
    } else if(name == "3+1") {
        return std::unique_ptr<PROmodel>(new PRO3p1(prop));
    } else if(name == "LBL") {
        return std::unique_ptr<PROmodel>(new PROLBL(prop));
    }
    log<LOG_ERROR>(L"%1% || Unrecognized model name %2%. Try numudis, nueapp or 3+1 for now. Terminating.") % __func__ % name.c_str();
    exit(EXIT_FAILURE);
}

}

#endif

