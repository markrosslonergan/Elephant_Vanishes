#include "PROfitter.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
#include "PROmetric.h"

#include <Eigen/Eigen>

#include <random>

using namespace PROfit;

static inline
std::vector<std::vector<double>> latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen) {
    std::vector<std::vector<double>> samples(num_samples, std::vector<double>(dimensions));

    for (size_t d = 0; d < dimensions; ++d) {

        std::vector<double> perm(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            perm[i] = (i + dis(gen)) / num_samples;  
        }
        std::shuffle(perm.begin(), perm.end(), gen);  
        for (size_t i = 0; i < num_samples; ++i) {
            samples[i][d] = perm[i]; 
        }
    }

    return samples;
}

static inline
std::vector<int> sorted_indices(const std::vector<double>& vec) {
    std::vector<int> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&vec](int i1, int i2) { return vec[i1] < vec[i2]; });
    return indices;
}

double PROfitter::Fit(PROmetric &metric) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);

    std::vector<std::vector<double>> latin_samples = latin_hypercube_sampling(n_multistart, ub.size(), d_uni,rng);
    for(std::vector<double> &pt: latin_samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            if(ub(i) != 3 || lb(i) != -3) {
                double width = std::isinf(ub(i)) || std::isinf(lb(i)) ? 4 : ub(i) - lb(i);
                double center = std::isinf(ub(i)) ? lb(i) + width/2.0 :
                                std::isinf(lb(i)) ? ub(i) - width/2.0 :
                                (ub(i) + lb(i)) / 2.0;
                double randpt = pt[i] / 4.0;
                pt[i] = center + randpt * width;
            }
        }
    }
    std::vector<double> chi2s_multistart;
    chi2s_multistart.reserve(n_multistart);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs : %2%") % __func__ % n_multistart ;
    for(int s = 0; s < n_multistart; s++){
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXd grad = Eigen::VectorXd::Constant(x.size(), 0);
        double fx =  metric(x, grad, false);
        chi2s_multistart.push_back(fx);

    }
    //Sort so we can take the best N_localfits for further zoning
    std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    

    log<LOG_INFO>(L"%1% || Ending MultiGlobal Best two are : %2% and %3%") % __func__ % chi2s_multistart[best_multistart[0]] %   chi2s_multistart[best_multistart[1]];
    log<LOG_INFO>(L"%1% || Best Points is  : %2% ") % __func__ % latin_samples[best_multistart[0]];

    std::vector<double> chi2s_localfits;
    chi2s_localfits.reserve(n_localfit);
    double chimin = 9999999;

    log<LOG_INFO>(L"%1% || Starting Local Gradients runs : %2%") % __func__ % n_localfit ;
    for(int s = 0; s < n_localfit; s++){
        //Get the nth
        Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>( latin_samples[best_multistart[s]].data(), latin_samples[best_multistart[s]].size());   
        int niter;
        double fx;
        try {
            niter = solver.minimize(metric, x, fx, lb, ub);
        } catch(std::runtime_error &except) {
            log<LOG_ERROR>(L"%1% || Fit failed on niter,%2% : %3%") % __func__ % niter % except.what();
        }
        chi2s_localfits.push_back(fx);
        if(fx<chimin){
            best_fit = x;
            chimin = fx;
        }
        log<LOG_INFO>(L"%1% ||  LocalGrad Run : %2% has a chi %3%") % __func__ % s % fx;
        std::string spec_string = "";
        for(auto &f : x) spec_string+=" "+std::to_string(f); 
        log<LOG_INFO>(L"%1% || Best Point is  : %2% ") % __func__ % spec_string.c_str();
    }


    // and do CV
    log<LOG_INFO>(L"%1% || Starting CV fit ") % __func__  ;
    int niter;
    double fx;
    Eigen::VectorXd x;
    try {
        x = Eigen::VectorXd::Constant(best_fit.size(), 0.012);
        niter = solver.minimize(metric, x, fx, lb, ub);
    } catch(std::runtime_error &except) {
        log<LOG_ERROR>(L"%1% || Fit failed, %2%") % __func__ % except.what();
    }
    chi2s_localfits.push_back(fx);
    if(fx < chimin){
        best_fit = x;
        chimin = fx;
    }

    log<LOG_INFO>(L"%1% ||  CV Run has a chi %2%") % __func__ %  fx;
    std::string spec_string = "";
    for(auto &f : x) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || Best Point post CV is  : %2% ") % __func__ % spec_string.c_str();



    log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
    spec_string = "";
    for(auto &f : best_fit) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

    return chimin;
}

