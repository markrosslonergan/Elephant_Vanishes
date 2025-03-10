#include "PROfitter.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
#include "PROmetric.h"

#include <Eigen/Eigen>

#include <random>

using namespace PROfit;

static inline
std::vector<std::vector<float>> latin_hypercube_sampling(size_t num_samples, size_t dimensions, std::uniform_real_distribution<float>&dis, std::mt19937 &gen) {
    std::vector<std::vector<float>> samples(num_samples, std::vector<float>(dimensions));

    for (size_t d = 0; d < dimensions; ++d) {

        std::vector<float> perm(num_samples);
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
std::vector<int> sorted_indices(const std::vector<float>& vec) {
    std::vector<int> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&vec](int i1, int i2) { return vec[i1] < vec[i2]; });
    return indices;
}

float PROfitter::Fit(PROmetric &metric, const Eigen::VectorXf &seed_pt ) {
    // resetting the number of fit failures before starting
    n_failures = 0;
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);

    std::vector<std::vector<float>> latin_samples = latin_hypercube_sampling(n_multistart, ub.size(), d_uni,rng);

    //Rescale the latin hypercube now at -2 to 2, scale to real bounds.
    for(std::vector<float> &pt: latin_samples) {
        for(size_t i = 0; i < pt.size(); ++i) {
            if(ub(i) != 3 || lb(i) != -3) {
                float width = std::isinf(ub(i)) || std::isinf(lb(i)) ? 4 : ub(i) - lb(i);
                float center = std::isinf(ub(i)) ? lb(i) + width/2.0 :
                                std::isinf(lb(i)) ? ub(i) - width/2.0 :
                                (ub(i) + lb(i)) / 2.0;
                float randpt = pt[i] / 4.0;
                pt[i] = center + randpt * width;
            }
        }
    }
    std::vector<float> chi2s_multistart;
    chi2s_multistart.reserve(n_multistart);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs : %2%") % __func__ % n_multistart ;
    for(int s = 0; s < n_multistart; s++){
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXf grad = Eigen::VectorXf::Constant(x.size(), 0);
	log<LOG_DEBUG>(L"%1% || vectors set up") % __func__ ;	
        float fx =  metric(x, grad, false);
	log<LOG_DEBUG>(L"%1% || fx is  : %2% ") % __func__ % fx;
        chi2s_multistart.push_back(fx);
    }

    //Sort so we can take the best N_localfits for further zoning
    std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    

    std::string local_fits = "";
    for(int i = 0; i < n_localfit; ++i) {
        local_fits += " " + std::to_string(chi2s_multistart[best_multistart[i]]);
    }
    log<LOG_INFO>(L"%1% || Ending MultiGlobal. Best %2% are:%3%") % __func__ % n_localfit % local_fits.c_str();
    log<LOG_DEBUG>(L"%1% || Best Points is  : %2% ") % __func__ % latin_samples[best_multistart[0]];

    std::vector<float> chi2s_localfits;
    chi2s_localfits.reserve(n_localfit);
    float chimin = 9999999;

    log<LOG_INFO>(L"%1% || Starting Local Gradients runs : %2%") % __func__ % n_localfit ;
    for(int s = 0; s < n_localfit; s++){
        //Get the nth
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>( latin_samples[best_multistart[s]].data(), latin_samples[best_multistart[s]].size());   
        int niter=0;
        float fx;
        try {
            niter = solver.minimize(metric, x, fx, lb, ub);
        } catch(std::runtime_error &except) {
            log<LOG_WARNING>(L"%1% || Fit failed on niter,%2% : %3%") % __func__ % niter % except.what();
	    n_failures++;
        }
        chi2s_localfits.push_back(fx);
        if(fx<chimin){
            best_fit = x;
            chimin = fx;
        }
        log<LOG_INFO>(L"%1% ||  LocalGrad Run : %2% has a chi %3%") % __func__ % s % fx;
        std::string spec_string = "";
        for(auto &f : x) spec_string+=" "+std::to_string(f); 
        log<LOG_DEBUG>(L"%1% || Best Point is  : %2% ") % __func__ % spec_string.c_str();
    }


    // and do Seeded Point
    int niter=0;
    float fx;
    Eigen::VectorXf x;

    if(seed_pt.norm()>0){
        log<LOG_INFO>(L"%1% || Starting Seed fit ") % __func__  ;
        try {
            x = seed_pt;   
            niter = solver.minimize(metric, x, fx, lb, ub);
        } catch(std::runtime_error &except) {
            log<LOG_WARNING>(L"%1% || Fit failed, %2%") % __func__ % except.what();
	    n_failures++;
        }
        chi2s_localfits.push_back(fx);
        if(fx < chimin){
            best_fit = x;
            chimin = fx;
        }

        log<LOG_INFO>(L"%1% ||  Seed Run has a chi %2%") % __func__ %  fx;
        std::string spec_string = "";
        for(auto &f : x) spec_string+=" "+std::to_string(f); 
        log<LOG_DEBUG>(L"%1% || Best Point post Seed is  : %2% ") % __func__ % spec_string.c_str();
    }


    // and do CV
    log<LOG_INFO>(L"%1% || Starting CV fit x.size %2% fx %3% lb.size %4% ub.size %5%") % __func__ % x.size() % fx  % lb.size()  %ub.size() ;
    try {
        x = Eigen::VectorXf::Constant(ub.size(), 0.);
        niter = solver.minimize(metric, x, fx, lb, ub);
    } catch(std::runtime_error &except) {
        log<LOG_WARNING>(L"%1% || Fit failed, %2%") % __func__ % except.what();
	n_failures++;
    }
    chi2s_localfits.push_back(fx);
    if(fx < chimin){
        best_fit = x;
        chimin = fx;
    }

    log<LOG_INFO>(L"%1% ||  CV Run has a chi %2%") % __func__ %  fx;
    std::string spec_string = "";
    for(auto &f : x) spec_string+=" "+std::to_string(f); 
    log<LOG_DEBUG>(L"%1% || Best Point post CV is  : %2% ") % __func__ % spec_string.c_str();



    log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
    spec_string = "";
    for(auto &f : best_fit) spec_string+=" "+std::to_string(f); 
    log<LOG_DEBUG>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

    return chimin;
}

