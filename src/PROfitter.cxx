#include "PROfitter.h"
#include "PROlog.h"
#include "PROmetric.h"
#include "PROswarm.h"

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
    std::mt19937 rng;
    rng.seed(seed);
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
    if(seed_pt.norm()>0){
        std::vector<float> std_vec(seed_pt.data(), seed_pt.data() + seed_pt.size());
        latin_samples.push_back(std_vec);
    }



    std::vector<float> chi2s_multistart;
    chi2s_multistart.reserve(n_multistart);

    log<LOG_INFO>(L"%1% || Starting MultiGlobal runs : %2%") % __func__ % n_multistart ;
    for(int s = 0; s < n_multistart; s++){
        Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>(latin_samples[s].data(), latin_samples[s].size());
        Eigen::VectorXf grad = Eigen::VectorXf::Constant(x.size(), 0);
        float fx =  metric(x, grad, false);
        chi2s_multistart.push_back(fx);
    }
    //Sort so we can take the best N_localfits for further zoning with a PSO
    std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    

    log<LOG_DEBUG>(L"%1% || Best Point is  : %2% ") % __func__ % latin_samples[best_multistart[0]];

    std::vector<std::vector<float>> swarm_start_points;
    int niter=0;
    float fx;
    for(int s = 0; s < n_swarm_particles; s++){
        swarm_start_points.push_back(latin_samples[best_multistart[s]]);
    }

    PROswarm PSO(metric, rng, swarm_start_points, lb, ub , n_swarm_iterations);
    PSO.runSwarm(metric, rng);
    Eigen::VectorXf x;  

    float chimin = 9999999;
    std::vector<float> chi2s_localfits;
    niter=0;

    bool success = false;

    x = PSO.getGlobalBestPosition();

    for (size_t attempt = 1; attempt <= n_max_local_retries; ++attempt) {
        try {
            log<LOG_INFO>(L"%1% || Starting local minimization attempt %2%/%3%") % __func__ % attempt % n_max_local_retries;
            niter = solver.minimize(metric, x, fx, lb, ub);
            chi2s_localfits.push_back(fx);

            if (fx < chimin) {
                best_fit = x;
                chimin = fx;
            }

            log<LOG_INFO>(L"%1% || Minimization successful, chi %2% after %3% iterations") % __func__ % fx % niter;

            std::string spec_string = "";
            for (auto &f : x) spec_string += " " + std::to_string(f);
            log<LOG_DEBUG>(L"%1% || Best Point after minimization: %2%") % __func__ % spec_string.c_str();

            success = true;
            break;

        } catch (const std::runtime_error &except) {
            log<LOG_WARNING>(L"%1% || Minimization attempt %2%/%3% failed: %4%") % __func__ % attempt % n_max_local_retries % except.what();
        }
    }

    if (!success) {
        log<LOG_WARNING>(L"%1% || All minimization attempts failed, falling back to PSO best") % __func__;
        best_fit = PSO.getGlobalBestPosition();
        chimin = PSO.getGlobalBestScore();
    }


    /*
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
    for(int i = 0; i < n_swarm_particles; ++i) {
    local_fits += " " + std::to_string(chi2s_multistart[best_multistart[i]]);
    }
    log<LOG_INFO>(L"%1% || Ending MultiGlobal. Best %2% are:%3%") % __func__ % n_swarm_particles % local_fits.c_str();
    log<LOG_DEBUG>(L"%1% || Best Points is  : %2% ") % __func__ % latin_samples[best_multistart[0]];

    std::vector<float> chi2s_localfits;
    chi2s_localfits.reserve(n_swarm_particles);
    float chimin = 9999999;

    int niter=0;
    float fx;
    log<LOG_INFO>(L"%1% || Starting Local Gradients runs : %2%") % __func__ % n_swarm_particles ;
    for(int s = 0; s < n_swarm_particles; s++){
//Get the nth
Eigen::VectorXf x = Eigen::Map<Eigen::VectorXf>( latin_samples[best_multistart[s]].data(), latin_samples[best_multistart[s]].size());   
try {
niter = solver.minimize(metric, x, fx, lb, ub);
} catch(std::runtime_error &except) {
log<LOG_WARNING>(L"%1% || Fit failed on: %2%") % __func__ % except.what();
continue;
}
chi2s_localfits.push_back(fx);
if(fx<chimin){
best_fit = x;
chimin = fx;
}
log<LOG_INFO>(L"%1% ||  LocalGrad Run : %2% has a chi %3% after %4% iterations") % __func__ % s % fx % niter;
std::string spec_string = "";
for(auto &f : x) spec_string+=" "+std::to_string(f); 
log<LOG_DEBUG>(L"%1% || Best Point is  : %2% ") % __func__ % spec_string.c_str();
}
*/


// and do Seeded Point
/*if(seed_pt.norm()>0){
  log<LOG_INFO>(L"%1% || Starting Seed fit ") % __func__  ;
  try {
  x = seed_pt;   
  niter = solver.minimize(metric, x, fx, lb, ub);
  chi2s_localfits.push_back(fx);
  if(fx < chimin){
  best_fit = x;
  chimin = fx;
  }

  log<LOG_INFO>(L"%1% ||  Seed Run has a chi %2% after %3% iterations") % __func__ %  fx % niter;
  std::string spec_string = "";
  for(auto &f : x) spec_string+=" "+std::to_string(f); 
  log<LOG_DEBUG>(L"%1% || Best Point post Seed is  : %2% ") % __func__ % spec_string.c_str();
  } catch(std::runtime_error &except) {
  log<LOG_WARNING>(L"%1% || Fit failed, %2%") % __func__ % except.what();
  }
  }
  */

// and do CV
/*

   log<LOG_INFO>(L"%1% || Starting CV fit x.size %2% fx %3% lb.size %4% ub.size %5%") % __func__ % x.size() % fx  % lb.size()  %ub.size() ;
   try {
   x = Eigen::VectorXf::Constant(ub.size(), 0.012);
   niter = solver.minimize(metric, x, fx, lb, ub);
   chi2s_localfits.push_back(fx);
   if(fx < chimin){
   best_fit = x;
   chimin = fx;
   }

   log<LOG_INFO>(L"%1% ||  CV Run has a chi %2% after %3% iterations") % __func__ %  fx % niter;
   std::string spec_string = "";
   for(auto &f : x) spec_string+=" "+std::to_string(f); 
   log<LOG_DEBUG>(L"%1% || Best Point post CV is  : %2% ") % __func__ % spec_string.c_str();
   } catch(std::runtime_error &except) {
   log<LOG_WARNING>(L"%1% || Fit failed, %2%") % __func__ % except.what();
   }
   */

log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
std::string spec_string = "";
for(auto &f : best_fit) spec_string+=" "+std::to_string(f); 
log<LOG_DEBUG>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

return chimin;
}

