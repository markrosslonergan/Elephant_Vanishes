#include "PROsurf.h"

using namespace PROfit;

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

std::vector<int> sorted_indices(const std::vector<double>& vec) {
    std::vector<int> indices(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&vec](int i1, int i2) { return vec[i1] < vec[i2]; });
    return indices;
}



PROsurf::PROsurf(size_t nbinsx, LogLin llx, double x_lo, double x_hi, size_t nbinsy, LogLin lly, double y_lo, double y_hi) : nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXd::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXd::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
    if(llx == LogAxis) {
        x_lo = std::log10(x_lo);
        x_hi = std::log10(x_hi);
    }
    if(lly == LogAxis) {
        y_lo = std::log10(y_lo);
        y_hi = std::log10(y_hi);
    }
    for(size_t i = 0; i < nbinsx + 1; i++)
        edges_x(i) = x_lo + i * (x_hi - x_lo) / nbinsx;
    for(size_t i = 0; i < nbinsy + 1; i++)
        edges_y(i) = y_lo + i * (y_hi - y_lo) / nbinsy;
}

void PROsurf::FillSurfaceSimple(const PROconfig &config, const PROpeller &prop, const PROsyst &systs, const PROsc &osc, const PROspec &data, std::string filename, bool binned_weighting, int nThreads) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> d;
    
    std::ofstream chi_file;
    PROchi::EvalStrategy strat = binned_weighting ? PROchi::BinnedChi2 : PROchi::EventByEvent;

    if(!filename.empty()){
        chi_file.open(filename);
    }

    std::cout << "In FillSurfaceFast\n";

    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            std::cout << "Filling point " << i << " " << j << std::endl;
            LBFGSpp::LBFGSBParam<double> param;  
            param.epsilon = 1e-6;
            param.max_iterations = 100;
            param.max_linesearch = 50;
            param.delta = 1e-6;
            LBFGSpp::LBFGSBSolver<double> solver(param); 
            int nparams = systs.GetNSplines();
            std::vector<float> physics_params = {(float)edges_y(j), (float)edges_x(i)};//deltam^2, sin^22thetamumu
            PROchi chi("3plus1",&config,&prop,&systs,&osc, data, nparams, systs.GetNSplines(), strat, physics_params);
            Eigen::VectorXd lb = Eigen::VectorXd::Constant(nparams, -3.0);
            Eigen::VectorXd ub = Eigen::VectorXd::Constant(nparams, 3.0);
            Eigen::VectorXd x = Eigen::VectorXd::Constant(nparams, 0.0);

            double fx;
            int niter;
            std::vector<double> chi2s;
            int nfit = 0;
            do {
                nfit++;
                for(size_t i = 0; i < nparams; ++i)
                    x(i) = 0.3*d(rng);
                // x will be overwritten to be the best point found
                try {
                    niter = solver.minimize(chi, x, fx, lb, ub);
                } catch(std::runtime_error &except) {
                    log<LOG_ERROR>(L"%1% || Fit failed, %2%") % __func__ % except.what();
                    continue;
                }
                chi2s.push_back(fx);
            } while(chi2s.size() < 10 && nfit < 100);
            fx = *std::min_element(chi2s.begin(), chi2s.end());
            surface(i, j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
}

std::vector<surfOut> PROsurf::PointHelper(const PROconfig *config, const PROpeller *prop, const PROsyst *systs, const PROsc *osc, const PROspec *data, std::vector<surfOut> multi_physics_params, PROchi::EvalStrategy strat, bool binned_weighting, int start, int end){

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);

    std::vector<surfOut> outs;

    for(int i=start; i<end;i++){

        surfOut output;
        std::vector<float> physics_params = multi_physics_params[i].grid_val;
        output.grid_val = physics_params;
        output.grid_index = multi_physics_params[i].grid_index;

        LBFGSpp::LBFGSBParam<double> param;
        param.epsilon = 1e-6;
        param.max_iterations = 100;
        param.max_linesearch = 50;
        param.delta = 1e-6;

        LBFGSpp::LBFGSBSolver<double> solver(param);
        int nparams = systs->GetNSplines();
        PROchi chi("3plus1",config,prop,systs,osc,*data, nparams, systs->GetNSplines(), strat, physics_params);

        Eigen::VectorXd lb = Eigen::VectorXd::Constant(nparams, -3.0);
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(nparams, 3.0);
        Eigen::VectorXd x = Eigen::VectorXd::Constant(nparams, 0.0);
        Eigen::VectorXd grad = Eigen::VectorXd::Constant(nparams, 0.0);
        Eigen::VectorXd bestx = Eigen::VectorXd::Constant(nparams, 0.0);

        //First do 100 simple function calls suing LATIN hypercube setup
        double fx;
        int niter;
        int N_multistart = 100;
        std::vector<double> chi2s_multistart;
        std::vector<std::vector<double>> latin_samples = latin_hypercube_sampling(N_multistart, nparams,d_uni,rng);


        log<LOG_INFO>(L"%1% || Starting MultiGlobal runs : %2%") % __func__ % N_multistart ;
        for(int s=0; s<N_multistart; s++){


            x = Eigen::Map<Eigen::VectorXd>(latin_samples[s].data(), latin_samples[s].size());
            fx =  chi(x,grad,false);
            chi2s_multistart.push_back(fx);

        }
        //Sort so we can take the best N_localfits for further zoning
        std::vector<int> best_multistart = sorted_indices(chi2s_multistart);    

        log<LOG_INFO>(L"%1% || Ending MultiGlobal Best two are : %2% and %3%") % __func__ % chi2s_multistart[best_multistart[0]] %   chi2s_multistart[best_multistart[1]];
        log<LOG_INFO>(L"%1% || Best Points is  : %2% ") % __func__ % latin_samples[best_multistart[0]];

        int N_localfits = 5;
        std::vector<double> chi2s_localfits;
        int nfit = 0;
        double chimin = 9999999;

        log<LOG_INFO>(L"%1% || Starting Local Gradients runs : %2%") % __func__ % N_localfits ;
        for(int s=0; s<N_localfits; s++){
            //Get the nth
            x = Eigen::Map<Eigen::VectorXd>( latin_samples[best_multistart[s]].data(), latin_samples[best_multistart[s]].size());   
            try {
                niter = solver.minimize(chi, x, fx, lb, ub);
            } catch(std::runtime_error &except) {
                log<LOG_ERROR>(L"%1% || Fit failed, %2%") % __func__ % except.what();
            }
            chi2s_localfits.push_back(fx);
            if(fx<chimin){
                bestx = x;
                chimin=fx;
            }
            log<LOG_INFO>(L"%1% ||  LocalGrad Run : %2% has a chi %3%") % __func__ % s % fx;
            std::string spec_string = "";
            for(auto &f : x) spec_string+=" "+std::to_string(f); 
            log<LOG_INFO>(L"%1% || Best Point is  : %2% ") % __func__ % spec_string.c_str();


        }


        // and do CV
        log<LOG_INFO>(L"%1% || Starting CV fit ") % __func__  ;
        try {
            x = Eigen::VectorXd::Constant(nparams, 0.012);
            niter = solver.minimize(chi, x, fx, lb, ub);
        } catch(std::runtime_error &except) {
            log<LOG_ERROR>(L"%1% || Fit failed, %2%") % __func__ % except.what();
        }
        chi2s_localfits.push_back(fx);
        if(fx<chimin){
            bestx = x;
            chimin=fx;
        }

        log<LOG_INFO>(L"%1% ||  CV Run has a chi %2%") % __func__ %  fx;
        std::string spec_string = "";
        for(auto &f : x) spec_string+=" "+std::to_string(f); 
        log<LOG_INFO>(L"%1% || Best Point post CV is  : %2% ") % __func__ % spec_string.c_str();



        log<LOG_INFO>(L"%1% || FINAL has a chi %2%") % __func__ %  chimin;
        spec_string = "";
        for(auto &f : bestx) spec_string+=" "+std::to_string(f); 
        log<LOG_INFO>(L"%1% || FINAL is  : %2% ") % __func__ % spec_string.c_str();

        output.chi = chimin;
        outs.push_back(output);

    }

    return outs;

}


void PROsurf::FillSurface(const PROconfig &config, const PROpeller &prop, const PROsyst &systs, const PROsc &osc, const PROspec &data, std::string filename, bool binned_weighting, int nThreads) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> d;
    std::uniform_real_distribution<float> d_uni(-2.0, 2.0);
    PROchi::EvalStrategy strat = binned_weighting ? PROchi::BinnedChi2 : PROchi::EventByEvent;

    int nparams = systs.GetNSplines();
    Eigen::VectorXd lastx = Eigen::VectorXd::Constant(nparams, 0.01);

    std::ofstream chi_file;

    if(!filename.empty()){
        chi_file.open(filename);
    }

    data.plotSpectrum(config,"TTCV");

    std::vector<surfOut> grid;
    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            std::vector<int> grid_pts = {(int)i,(int)j};
            std::vector<float> physics_params = {(float)edges_y(j), (float)edges_x(i)};  //deltam^2, sin^22thetamumu
            surfOut pt; pt.grid_val = physics_params; pt.grid_index = grid_pts;
            grid.push_back(pt);
        }
    }

    int loopSize = grid.size();
    int chunkSize = loopSize / nThreads;

    //std::vector<surfOut> PointHelper(const PROconfig *config, const PROpeller *prop, const PROsyst *systs, const PROosc *osc, const PROspec *data, std::vector<surfOut>> multi_physics_params, PROchi::EvalStrategy strat, bool binned_weighting, int start, int end);

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                                        return this->PointHelper(&config, &prop, &systs, &osc, &data, grid, strat, binned_weighting, start, end);
                                        }));

    }

   std::vector<surfOut> combinedResults;
   for (auto& fut : futures) {
           std::vector<surfOut> result = fut.get();
           combinedResults.insert(combinedResults.end(), result.begin(), result.end());
   }

   for (const auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] %item.chi ;
        surface(item.grid_index[0], item.grid_index[1]) = item.chi;
        chi_file<<"\n"<<item.grid_val[1]<<" "<<item.grid_val[0]<<" "<<item.chi<<std::flush;
   }

}


