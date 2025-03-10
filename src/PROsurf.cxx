#include "PROsurf.h"
#include "PROfitter.h"
#include "PROlog.h"

#include <Eigen/Eigen>

using namespace PROfit;

PROsurf::PROsurf(PROmetric &metric, size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXf::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXf::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
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

void PROsurf::FillSurfaceStat(const PROconfig &config, std::string filename) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
    }

    PROsyst dummy_syst;
    dummy_syst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_bins_total, config.m_num_bins_total, 0);
    Eigen::VectorXf empty_vec;

    PROmetric *local_metric = metric.Clone();
    local_metric->override_systs(dummy_syst);

    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            Eigen::VectorXf physics_params{{(float)edges_y(j), (float)edges_x(i)}};
            float fx = (*local_metric)(physics_params, empty_vec, false);
            surface(i, j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    delete local_metric;
}

std::vector<profOut> PROfile::PROfilePointHelper(const PROsyst *systs, int start, int end, bool with_osc, const Eigen::VectorXf& init_seed) {

    std::vector<profOut> outs;
    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();
    int nparams = local_metric->GetModel().nparams + systs->GetNSplines();
    int nstep = 20;

    for(int i=start; i<end;i++){
        local_metric->reset();

        int which_spline= i;
        bool isphys = which_spline < local_metric->GetModel().nparams;
        //if(isphys) nstep=4*nstep;
        profOut output;

        log<LOG_INFO>(L"%1% || THREADS %2% in this batch if ( %3%,%4% )") % __func__ %  i % start % end;
        

        Eigen::VectorXf last_bf;
        if(init_seed.norm()>0) last_bf= init_seed;
        for(int i=0; i<=nstep;i++){
            Eigen::VectorXf ub, lb;

            if(with_osc) {
                lb = Eigen::VectorXf::Constant(nparams, -3.0);
                ub = Eigen::VectorXf::Constant(nparams, 3.0);
                //set physics to correct values
                for(int j=0; j<local_metric->GetModel().nparams; j++){
                    ub(j) = local_metric->GetModel().ub(j);
                    lb(j) = local_metric->GetModel().lb(j); 
                }
                //upper lower bounds for splines
                for(int j = local_metric->GetModel().nparams; j < nparams; ++j) {
                    lb(j) = systs->spline_lo[j-2];
                    ub(j) = systs->spline_hi[j-2];
                }
            } else {
                ub = Eigen::VectorXf::Map(systs->spline_hi.data(), systs->spline_hi.size());
                lb = Eigen::VectorXf::Map(systs->spline_lo.data(), systs->spline_lo.size());
                nparams = systs->GetNSplines();
            }



            float which_value =  std::isinf(lb(which_spline)) ? -3 + (ub(which_spline) - (-3)) * i / (float)nstep :   lb(which_spline) + (ub(which_spline) - lb(which_spline)) * i / (float)nstep;
            float fx;
            output.knob_vals.push_back(which_value);

            //log<LOG_INFO>(L"%1% || Fixed value of spline # %2% is value %3%, has a chi PRE of : %4% (i %5% nstep %6% || ub %7% lb %8% ") % __func__ % which_spline % which_value % fx % i % nstep % lb.size() % ub.size() ;
            //std::vector<float> ubv(ub.data(), ub.data() + ub.size());
            //std::vector<float> lbv(lb.data(), lb.data() + lb.size());
            //log<LOG_INFO>(L"%1% || lb %2%  ub %3% ") % __func__ % lbv % ubv  ;

            lb[which_spline] = which_value;
            ub[which_spline] = which_value;

            LBFGSpp::LBFGSBParam<float> param;
            param.epsilon = 1e-6;
            param.max_iterations = 50;
            param.max_linesearch = 100;
            param.delta = 1e-6;
	    param.past = 1.e-4;
            local_metric->fixSpline(which_spline,which_value);

            PROfitter fitter(ub, lb, param);
            if(last_bf.norm()>0){
                fx = fitter.Fit(*local_metric,last_bf);
            }else{
                fx = fitter.Fit(*local_metric);
            }
            output.knob_chis.push_back(fx);
            last_bf = fitter.best_fit;
          

            std::string spec_string = "";
            for(auto &f : fitter.best_fit) spec_string+=" "+std::to_string(f); 
            log<LOG_INFO>(L"%1% || Fixed value of spline # %2% is value %3%, has a chi post of : %4% (i %5% nstep %6% ") % __func__ % which_spline % which_value % fx % i % nstep;
            log<LOG_INFO>(L"%1% || at a BF param value of @ %2%") % __func__ %  spec_string.c_str();

        }    //end spline loop        

       outs.push_back(output);

    }//end thread

    delete local_metric;

    return outs;
}

std::vector<surfOut> PROsurf::PointHelper(std::vector<surfOut> multi_physics_params, int start, int end){
    std::vector<surfOut> outs;

    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();

    for(int i=start; i<end;i++){
        local_metric->reset();

        surfOut output;
        std::vector<float> physics_params = multi_physics_params[i].grid_val;
        output.grid_val = physics_params;
        output.grid_index = multi_physics_params[i].grid_index;

        int nparams = local_metric->GetModel().nparams + local_metric->GetSysts().GetNSplines() - 2;

        if(nparams == 0) {
            Eigen::VectorXf empty_vec, 
                            params = Eigen::VectorXf::Map(physics_params.data(), physics_params.size());
            output.chi = (*local_metric)(params, empty_vec, false);
            outs.push_back(output);
            continue;
        }

        LBFGSpp::LBFGSBParam<float> param;
        param.epsilon = 1e-6;
        param.max_iterations = 50;
        param.max_linesearch = 100;
        param.delta = 1e-6;
	param.past = 1e-4;
        Eigen::VectorXf lb(nparams+2);
        lb << local_metric->GetModel().lb, Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), local_metric->GetSysts().spline_lo.size());
        Eigen::VectorXf ub(nparams+2);
        ub << local_metric->GetModel().ub, Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), local_metric->GetSysts().spline_hi.size());

        lb(x_idx) = multi_physics_params[i].grid_val[1];
        ub(x_idx) = multi_physics_params[i].grid_val[1];
        lb(y_idx) = multi_physics_params[i].grid_val[0];
        ub(y_idx) = multi_physics_params[i].grid_val[0];

        PROfitter fitter(ub, lb, param);
        output.chi = fitter.Fit(*local_metric);
        outs.push_back(output);
    }
    
    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(std::string filename, int nThreads) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
    }

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

    std::vector<std::future<std::vector<surfOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
            return this->PointHelper(grid, start, end);
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

// findMinAndBounds(g.get(),1.0, range) // acting on a graph containing [(p-bf)_i/sigma_i, chisquare]     
// returns minX = x(minY = min chi square --> should be 0) and min,max of the x range corresponding to delta chi square = val wrt to the best fit 1 (so to +/- 1 sigma wrt best fit in the example)   
std::vector<float> findMinAndBounds(TGraph *g, float val,float range) {
    float step = 0.001;
    range = range+step;
    int n = g->GetN();
    float minY = 1e9, minX = 0;
    for (int i = 0; i < n; ++i) {
        double x, y;
        g->GetPoint(i, x,y);
        if (y < minY) {
            minY = y;
            minX = x;
        }
    }
    //..ok so minX is the min and Current minY is the chi^2. Want this to be delta chi^2 -- if everything is okay from the fit we should have minY = 0 
    if( minY > 1.e-3)
      log<LOG_INFO>(L"%1% || Problem in the fit! MinY is not zero, but = %2%") % __func__ % minY;     
    
    float leftX = minX, rightX = minX;
    
    // Search to the left of the minimum
    for (float x = minX; x >= -range; x -= step) {
        float y = g->Eval(x) - minY; //DeltaChi^2
        if (y >= val) {
            leftX = x;
            break;
        }
    }
    

    // Search to the right of the minimum
    for (float x = minX; x <= range; x += step) {
        float y = g->Eval(x)-minY;
        if (y >= val) {
            rightX = x;
            break;
        }
    }
    
    return {minX,leftX,rightX};
}


PROfile::PROfile(const PROconfig &config, const PROpeller &prop, const PROsyst &systs, const PROmodel &model, const PROspec &data, PROmetric &metric, std::string filename, bool with_osc, int nThreads, const Eigen::VectorXf & init_seed) : metric(metric) {

    LBFGSpp::LBFGSBParam<float> param;
    param.epsilon = 1e-6;
    param.max_iterations = 50;
    param.max_linesearch = 100;
    param.delta = 1e-6;
    param.past = 1.e-4;

    LBFGSpp::LBFGSBSolver<float> solver(param);
    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    std::vector<float> physics_params; 
    std::vector<float> physics_params_errors; // err_low_Dmsq, err_up_Dmsq, err_low_sinsq, err_up_sinsq
    
    std::vector<std::unique_ptr<TGraph>> graphs; // a graph to store for each of the fit parameters X = (par values - best fit values)/sigma, Y = chi square values ( par values) 

    //hack
    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data()); // same as graph but at a set of predefined points to illustrate the prior
    // it makes sense only for the spline systs (the other have a flat prior) and btw center can be different from zero - see sinsq2th

    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    int loopSize = nparams;
    if(nThreads>loopSize){
        log<LOG_INFO>(L"%1% || nThreads is > loopSize (nparams) : %2% > %3%. Setting equal ") % __func__ % nThreads % loopSize ;
	nThreads = loopSize;
    }

    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<profOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
            return this->PROfilePointHelper(&systs, start, end,with_osc,init_seed);
        }));

    }

    std::vector<profOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<profOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }


    int depth = std::ceil(nparams/4.0);
    TCanvas *c =  new TCanvas(filename.c_str(), filename.c_str() , 400*4, 400*depth);
    c->Divide(4,depth);

    int w = 0;
    for(auto & out: combinedResults){
        
        log<LOG_INFO>(L"%1% || Knob Values: %2%") % __func__ %  out.knob_vals;
        log<LOG_INFO>(L"%1% || Knob Chis: %2%") % __func__ %  out.knob_chis;

        c->cd(w+1);
        std::unique_ptr<TGraph> g = std::make_unique<TGraph>(out.knob_vals.size(), out.knob_vals.data(), out.knob_chis.data());
        std::string xval = w < model.nparams ? "Log_{10}(" + model.pretty_param_names[w]+")" :"#sigma Shift"  ;
        std::string tit = names[w]+ ";"+xval+"; #Chi^{2}";
        g->SetTitle(tit.c_str());
        graphs.push_back(std::move(g));
        graphs.back()->Draw("AL");
        graphs.back()->SetLineWidth(2);

        if(w>=model.nparams){
            gprior->Draw("L same");
            gprior->SetLineStyle(2);
            gprior->SetLineWidth(1);
        }
        w++;
    }

    c->SaveAs((filename+".pdf").c_str(),"pdf");

    delete c;

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 40*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.5);
    //plot 2sigma also? default no, as its messier
    bool twosig = false;
    int nBins = nparams;

    std::vector<float> bfvalues;
    std::vector<float> values1_up;
    std::vector<float> values1_down;
    std::vector<float> values1_errup;
    std::vector<float> values1_errdown;

    std::vector<float> values2_up;
    std::vector<float> values2_down;

    std::vector<float> barvalues;
    std::vector<float> barvalues_err;

    log<LOG_INFO>(L"%1% || Getting BF, +/- one sigma ranges. Is Two sigma turned on? : %2% ") % __func__ % twosig;

    int count = 0;
    for(auto &g:graphs){
        //if(metric->GetModel().nparams)continue;
        float range = count == 0 ? 2.0 : count == 1 ? 1.0 : 3.0; // log10(dmsq) = [-2,2], log10(sinsq(2th)) should be (-inf,0) which can become [-1,1] ???, spline systs [-3,3]
        std::vector<float> tmp = findMinAndBounds(g.get(),1.0, range); // find x corresponding to min of the chi square == best fit that should coincide with the value that gives you chi square = 0 
	// val == 1. means finding min and max on the axis of the parameter that correspond to delta chi square == 1 which should coincide with switching of +/- 1 sigma from the best fit    
	barvalues.push_back(float(count)+0.5); // to center the bar
	barvalues_err.push_back(0.4);// to make sure there's space for the star
        bfvalues.push_back(tmp[0]);

	values1_up.push_back(tmp[2]);
	values1_down.push_back(tmp[1]);
	values1_errdown.push_back(abs(tmp[1]-tmp[0])); // mu-1sigma - mu == -1 sigma 
        values1_errup.push_back(abs(tmp[2]-tmp[0]));  // mu+1sigma - mu == + 1 sigma 
	log<LOG_DEBUG>(L"%1% || Results of findMinAndBounds : %2% %3% %4% ") % __func__ % tmp[0] % tmp[1] % tmp[2];
	log<LOG_DEBUG>(L"%1% || Barvalues : %2% %3% %4% %5%") % __func__ % count % barvalues[count] % barvalues_err[count] % barvalues_err[count];
	log<LOG_DEBUG>(L"%1% || Bfvalues : %2% %3% ") % __func__ % count % bfvalues[count];
	log<LOG_DEBUG>(L"%1% || RangeValues : %2% %3% %4% ") % __func__ % count % values1_down[count] % values1_up[count];
	log<LOG_DEBUG>(L"%1% || ErrValues : %2% %3% %4% ") % __func__ % count % values1_errdown[count] % values1_errup[count];

	if( with_osc && count < model.nparams*with_osc ){
	  physics_params_errors.push_back( tmp[1]-tmp[0] );
	  physics_params_errors.push_back( tmp[2]-tmp[0] );
	  log<LOG_DEBUG>(L"%1% || Saving %2% and %3%") % __func__ % values1_errdown[count] % values1_errup[count];	  
	}
	  
        if(twosig){
	  // val == 4 corresponds to delta chi square = 4 and thus +/- 2 sigma from the best fit  
	  std::vector<float> tmp2 = findMinAndBounds(g.get(),4.0,range);
	  values2_down.push_back(abs(tmp2[1]-tmp[0])); // left variation
	  values2_up.push_back(abs(tmp2[2]-tmp[0])); // x(mu+2*sigma) - x(mu) // right variation 
        }
        count++;
    }

    
    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    float minVal = *std::min_element(values1_down.begin(), values1_down.end());
    float maxVal = *std::max_element(values1_up.begin(), values1_up.end());
    // npoints, vx, vy, errx_left, errx_right == errx_left, erry_low, erry_high
    TGraphAsymmErrors *h1 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values1_errdown.data(), values1_errup.data());
    h1->SetFillColor(kBlue-7);
    h1->SetStats(0);
    h1->SetMinimum(min(-1.2,minVal*1.2));
    h1->SetMaximum(max(1.2,maxVal*1.2));
    h1->GetXaxis()->SetNdivisions(barvalues.size());  // Set number of tick marks                                                                                                                          
    h1->GetXaxis()->SetLabelSize(0);  // Hide default numerical labels                                                                                                                                     
    h1->SetTitle("");
    h1->Draw("A2");
    h1->GetYaxis()->SetTitle("#sigma Shift");

    float y_min = h1->GetYaxis()->GetXmin();  // Get Y-axis min value for label positioning                                                                                                                
    for (size_t i = 0; i < barvalues.size(); ++i) {
      TText* text = new TText(barvalues[i], y_min - 0.5, names[i].c_str());  // Position text below axis                                                                                                   
      text->SetTextAlign(22);  // Center align the text                                                                                                                                                    
      text->SetTextSize(0.03); // Adjust text size                                                                                                                                                         
      text->SetTextAngle(45); //Angle the text                                                                                                                                                             
      text->Draw();
    }

    c2->Update();

    if (twosig) {
      TGraphAsymmErrors *h2 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values2_down.data(), values2_up.data());
      h2->SetFillColor(38);
      h2->SetStats(0);
      h2->SetTitle("");
      h2->Draw("A2");
      h2->GetYaxis()->SetTitle("");
    }


    TLine l(0,0,nBins+0.5,0);
    l.SetLineStyle(2);
    l.SetLineColor(kBlack);
    l.SetLineWidth(1);
    l.Draw();

    for (int i = 0; i < nBins; ++i) {
      TMarker* star = new TMarker(i+0.5, bfvalues[i], 29);
      star->SetMarkerSize(0.5);
      star->SetMarkerColor(kBlack);
      star->Draw();
    }

    if(!twosig)
      c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    else
      c2->SaveAs((filename+"_2sigma.pdf").c_str(),"pdf");
    delete c2;
    asym_error_phys.clear();
    if( physics_params_errors.size() == 4)
      asym_error_phys = physics_params_errors;
    return;
}
