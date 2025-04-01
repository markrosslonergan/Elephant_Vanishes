#include "PROsurf.h"
#include "LBFGSpp/Param.h"
#include "PROfitter.h"
#include "PROlog.h"

#include <Eigen/Eigen>
#include <cmath> 
#include "TLatex.h"

using namespace PROfit;

PROsurf::PROsurf(PROmetric &metric,  size_t x_idx, size_t y_idx, size_t nbinsx, LogLin llx, float x_lo, float x_hi, size_t nbinsy, LogLin lly, float y_lo, float y_hi) : metric(metric), x_idx(x_idx), y_idx(y_idx), nbinsx(nbinsx), nbinsy(nbinsy), edges_x(Eigen::VectorXf::Constant(nbinsx + 1, 0)), edges_y(Eigen::VectorXf::Constant(nbinsy + 1, 0)), surface(nbinsx, nbinsy) {
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

void PROsurf::FillSurfaceStat(const PROconfig &config, const LBFGSpp::LBFGSBParam<float> &param, std::string filename) {
    std::ofstream chi_file;
    if(!filename.empty()){
        chi_file.open(filename);
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";

        chi_file << "xval yval chi2";
        // TODO: Not saving this info for stats only right now (but we should)
        //for(size_t i = 0; i < metric.GetModel().nparams; ++i)
        //    chi_file << " p" << i;
        chi_file << "\n";
    }

    // I think this will be needed for stat fits with more than 2 physics parameters
    (void)param;

    PROsyst dummy_syst;
    dummy_syst.fractional_covariance = Eigen::MatrixXf::Constant(config.m_num_bins_total, config.m_num_bins_total, 0);
    Eigen::VectorXf empty_vec;

    PROmetric *local_metric = metric.Clone();
    local_metric->override_systs(dummy_syst);
    float min_chi = 1e9;

    for(size_t i = 0; i < nbinsx; i++) {
        for(size_t j = 0; j < nbinsy; j++) {
            Eigen::VectorXf physics_params{{(float)edges_y(j), (float)edges_x(i)}};
            float fx = (*local_metric)(physics_params, empty_vec, false);
            if(fx < min_chi) min_chi = fx;
            surface(i, j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    for(size_t i = 0; i < nbinsx; ++i) {
        for(size_t j = 0; j < nbinsy; ++j) {
            float fx = surface(i,j); 
            fx -= min_chi;
            surface(i,j) = fx;
            if(!filename.empty()){
                chi_file<<"\n"<<edges_x(i)<<" "<<edges_y(j)<<" "<<fx<<std::flush;
            }
        }
    }
    delete local_metric;
}

std::vector<profOut> PROfile::PROfilePointHelper(const PROsyst *systs, const LBFGSpp::LBFGSBParam<float> &param, int start, int end, bool with_osc, const Eigen::VectorXf& init_seed, uint32_t seed) {

    std::vector<profOut> outs;
    // Make a local copy for this thread
    PROmetric *local_metric = metric.Clone();
    int nparams = local_metric->GetModel().nparams + systs->GetNSplines();
    int nstep = 20;

    for(int i=start; i<end;i++){
        local_metric->reset();

        size_t which_spline= i;
        bool isphys = which_spline < local_metric->GetModel().nparams;
        if(isphys) nstep=4*nstep;
        profOut output;

        log<LOG_INFO>(L"%1% || THREADS %2% in this batch if ( %3%,%4% )") % __func__ %  i % start % end;


        Eigen::VectorXf last_bf;
        if(init_seed.norm()>0) last_bf= init_seed;
        for(int i=0; i<=nstep;i++){
            Eigen::VectorXf ub, lb;

            if(with_osc) {
                lb = Eigen::VectorXf::Constant(nparams, -3.0);
                ub = Eigen::VectorXf::Constant(nparams, 3.0);
                size_t nphys = local_metric->GetModel().nparams;
                //set physics to correct values
                for(size_t j=0; j<nphys; j++){
                    ub(j) = local_metric->GetModel().ub(j);
                    lb(j) = local_metric->GetModel().lb(j); 
                }
                //upper lower bounds for splines
                for(int j = nphys; j < nparams; ++j) {
                    lb(j) = systs->spline_lo[j-nphys];
                    ub(j) = systs->spline_hi[j-nphys];
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

            local_metric->fixSpline(which_spline,which_value);

            PROfitter fitter(ub, lb, param, seed+i);
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

std::vector<surfOut> PROsurf::PointHelper(const LBFGSpp::LBFGSBParam<float> &param, std::vector<surfOut> multi_physics_params, int start, int end, uint32_t seed){

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
            output.best_fit = params; 
            outs.push_back(output);
            continue;
        }

        Eigen::VectorXf lb(nparams+2);
        lb << local_metric->GetModel().lb, Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), local_metric->GetSysts().spline_lo.size());
        Eigen::VectorXf ub(nparams+2);
        ub << local_metric->GetModel().ub, Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), local_metric->GetSysts().spline_hi.size());

        lb(x_idx) = multi_physics_params[i].grid_val[1];
        ub(x_idx) = multi_physics_params[i].grid_val[1];
        lb(y_idx) = multi_physics_params[i].grid_val[0];
        ub(y_idx) = multi_physics_params[i].grid_val[0];

        PROfitter fitter(ub, lb, param, seed+i);
        output.chi = fitter.Fit(*local_metric);
        output.best_fit = fitter.best_fit;
        outs.push_back(output);
    }

    delete local_metric;

    return outs;
}


void PROsurf::FillSurface(const LBFGSpp::LBFGSBParam<float> &param, std::string filename, PROseed &proseed, int nThreads) {
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
                    return this->PointHelper(param, grid, start, end, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<surfOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<surfOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }

    if(filename != "") {
        chi_file << "Dimensions: " << nbinsx << " " << nbinsy << "\n";
        chi_file << "Fixed indices: " << x_idx << " " << y_idx << "\n";
        chi_file << "Parameters:\n";
        for(const auto &name: metric.GetModel().param_names) chi_file << name << "\n";
        for(const auto &name: metric.GetSysts().spline_names) chi_file << name << "\n";

        chi_file << "\nxval yval chi2";
        for(size_t i = 0; i < metric.GetModel().nparams + metric.GetSysts().GetNSplines(); ++i)
            chi_file << " p" << i;
    }
    float min_chi = 1e9;
    for(const auto &item: combinedResults) {
        if(item.chi < min_chi) min_chi = item.chi;
    }
    for (const auto& item : combinedResults) {
        log<LOG_INFO>(L"%1% || Finished  : %2% %3% %4%") % __func__ % item.grid_val[1] % item.grid_val[0] % (item.chi - min_chi);
        surface(item.grid_index[0], item.grid_index[1]) = item.chi - min_chi;
        results.push_back({item.grid_index[0], item.grid_index[1], item.best_fit, (item.chi-min_chi)});
        if(filename != "") {
            chi_file<<"\n"<<item.grid_val[1]<<" "<<item.grid_val[0]<<" "<<(item.chi-min_chi);
            for(float val: item.best_fit)
                chi_file << " " << val;
        }
    }
}

std::vector<float> findMinAndBounds(TGraph *g, float val, float lo, float hi) {
    float step = 0.001;
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
    //..ok so minX is the min and Currentl minY is the chi^2. Want this to be delta chi^2

    float leftX = minX, rightX = minX;

    // Search to the left of the minimum
    for (float x = minX; x >= lo; x -= step) {
        float y = g->Eval(x) - minY; //DeltaChi^2
        if (y >= val) {
            leftX = x;
            break;
        } else if(x - step < lo) {
            // If at end of loop and haven't found left side
            leftX = lo;
        }
    }

    // Search to the right of the minimum
    for (float x = minX; x <= hi; x += step) {
        float y = g->Eval(x)-minY;
        if (y >= val) {
            rightX = x;
            break;
        } else if(x + step > hi) {
            // If at end of loop and haven't found right side
            rightX = hi;
        }
    }

    return {minX,leftX,rightX};
}


PROfile::PROfile(const PROconfig &config, const PROsyst &systs, const PROmodel &model, PROmetric &metric, PROseed &proseed, const LBFGSpp::LBFGSBParam<float> &param, std::string filename, bool with_osc, int nThreads, const Eigen::VectorXf & init_seed, const Eigen::VectorXf & true_params) : metric(metric) {
    LBFGSpp::LBFGSBSolver<float> solver(param);
    int nparams = systs.GetNSplines() + model.nparams*with_osc;
    std::vector<float> physics_params; 

    std::vector<std::unique_ptr<TGraph>> graphs; 

    //hack
    std::vector<float> priorX;
    std::vector<float> priorY;

    for(int i=0; i<=30;i++){
        float which_value = -3.0+0.2*i;
        priorX.push_back(which_value);
        priorY.push_back(which_value*which_value);

    }
    std::unique_ptr<TGraph> gprior = std::make_unique<TGraph>(priorX.size(), priorX.data(), priorY.data());

    std::vector<std::string> names;
    if(with_osc) for(const auto& name: model.pretty_param_names) names.push_back(name);
    for(const auto &name: systs.spline_names) names.push_back(name);

    int loopSize = nparams;
    if(nThreads>loopSize){
        nThreads = loopSize;
        log<LOG_INFO>(L"%1% || nThreads is < loopSize (nparams) : %2% <  %3%. Setting equal ") % __func__ % nThreads % loopSize ;
    }

    int chunkSize = loopSize / nThreads;

    std::vector<std::future<std::vector<profOut>>> futures; 

    log<LOG_INFO>(L"%1% || Starting THREADS  : %2% , Loops %3%, Chunks %4%") % __func__ % nThreads % loopSize % chunkSize;

    for (int t = 0; t < nThreads; ++t) {
        int start = t * chunkSize;
        int end = (t == nThreads - 1) ? loopSize : start + chunkSize;
        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
                    return this->PROfilePointHelper(&systs, param, start, end,with_osc,init_seed, proseed.getThreadSeeds()->at(t));
                    }));

    }

    std::vector<profOut> combinedResults;
    for (auto& fut : futures) {
        std::vector<profOut> result = fut.get();
        combinedResults.insert(combinedResults.end(), result.begin(), result.end());
    }


    //create all graphs, used directly in first setion
    for(auto & out: combinedResults){
        log<LOG_INFO>(L"%1% || Knob Values: %2%") % __func__ %  out.knob_vals;
        log<LOG_INFO>(L"%1% || Knob Chis: %2%") % __func__ %  out.knob_chis;
        std::unique_ptr<TGraph> g = std::make_unique<TGraph>(out.knob_vals.size(), out.knob_vals.data(), out.knob_chis.data());
        graphs.push_back(std::move(g));
    }

    //Analyze them, used in later section
    //plot 2sigma also? default no, as its messier
    bool twosig = false;
    int nBins = nparams;

    std::vector<float> bfvalues;
    std::vector<float> values1_up;
    std::vector<float> values1_down;
    std::vector<float> values1_errup;
    std::vector<float> values1_errdown;

    std::vector<float> barvalues;
    std::vector<float> barvalues_err;

    std::vector<float> values2_up;
    std::vector<float> values2_down;


    log<LOG_INFO>(L"%1% || Getting BF, +/- one sigma ranges. Is Two sigma turned on? : %2% ") % __func__ % twosig;

    size_t count = 0;
    for(auto &g:graphs){
        //if(metric->GetModel().nparams)continue;
        float lo = count < metric.GetModel().nparams ? metric.GetModel().lb(count) :
            metric.GetSysts().spline_lo[count - metric.GetModel().nparams];
        if(std::isinf(lo)) lo = lo < 0 ? -5 : 5;
        float hi = count < metric.GetModel().nparams ? metric.GetModel().ub(count) :
            metric.GetSysts().spline_hi[count - metric.GetModel().nparams];
        if(std::isinf(hi)) hi = hi < 0 ? -5 : 5;
        std::vector<float> tmp = findMinAndBounds(g.get(),1.0, lo, hi);
        barvalues.push_back(float(count)+0.5);
        barvalues_err.push_back(0.4);
        bfvalues.push_back(tmp[0]);
        values1_down.push_back(tmp[1]);
        values1_up.push_back(tmp[2]);
        values1_errdown.push_back(abs(tmp[1]-tmp[0]));
        values1_errup.push_back(abs(tmp[2]-tmp[0]));
        log<LOG_DEBUG>(L"%1% || Results of findMinAndBounds : %2% %3% %4% ") % __func__ % tmp[0] % tmp[1] % tmp[2];
        log<LOG_DEBUG>(L"%1% || Barvalues : %2% %3% %4% %5%") % __func__ % count % barvalues[count] % barvalues_err[count] % barvalues_err[count];
        log<LOG_DEBUG>(L"%1% || Bfvalues : %2% %3% ") % __func__ % count % bfvalues[count];
        log<LOG_DEBUG>(L"%1% || RangeValues : %2% %3% %4% ") % __func__ % count % values1_down[count] % values1_up[count];
        log<LOG_DEBUG>(L"%1% || ErrValues : %2% %3% %4% ") % __func__ % count % values1_errdown[count] % values1_errup[count];
        if(twosig){
            std::vector<float> tmp2 = findMinAndBounds(g.get(),4.0,lo, hi);
            values2_down.push_back(abs(tmp2[1]-tmp[0]));
            values2_up.push_back(abs(tmp2[2]-tmp[0]));
        }
        count++;
    }




    //First plot
    int depth = std::ceil((nparams+model.nparams)/4.0);
    TCanvas *c =  new TCanvas(filename.c_str(), filename.c_str() , 400*4, 400*depth);
    c->Divide(4,depth);


    size_t zoom_shift = 0;
    for(size_t w = 0; w< combinedResults.size(); w++ ){

        c->cd(w+1+zoom_shift);
        std::string xval = w < model.nparams ? "Log_{10}(" + model.pretty_param_names[w]+")" :"#sigma Shift"  ;
        std::string tit = names[w]+ ";"+xval+"; #Chi^{2}";
        graphs[w]->SetTitle(tit.c_str());
        graphs[w]->Draw("AL");
        graphs[w]->SetLineWidth(2);
        
        TLine* line = new TLine(graphs[w]->GetXaxis()->GetXmin(), 1, graphs[w]->GetXaxis()->GetXmax(), 1);
        line->SetLineStyle(3);  // Dotted line style (1 is solid, 2 is dashed, 3 is dotted)
        line->SetLineWidth(1);  // Thin line
        line->SetLineColor(kBlack);  // Set color (black for visibility)
        line->Draw();

        if(w<model.nparams) graphs[w]->SetLineColor(kBlue-7);

        if(w>=model.nparams){
            gprior->Draw("L same");
            gprior->SetLineStyle(2);
            gprior->SetLineWidth(2);
            gprior->SetLineColor(kRed-7);
        }

        if(w==model.nparams-1){
            //on past physics param, lets do a quick zoom, stepping back though the physics param
            for(size_t zs = 0; zs<model.nparams; zs++){
                c->cd(w+1+zs+1);
                std::unique_ptr<TGraph> graphClone = std::make_unique<TGraph>(*graphs[w-zs]);
                graphClone->Draw("AL");
                std::string newTitle = std::string(graphClone->GetTitle()) + " Zoomed 1#sigma";
                graphClone->SetTitle(newTitle.c_str());
                graphClone->SetLineColor(kViolet);
                float vd = std::min(values1_down[w]*0.8,values1_up[w]*1.2) ;
                float vu = std::max(values1_down[w]*0.8,values1_up[w]*1.2) ;
                graphClone->GetXaxis()->SetLimits(vd,vu); 
                graphClone->GetYaxis()->SetRangeUser(graphClone->Eval(vd), graphClone->Eval(vu) );

                TLine *line1 = new TLine(vd, 1, vu, 1);
                line1->SetLineStyle(3);  
                line1->SetLineWidth(1);  
                line1->SetLineColor(kBlack); 
                line1->Draw();

                TLine* line2 = new TLine(vd, graphs[w]->Eval(vd) ,vd, 0);
                line2->SetLineStyle(3);  
                line2->SetLineWidth(1);  
                line2->SetLineColor(kBlack); 
                line2->Draw();

                TLine *line3 = new TLine(vu, graphs[w]->Eval(vu) ,vu, 0);
                line3->SetLineStyle(3);  
                line3->SetLineWidth(1);  
                line3->SetLineColor(kBlack); 
                line3->Draw();

            }
            zoom_shift=model.nparams;
        }
    }

    c->SaveAs((filename+".pdf").c_str(),"pdf");

    delete c;

    //Next version
    TCanvas *c2 =  new TCanvas((filename+"1sigma").c_str(), (filename+"1sigma").c_str() , 40*nparams, 400);
    c2->cd();
    c2->SetBottomMargin(0.25);
    c2->SetRightMargin(0.05);

    log<LOG_DEBUG>(L"%1% || Are all lines the same : %2% %3% %4% %5% %6%") % __func__ % nBins % barvalues.size() % bfvalues.size() % values1_down.size() % values1_up.size() ;

    float minVal = *std::min_element(values1_down.begin(), values1_down.end());
    float maxVal = *std::max_element(values1_up.begin(), values1_up.end());

    TGraphAsymmErrors *h1 = new TGraphAsymmErrors(barvalues.size(),barvalues.data(), bfvalues.data(), barvalues_err.data(), barvalues_err.data(), values1_errdown.data(), values1_errup.data());
    h1->SetFillColor(kBlue-7);
    h1->SetStats(0);
    //h1->SetMinimum(min(-1.2,minVal*1.2));
    h1->SetMinimum(minVal*1.2);
    h1->SetMaximum(maxVal*1.2);

    h1->GetXaxis()->SetNdivisions(barvalues.size());  // Set number of tick marks
    h1->GetXaxis()->SetLabelSize(0);  // Hide default numerical labels

    h1->SetTitle("");
    h1->Draw("A2");
    //h1->GetYaxis()->SetTitle("#sigma Shift");
    onesig = *h1;

    float y_min = h1->GetMinimum();
    for (size_t i = 0; i < barvalues.size(); ++i) {
        std::string label = i < model.nparams ? "Log_{10}(" + model.pretty_param_names[i]+")" : config.m_mcgen_variation_plotname_map.at(names[i]);
        TLatex* text = new TLatex(barvalues[i], y_min - 0.05, label.c_str());  // Position text below axis
        text->SetTextAlign(13);  
        text->SetTextSize(0.03); 
        text->SetTextAngle(-45); 
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
        TMarker* initstar = new TMarker(i+0.5, init_seed[i], 29);
        initstar->SetMarkerSize(0.6); 
        initstar->SetMarkerColor(kBlue); 
        initstar->Draw();

        if (i < true_params.size()) {

            TMarker* truestar = new TMarker(i+0.5, true_params[i], 29);
            truestar->SetMarkerSize(0.5); 
            truestar->SetMarkerColor(kRed); 
            truestar->Draw();
        }

        TMarker* star = new TMarker(i+0.5, bfvalues[i], 29);
        star->SetMarkerSize(0.5); 
        star->SetMarkerColor(kBlack); 
        star->Draw();
    }



    c2->SaveAs((filename+"_1sigma.pdf").c_str(),"pdf");
    delete c2;

    return;
}

