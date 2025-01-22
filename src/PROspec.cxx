#include "PROspec.h"

#include <random>

using namespace PROfit;

PROspec::PROspec(size_t num_bins):
    nbins(num_bins),
    spec(Eigen::VectorXd::Zero(num_bins)),
    error(Eigen::VectorXd::Zero(num_bins)){
    }

PROspec PROspec::PoissonVariation(const PROspec &s) {
    static std::random_device rd;
    std::mt19937 gen(rd());

    PROspec newSpec(s.nbins);

    for(size_t i = 0; i < s.nbins; i++) {
        std::poisson_distribution<> d(s.GetBinContent(i));
        newSpec.Fill(i, d(gen));
    }

    return newSpec;
}

size_t PROspec::GetNbins() const{
    return nbins;
}

void PROspec::Zero(){
    log<LOG_INFO>(L"%1% || Zero out spectrum") % __func__ ;
    spec.setZero();
    error.setZero();
    return;
}

void PROspec::Print() const {
    std::string spec_string = "";
    for(auto &f : spec) spec_string+=" "+std::to_string(f); 
    log<LOG_INFO>(L"%1% || %2%" ) % __func__ % spec_string.c_str();
    return;
}


void PROspec::Fill(int bin_index, double weight){
    //Removed to help speed up filling
    spec(bin_index) += weight;
    float tmp_err = error(bin_index);
    error(bin_index) = std::sqrt(tmp_err*tmp_err + weight*weight);

    return;
}

void PROspec::QuickFill(int bin_index, double weight){
    spec(bin_index) += weight;
    return;
}


TH1D PROspec::toTH1D(PROconfig const & inconfig, int subchannel_index) const{

    int global_bin_start = inconfig.GetGlobalBinStart(subchannel_index);
    int channel_index = inconfig.GetChannelIndex(subchannel_index);

    //set up hist specs
    int nbins = inconfig.m_channel_num_bins[channel_index];
    const std::vector<double>& bin_edges = inconfig.GetChannelBinEdges(channel_index);
    std::string hist_name = inconfig.m_fullnames[subchannel_index];
    std::string xaxis_title = inconfig.m_channel_units[channel_index];


    //fill 1D hist
    TH1D hSpec(hist_name.c_str(),hist_name.c_str(), nbins, &bin_edges[0]); 
    hSpec.GetXaxis()->SetTitle(xaxis_title.c_str());
    for(int i = 1; i <= nbins; ++i){
        hSpec.SetBinContent(i, spec(global_bin_start + i -1));
        hSpec.SetBinError(i, error(global_bin_start + i -1));
    }

    return hSpec;

}


TH1D PROspec::toTH1D(const PROconfig& inconfig, const std::string& subchannel_fullname) const{
    int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
    return this->toTH1D(inconfig, subchannel_index);
}


void PROspec::toROOT(const PROconfig& inconfig, const std::string& output_name){

    TFile* f = new TFile(output_name.c_str(), "recreate");
    const std::vector<std::string>& all_subchannels = inconfig.m_fullnames;

    for(size_t i = 0; i!= all_subchannels.size(); ++i){
        auto& subchannel_fullname = all_subchannels[i];

        TH1D h = this->toTH1D(inconfig, subchannel_fullname);
        h.Write();
    }
    f->Write();
    f->Close();

    delete f;
    return;
}

bool PROspec::SameDim(const PROspec& a, const PROspec& b){
    if(a.nbins != b.nbins){
        log<LOG_ERROR>(L"%1% || Two spectra have different bins: %2 vs. %3") % __func__ % a.nbins % b.nbins;
        return false;
    }
    return true;
}

PROspec PROspec::operator+(const PROspec& b) const{
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXd sum_spec = this->spec + b.spec;
    Eigen::VectorXd error_spec = this->eigenvector_sqrt_quadrature_sum(this->error, b.error); 
    return PROspec(sum_spec, error_spec);
}

PROspec& PROspec::operator+=(const PROspec& b){
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    this->spec += b.spec;
    this->error = this->eigenvector_sqrt_quadrature_sum(this->error, b.error);

    return *this;
}

PROspec PROspec::operator-(const PROspec& b) const{
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXd sum_spec = this->spec - b.spec;
    Eigen::VectorXd error_spec = this->eigenvector_sqrt_quadrature_sum(this->error, b.error); 
    return PROspec(sum_spec, error_spec);
}

PROspec& PROspec::operator-=(const PROspec& b){
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    this->spec -= b.spec;
    this->error = this->eigenvector_sqrt_quadrature_sum(this->error, b.error);

    return *this;
}

PROspec PROspec::operator/(const PROspec& b) const{
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }


    Eigen::VectorXd ratio_spec = this->eigenvector_division(this->spec, b.spec);

    //calculate relative error, and sqrt of quadratic sum of relative error
    Eigen::VectorXd this_relative_error = this->eigenvector_division(this->error, this->spec), b_relative_error = this->eigenvector_division(b.error, b.spec);
    Eigen::VectorXd ratio_relative_error = this->eigenvector_sqrt_quadrature_sum(this_relative_error, b_relative_error);

    Eigen::VectorXd ratio_error = this->eigenvector_multiplication(ratio_spec, ratio_relative_error);
    return PROspec(ratio_spec, ratio_error); 
}

PROspec& PROspec::operator/=(const PROspec& b) {
    //check if dimension 
    if(!PROspec::SameDim(*this, b)){
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }



    this->spec =  this->eigenvector_division(this->spec, b.spec);
    //calculate relative error, and sqrt of quadratic sum of relative error
    Eigen::VectorXd this_relative_error = this->eigenvector_division(this->error, this->spec), b_relative_error = this->eigenvector_division(b.error, b.spec);
    Eigen::VectorXd ratio_relative_error = this->eigenvector_sqrt_quadrature_sum(this_relative_error, b_relative_error);

    this->error  = this->eigenvector_multiplication(this->spec, ratio_relative_error);
    return *this; 
}

PROspec PROspec::operator*(double scale) const{

    Eigen::VectorXd scaled_spec = scale * this->spec, scaled_error = scale * this->error;
    return PROspec(scaled_spec, scaled_error);
}

PROspec& PROspec::operator*=(double scale){

    this->spec *= scale; 
    this->error *= scale;
    return *this;
}

Eigen::VectorXd PROspec::eigenvector_sqrt_quadrature_sum(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const{
    int nbin = a.size();
    Eigen::VectorXd error_spec = Eigen::VectorXd::Zero(nbin); 
    error_spec = ((a.array()).square() + (b.array()).square()).sqrt();
    return error_spec;
}

Eigen::VectorXd PROspec::eigenvector_division(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const{
    int nbin = a.size();
    Eigen::VectorXd ratio_spec = Eigen::VectorXd::Zero(nbin);
    for(int i = 0; i != nbin; ++i){
        if(b(i) == 0){
            if(a(i) !=0 ){
                log<LOG_ERROR>(L"%1% || Divide by Zero. Numerator: %2%, denominator: %3% ") % __func__ % a(i) % b(i);
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                log<LOG_DEBUG>(L"%1% || Both numerator and denominator are zero, setting the ratio to 1.") % __func__;
                ratio_spec(i) = 1.0;
            }
        }else
            ratio_spec(i) = a(i) / b(i);
    }
    return ratio_spec;
}

Eigen::VectorXd PROspec::eigenvector_multiplication(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const{
    int nbin = a.size();
    Eigen::VectorXd ratio_spec = Eigen::VectorXd::Zero(nbin);
    for(int i = 0; i != nbin; ++i){
        ratio_spec(i) = a(i) * b(i);
    }
    return ratio_spec;
}

void PROspec::plotSpectrum(const PROconfig& inconfig, const std::string& output_name) const{

    bool div_bin = true;
    int n_subplots = inconfig.m_num_channels*inconfig.m_num_modes*inconfig.m_num_detectors;
    log<LOG_DEBUG>(L"%1% || Creatign canvas with  %2% subplots") % __func__ % n_subplots;

    TCanvas *c =  new TCanvas(output_name.c_str(), output_name.c_str(), 800*n_subplots, 600);
    c->Divide(n_subplots,1);

    std::vector<std::unique_ptr<TH1D>> hists;  
    std::vector<std::unique_ptr<THStack>> stacks;
    std::vector<std::unique_ptr<TLegend>> legs;

    size_t global_subchannel_index = 0;
    size_t global_channel_index = 0;
    for(size_t im = 0; im < inconfig.m_num_modes; im++){
        for(size_t id =0; id < inconfig.m_num_detectors; id++){
            for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                c->cd(global_channel_index+1);
               
                std::unique_ptr<THStack> s = std::make_unique<THStack>((output_name+std::to_string(global_channel_index)).c_str(),(output_name+std::to_string(global_channel_index)).c_str());
                stacks.push_back(std::move(s));

                std::unique_ptr<TLegend> leg = std::make_unique<TLegend>(0.59,0.89,0.59,0.89);
                leg->SetFillStyle(0);
                leg->SetLineWidth(0);
                legs.push_back(std::move(leg));
            

                for(size_t sc = 0; sc < inconfig.m_num_subchannels[ic]; sc++){
                    const std::string& subchannel_name  = inconfig.m_fullnames[global_subchannel_index];
                    const std::string& color = inconfig.m_subchannel_colors[ic][sc];
                    int rcolor = 0;
                    if(color=="NONE"){
                        rcolor = kRed-7; 
                    }else{
                        rcolor = inconfig.HexToROOTColor(color);
                    }


                    std::unique_ptr<TH1D> htmp = std::make_unique<TH1D>(toTH1D(inconfig, global_subchannel_index));
                    htmp->SetLineWidth(1);
                    htmp->SetLineColor(kBlack);
                    htmp->SetFillColor(rcolor);


                    hists.push_back(std::move(htmp));  // Move the unique_ptr into the vector

                    log<LOG_DEBUG>(L"%1% || Printot %2% %3% %4% %5% %6% : Integral %7% ") % __func__ % global_channel_index % global_subchannel_index % subchannel_name.c_str() % sc % ic % hists.back()->Integral();
            
                    stacks.back()->Add(hists.back().get());
                    legs.back()->AddEntry(hists.back().get(), inconfig.m_subchannel_plotnames[ic][sc].c_str() ,"f");


                    ++global_subchannel_index;
                }//end subchan


                if(div_bin){
                    TList *stlists = (TList*)stacks.back()->GetHists();
                    for(const auto&& obj: *stlists){
                        ((TH1*)obj)->Scale(1,"width");
                        log<LOG_DEBUG>(L"%1% || Stack contains  %2% ") % __func__ % obj->GetName();
                    }
                }

                log<LOG_DEBUG>(L"%1% || Stack Draw ") % __func__ ;
                stacks.back()->Draw("hist");
                log<LOG_DEBUG>(L"%1% || Legend Draw ") % __func__ ;
                legs.back()->Draw();

                stacks.back()->SetTitle((inconfig.m_mode_names[im]  +" "+ inconfig.m_detector_names[id]+" "+ inconfig.m_channel_names[ic]).c_str());
                stacks.back()->GetXaxis()->SetTitle(inconfig.m_channel_units[ic].c_str());
                stacks.back()->GetYaxis()->SetTitle("Events/GeV");

                ++global_channel_index;
            }//end chan
        }//end det
    }//end mode

    c->SaveAs(("PROplot_"+output_name+".pdf").c_str(),"pdf");

    delete c;
    return;
}
