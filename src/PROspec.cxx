#include "PROspec.h"
using namespace PROfit;

PROspec::PROspec(int num_bins):
    nbins(num_bins),
    spec(Eigen::VectorXd::Zero(num_bins)),
    error(Eigen::VectorXd::Zero(num_bins)){
    }

int PROspec::GetNbins() const{
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
    //log<LOG_DEBUG>(L"%1% || Fill in weight: %2% to bin: %3%") % __func__ % weight % bin_index;
    spec(bin_index) += weight;
    error(bin_index) = std::sqrt(pow(error(bin_index), 2.0) + std::pow(weight, 2.0));
    return;
}

void PROspec::QuickFill(int bin_index, double weight){
    //log<LOG_DEBUG>(L"%1% || Fill in weight: %2% to bin: %3%") % __func__ % weight % bin_index;
    spec(bin_index) += weight;
    return;
}


TH1D PROspec::toTH1D(PROconfig const & inconfig, int subchannel_index){

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


TH1D PROspec::toTH1D(const PROconfig& inconfig, const std::string& subchannel_fullname){
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
    for(int i = 0; i != nbin; ++i){
	error_spec(i) = sqrt(pow(a(i), 2.0) + pow(b(i), 2.0));
    }
    return error_spec;
}

Eigen::VectorXd PROspec::eigenvector_division(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const{
    int nbin = a.size();
    Eigen::VectorXd ratio_spec = Eigen::VectorXd::Zero(nbin);
    for(int i = 0; i != nbin; ++i){
	if(b(i) == 0){
	    log<LOG_ERROR>(L"%1% || Divide by Zero") % __func__ ;
	    log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
	}
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

