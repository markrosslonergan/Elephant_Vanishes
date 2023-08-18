#include "PROtocall.h"

namespace PROfit{

int FindGlobalBin(const PROconfig &inconfig, double reco_value, const std::string& subchannel_fullname){
    int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
    return FindGlobalBin(inconfig, reco_value, subchannel_index);
}

int FindGlobalBin(const PROconfig &inconfig, double reco_value, int subchannel_index){
    int global_bin_start = inconfig.GetGlobalBinStart(subchannel_index);
    int channel_index = inconfig.GetChannelIndex(subchannel_index);
    int local_bin = FindLocalBin(inconfig, reco_value, channel_index);
    return local_bin == -1 ? -1 : global_bin_start + local_bin;
}


int FindLocalBin(const PROconfig &inconfig, double reco_value, int channel_index){
    
    //find local bin 
    const std::vector<double>& bin_edges = inconfig.GetChannelBinEdges(channel_index);
    auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), reco_value);

    //over/under-flow, don't care for now
    if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
	log<LOG_DEBUG>(L"%1% || Reco value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % reco_value;
	log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
	return -1; 
    }
    return pos_iter - bin_edges.begin() - 1; 
}

int FindGlobalTrueBin(const PROconfig &inconfig, double true_value, const std::string& subchannel_fullname){
    int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
    return FindGlobalTrueBin(inconfig, true_value, subchannel_index);
}

int FindGlobalTrueBin(const PROconfig &inconfig, double true_value, int subchannel_index){
    int global_bin_start = inconfig.GetGlobalTrueBinStart(subchannel_index);
    int channel_index = inconfig.GetChannelIndex(subchannel_index);
    if(inconfig.GetChannelNTrueBins(channel_index) == 0){
	log<LOG_ERROR>(L"%1% || Subchannel %2% does not have true bins") % __func__ % subchannel_index;
	log<LOG_ERROR>(L"%1% || Return global bin of -1") % __func__ ;
	return -1;
    }
    int local_bin = FindLocalTrueBin(inconfig, true_value, channel_index);
    return local_bin == -1 ? -1 : global_bin_start + local_bin;
}


int FindLocalTrueBin(const PROconfig &inconfig, double true_value, int channel_index){
    
    //find local bin 
    const std::vector<double>& bin_edges = inconfig.GetChannelTrueBinEdges(channel_index);
    auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), true_value);

    //over/under-flow, don't care for now
    if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
	log<LOG_DEBUG>(L"%1% || True value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % true_value;
	log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
	return -1; 
    }
    return pos_iter - bin_edges.begin() - 1; 
}

int FindSubchannelIndexFromGlobalBin(const PROconfig &inconfig, int global_bin, bool reco_bin ){
   if(reco_bin)
   	return inconfig.GetSubchannelIndexFromGlobalBin(global_bin);
   else
	return inconfig.GetSubchannelIndexFromGlobalTrueBin(global_bin);
}

Eigen::MatrixXd CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix){
    Eigen::MatrixXd collapsing_matrix = inconfig.GetCollapsingMatrix();
    int num_bin_before_collapse = collapsing_matrix.rows();
    if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
	log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
	log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    log<LOG_DEBUG>(L"%1% || Collapsed matrix will be %2% x %3%") % __func__ % collapsing_matrix.cols() % collapsing_matrix.cols();
    Eigen::MatrixXd result_matrix = collapsing_matrix.transpose() * full_matrix * collapsing_matrix;
    return result_matrix;
}

Eigen::VectorXd CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXd& full_vector){
    Eigen::MatrixXd collapsing_matrix = inconfig.GetCollapsingMatrix();
    if(full_vector.size() != collapsing_matrix.rows()){
	log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % collapsing_matrix.rows();
	log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    Eigen::VectorXd result_vector = collapsing_matrix.transpose() * full_vector;
    return result_vector;
}


Eigen::MatrixXd CollapseSubchannels(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix){
    Eigen::MatrixXd result(inconfig.m_num_bins_detector_block_collapsed,inconfig.m_num_bins_detector_block_collapsed);

    std::vector<std::vector<Eigen::MatrixXd>> Summed(inconfig.m_num_channels, std::vector<Eigen::MatrixXd>(inconfig.m_num_channels) );	
    for(int ic = 0; ic < inconfig.m_num_channels; ic++){
        for(int jc =0; jc < inconfig.m_num_channels; jc++){
            Summed[ic][jc].resize(inconfig.m_channel_num_bins[jc],inconfig.m_channel_num_bins[ic]); 
            Summed[ic][jc].setConstant(0);
        }
    }

    int mrow = 0.0;
    int mcol = 0.0;

    for(int ic = 0; ic < inconfig.m_num_channels; ic++){ 	 //Loop over all rows
        for(int jc =0; jc < inconfig.m_num_channels; jc++){ //Loop over all columns

            for(int m=0; m < inconfig.m_num_subchannels[ic]; m++){
                for(int n=0; n< inconfig.m_num_subchannels[jc]; n++){ //For each big block, loop over all subchannels summing toGether
                         int x1 = mrow+n*inconfig.m_channel_num_bins[jc];
                         int x2 = mrow + n*inconfig.m_channel_num_bins[jc]+inconfig.m_channel_num_bins[jc]-1;
                         int x3 = mcol + m*inconfig.m_channel_num_bins[ic];
                         int x4 =mcol+ m*inconfig.m_channel_num_bins[ic]+inconfig.m_channel_num_bins[ic]-1 ;
                         Summed[ic][jc] +=  full_matrix(Eigen::seq(x1,x2), Eigen::seq(x3,x4));
                }
            }
            mrow += inconfig.m_num_subchannels[jc]*inconfig.m_channel_num_bins[jc];//As we work our way left in columns, add on that many bins
        }//end of column loop

        mrow = 0; // as we end this row, reSet row count, but jump down 1 column
        mcol += inconfig.m_num_subchannels[ic]*inconfig.m_channel_num_bins[ic];
    }//end of row loop

    ///********************************* And put them back toGether! ************************//
    mrow = 0;
    mcol = 0;

    //Repeat again for Contracted matrix
    for(int ic = 0; ic < inconfig.m_num_channels; ic++){
        for(int jc =0; jc < inconfig.m_num_channels; jc++){

            int sizer = Summed[ic][jc].rows();
            result(Eigen::seqN(mrow,sizer), Eigen::seqN(mcol,sizer)) = Summed[ic][jc];

            mrow += inconfig.m_channel_num_bins[jc];
        }

        mrow = 0;
        mcol +=inconfig.m_channel_num_bins[ic];
    }

    return result;
}


Eigen::MatrixXd CollapseDetectors(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix){

    //FINISH
    Eigen::MatrixXd result(inconfig.m_num_bins_detector_block_collapsed,inconfig.m_num_bins_detector_block_collapsed);

    /*
    Mc.Zero();
    int nrow = num_bins_detector_block;// N_e_bins*N_e_spectra+N_m_bins*N_m_spectra;
    int crow = num_bins_detector_block_compressed; //N_e_bins+N_m_bins;

    for(int m =0; m< num_detectors; m++){
        for(int n =0; n< num_detectors; n++){
            TMatrixT<double> imat(nrow,nrow);
            TMatrixT<double> imatc(crow,crow);

            imat = M.GetSub(n*nrow,n*nrow+nrow-1, m*nrow,m*nrow+nrow-1);
            CollapseSubchannels(imat,imatc);
            Mc.SetSub(n*crow,m*crow,imatc);
        }
    }
    */
    return result;
}



Eigen::MatrixXd CollapseMatrix2(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix){

    Eigen::MatrixXd result(inconfig.m_num_bins_detector_block_collapsed,inconfig.m_num_bins_detector_block_collapsed);

    /*
    Mc.Zero();
    int nrow = num_bins_mode_block;// (N_e_bins*N_e_spectra+N_m_bins*N_m_spectra)*N_dets;
    int crow=  num_bins_mode_block_compressed;// (N_e_bins+N_m_bins)*N_dets;

    for(int m =0; m< num_modes ; m++){
        for(int n =0; n< num_modes; n++){

            TMatrixT<double> imat(nrow,nrow);
            TMatrixT<double> imatc(crow,crow);

            imat = M.GetSub(n*nrow,n*nrow+nrow-1, m*nrow,m*nrow+nrow-1);

            CollapseDetectors(imat,imatc);
            Mc.SetSub(n*crow,m*crow,imatc);

        }
    }
    */
    return result;
}

Eigen::VectorXd CollapseVector2(const PROconfig &inconfig, const Eigen::VectorXd& full_vector){
    Eigen::MatrixXd collapsing_matrix = inconfig.GetCollapsingMatrix();
    if(full_vector.size() != collapsing_matrix.rows()){
	log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % collapsing_matrix.rows();
	log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    Eigen::VectorXd result_vector = collapsing_matrix.transpose() * full_vector;
    return result_vector;
}



};
