#include "PROtocall.h"
#include "PROlog.h"

namespace PROfit{

    int FindGlobalBin(const PROconfig &inconfig, float reco_value, const std::string& subchannel_fullname){
        int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
        return FindGlobalBin(inconfig, reco_value, subchannel_index);
    }

    int FindGlobalBin(const PROconfig &inconfig, float reco_value, int subchannel_index){
        int global_bin_start = inconfig.GetGlobalBinStart(subchannel_index);
        int channel_index = inconfig.GetChannelIndex(subchannel_index);
        int local_bin = FindLocalBin(inconfig, reco_value, channel_index);
        return local_bin == -1 ? -1 : global_bin_start + local_bin;
    }


    int FindLocalBin(const PROconfig &inconfig, float reco_value, int channel_index){
        //find local bin 
        const std::vector<float>& bin_edges = inconfig.GetChannelBinEdges(channel_index);
        auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), reco_value);

        //over/under-flow, don't care for now
        if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
            log<LOG_DEBUG>(L"%1% || Reco value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % reco_value;
            log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
            return -1; 
        }
        return pos_iter - bin_edges.begin() - 1; 
    }

    int FindGlobalTrueBin(const PROconfig &inconfig, float true_value, const std::string& subchannel_fullname){
        int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
        return FindGlobalTrueBin(inconfig, true_value, subchannel_index);
    }

    int FindGlobalTrueBin(const PROconfig &inconfig, float true_value, int subchannel_index){
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


    int FindLocalTrueBin(const PROconfig &inconfig, float true_value, int channel_index){
        //find local bin 
        const std::vector<float>& bin_edges = inconfig.GetChannelTrueBinEdges(channel_index);
        auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), true_value);

        //over/under-flow, don't care for now
        if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
            log<LOG_DEBUG>(L"%1% || True value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % true_value;
            log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
            return -1; 
        }
        return pos_iter - bin_edges.begin() - 1; 
    }

    int FindLocalOtherBin(const PROconfig &inconfig, float other_value, int channel_index, int other_index) {
        //find local bin 
        const std::vector<float>& bin_edges = inconfig.GetChannelOtherBinEdges(channel_index, other_index);
        auto pos_iter = std::upper_bound(bin_edges.begin(), bin_edges.end(), other_value);

        //over/under-flow, don't care for now
        if(pos_iter == bin_edges.end() || pos_iter == bin_edges.begin()){
            log<LOG_DEBUG>(L"%1% || True value: %2% is in underflow or overflow bins, return bin of -1") % __func__ % other_value;
            log<LOG_DEBUG>(L"%1% || Channel %2% has bin lower edge: %3% and bin upper edge: %4%") % __func__ % channel_index % *bin_edges.begin() % bin_edges.back();
            return -1; 
        }
        return pos_iter - bin_edges.begin() - 1; 
    }

    int FindGlobalOtherBin(const PROconfig &inconfig, float other_value, int subchannel_index, int other_index) {
        int global_bin_start = inconfig.GetGlobalOtherBinStart(subchannel_index, other_index);
        int channel_index = inconfig.GetChannelIndex(subchannel_index);
        if(inconfig.GetChannelNOtherBins(channel_index, other_index) == 0){
            log<LOG_ERROR>(L"%1% || Subchannel %2% does not have other bins") % __func__ % subchannel_index;
            log<LOG_ERROR>(L"%1% || Return global bin of -1") % __func__ ;
            return -1;
        }
        int local_bin = FindLocalOtherBin(inconfig, other_value, channel_index, other_index);
        return local_bin == -1 ? -1 : global_bin_start + local_bin;
    }


    int FindGlobalOtherBin(const PROconfig &inconfig, float other_value, const std::string& subchannel_fullname, int other_index) {
        int subchannel_index = inconfig.GetSubchannelIndex(subchannel_fullname);
        return FindGlobalOtherBin(inconfig, other_value, subchannel_index, other_index);
    }


    int FindSubchannelIndexFromGlobalBin(const PROconfig &inconfig, int global_bin, bool reco_bin ){
        if(reco_bin)
            return inconfig.GetSubchannelIndexFromGlobalBin(global_bin);
        else
            return inconfig.GetSubchannelIndexFromGlobalTrueBin(global_bin);
    }

    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix){
        Eigen::MatrixXf collapsing_matrix = inconfig.GetCollapsingMatrix();
        
        int num_bin_before_collapse = collapsing_matrix.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        //log<LOG_DEBUG>(L"%1% || CT  %2% x %3%. Full matrix: %4% x %5% ") % __func__ % collapsing_matrix.transpose().rows() %  collapsing_matrix.transpose().cols() % full_matrix.rows() % full_matrix.cols();
        Eigen::MatrixXf result_matrix   = collapsing_matrix.transpose()*full_matrix*collapsing_matrix;
        return result_matrix;
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector){
        Eigen::MatrixXf collapsing_matrix = inconfig.GetCollapsingMatrix();
        if(full_vector.size() != collapsing_matrix.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % collapsing_matrix.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf result_vector = collapsing_matrix.transpose() * full_vector;
        return result_vector;
    }

    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index){
        Eigen::MatrixXf collapsing_matrix = inconfig.GetCollapsingMatrix(other_index);
        
        int num_bin_before_collapse = collapsing_matrix.rows();
        if(full_matrix.rows() != num_bin_before_collapse || full_matrix.cols() != num_bin_before_collapse){
            log<LOG_ERROR>(L"%1% || Matrix dimension doesn't match expected size. Provided matrix: %2% x %3%. Expected matrix size: %4% x %5%") % __func__ % full_matrix.rows() % full_matrix.cols() % num_bin_before_collapse% num_bin_before_collapse;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        //log<LOG_DEBUG>(L"%1% || CT  %2% x %3%. Full matrix: %4% x %5% ") % __func__ % collapsing_matrix.transpose().rows() %  collapsing_matrix.transpose().cols() % full_matrix.rows() % full_matrix.cols();
        Eigen::MatrixXf result_matrix   = collapsing_matrix.transpose()*full_matrix*collapsing_matrix;
        return result_matrix;
    }

    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index){
        Eigen::MatrixXf collapsing_matrix = inconfig.GetCollapsingMatrix(other_index);
        if(full_vector.size() != collapsing_matrix.rows()){
            log<LOG_ERROR>(L"%1% || Vector dimension doesn't match expected size. Provided vector size: %2% . Expected size: %3%") % __func__ % full_vector.size() % collapsing_matrix.rows();
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf result_vector = collapsing_matrix.transpose() * full_vector;
        return result_vector;
    }

    std::string getIcon(){

        std::string icon = R"(
       $$$$$                                                                                                                                                             $$$$$      
       $:::$                                                                                                                                                             $:::$      
   $$$$$:::$$$$$$                  PPPPPPPPPPPPPPPPP   RRRRRRRRRRRRRRRRR        OOOOOOOOO        ffffffffffffffff    iiii          tttt                              $$$$$:::$$$$$$ 
 $$::::::::::::::$                 P::::::::::::::::P  R::::::::::::::::R     OO:::::::::OO     f::::::::::::::::f  i::::i      ttt:::t                            $$::::::::::::::$
$:::::$$$$$$$::::$                 P::::::PPPPPP:::::P R::::::RRRRRR:::::R  OO:::::::::::::OO  f::::::::::::::::::f  iiii       t:::::t                           $:::::$$$$$$$::::$
$::::$       $$$$$                 PP:::::P     P:::::PRR:::::R     R:::::RO:::::::OOO:::::::O f::::::fffffff:::::f             t:::::t                           $::::$       $$$$$
$::::$                               P::::P     P:::::P  R::::R     R:::::RO::::::O   O::::::O f:::::f       ffffffiiiiiiittttttt:::::ttttttt                     $::::$            
$::::$                               P::::P     P:::::P  R::::R     R:::::RO:::::O     O:::::O f:::::f             i:::::it:::::::::::::::::t                     $::::$            
$:::::$$$$$$$$$                      P::::PPPPPP:::::P   R::::RRRRRR:::::R O:::::O     O:::::Of:::::::ffffff        i::::it:::::::::::::::::t                     $:::::$$$$$$$$$   
 $$::::::::::::$$  ---------------   P:::::::::::::PP    R:::::::::::::RR  O:::::O     O:::::Of::::::::::::f        i::::itttttt:::::::tttttt     ---------------  $$::::::::::::$$ 
   $$$$$$$$$:::::$ -:::::::::::::-   P::::PPPPPPPPP      R::::RRRRRR:::::R O:::::O     O:::::Of::::::::::::f        i::::i      t:::::t           -:::::::::::::-    $$$$$$$$$:::::$
            $::::$ ---------------   P::::P              R::::R     R:::::RO:::::O     O:::::Of:::::::ffffff        i::::i      t:::::t           ---------------             $::::$
            $::::$                   P::::P              R::::R     R:::::RO:::::O     O:::::O f:::::f              i::::i      t:::::t                                       $::::$
$$$$$       $::::$                   P::::P              R::::R     R:::::RO::::::O   O::::::O f:::::f              i::::i      t:::::t    tttttt                 $$$$$       $::::$
$::::$$$$$$$:::::$                 PP::::::PP          RR:::::R     R:::::RO:::::::OOO:::::::Of:::::::f            i::::::i     t::::::tttt:::::t                 $::::$$$$$$$:::::$
$::::::::::::::$$                  P::::::::P          R::::::R     R:::::R OO:::::::::::::OO f:::::::f            i::::::i     tt::::::::::::::t                 $::::::::::::::$$ 
 $$$$$$:::$$$$$                    P::::::::P          R::::::R     R:::::R   OO:::::::::OO   f:::::::f            i::::::i       tt:::::::::::tt                  $$$$$$:::$$$$$   
      $:::$                        PPPPPPPPPP          RRRRRRRR     RRRRRRR     OOOOOOOOO     fffffffff            iiiiiiii         ttttttttttt                         $:::$       
      $$$$$                                                                                                                                                             $$$$$       
      )";                                                                                                                                                                                    
                                                                                                                                                                                    
                                                                                                                                                                                    
               return icon;
    };
};
