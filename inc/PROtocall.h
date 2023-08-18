#ifndef PRO_TO_CALL_H
#define PRO_TO_CALL_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"

namespace PROfit{


    /* Function: given a value for reconstructed variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the reco value is out of range
     */
    int FindLocalBin(const PROconfig &inconfig, double reco_value, int channel_index);
    /* Function: given a value for reconstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the reco value is out of range, then return value of -1
     */
    int FindGlobalBin(const PROconfig &inconfig, double reco_value, int subchannel_index);
    int FindGlobalBin(const PROconfig &inconfig, double reco_value, const std::string& subchannel_fullname);

    /* Function: given a value for true variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the true value is out of range
     */
    int FindLocalTrueBin(const PROconfig &inconfig, double true_value, int channel_index);

    /* Function: given a value for truenstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the true value is out of range, then return value of -1
     */
    int FindGlobalTrueBin(const PROconfig &inconfig, double true_value, int subchannel_index);
    int FindGlobalTrueBin(const PROconfig &inconfig, double true_value, const std::string& subchannel_fullname);


    /* Function: given a global bin index in the full vector, return the index of the subchannle this bin belongs to
     * Parameter:
     * 	 	inconfig:     a reference to PROconfig object, needed for calculating index 
     * 	 	global_bin:   global bin index. It can be a global true bin index, or global reco bin index
     * 	 	reco_bin:     boolean. If set to true, function will assume given bin is global reco bin, and return associated global subchannel index. Otherwise return subchnnale index for true bin.
     * 	 		      Default to true.
     */
    int FindSubchannelIndexFromGlobalBin(const PROconfig &inconfig, int global_bin, bool reco_bin = true);

    /* Function: given a full matrix, collapse the matrix */
    Eigen::MatrixXd CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix);

    /* Function: given a full vector (that contains reco), collapse the vector */
    Eigen::VectorXd CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXd& full_vector);

    /* Function: given a full matrix, collapse the matrix (For loop version from SBNfit)  */
    Eigen::MatrixXd CollapseMatrix2(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix);

    /* Function: given a full vector (that contains reco), collapse the vector (For loop version from SBNfit )*/
    Eigen::VectorXd CollapseMatrix2(const PROconfig &inconfig, const Eigen::VectorXd& full_vector);

    Eigen::MatrixXd CollapseSubchannels(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix);
    Eigen::MatrixXd CollapseDetectors(const PROconfig &inconfig, const Eigen::MatrixXd& full_matrix);
};

#endif
