/**
 * @file PROtocall.h
 * @brief Utility functions for bin-finding, matrix collapsing, and miscellaneous helpers.
 * @author PROfit Collaboration
 *
 * @details Provides the lower-level utility functions used throughout PROfit for:
 *   - mapping continuous variable values to global or local bin indices,
 *   - collapsing full (mode×detector×channel×subchannel) matrices and vectors into
 *     channel-level representations using the collapsing matrix from PROconfig,
 *   - computing square-root (Cholesky or SVD) factorisations of covariance matrices, and
 *   - printing debug information about the variable/binning layout.
 */
#ifndef PRO_TO_CALL_H
#define PRO_TO_CALL_H

// C++ include 
#include <algorithm>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <stdexcept>

// PROfit include 
#include "PROlog.h"
#include "PROconfig.h"

// Root includes
namespace PROfit{


       /* Function: given a value for true variable, figure out which local bin in the histogram it belongs to
     * Note: bin index start from 0, not 1
     * Note: return value of -1 means the true value is out of range
     */
    int FindLocalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int channel_index, int other_index);

    /* Function: given a value for true or reconstructed variable, figure out which global bin it belongs to
     * Note: bin index start from 0, not 1
     * Note: if  the true value is out of range, then return value of -1
     */
    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, int subchannel_index, int other_index);
    int FindGlobalVariableBin(const PROconfig &inconfig, const BranchVariable::Value &other_value, const std::string& subchannel_fullname, int other_index);


    /**
     * @brief Collapse a full mode×detector×channel×subchannel covariance matrix to the channel level.
     * @details Uses the primary collapsing matrix (variable index i_prime) from @p inconfig.
     * The collapsed matrix M' = T^T M T where T is the collapsing matrix.
     * @param inconfig    Analysis configuration providing the collapsing matrix.
     * @param full_matrix Full covariance matrix (size m_num_variable_bins_total × same).
     * @return Collapsed covariance matrix.
     */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix);

    /**
     * @brief Collapsed, spectrum-scaled covariance: T^T diag(spec) F diag(spec) T.
     * @details Computes S^T F S with S = diag(spec)*T kept sparse, so the full-binning
     * dense N x N matrix diag(spec)*F*diag(spec) is never materialized. This is the
     * per-evaluation systematic covariance used by the chi^2 metrics — algebraically
     * identical to CollapseMatrix(config, spec.asDiagonal()*F*spec.asDiagonal()) but
     * without the two dense N x N temporaries per call.
     * @param inconfig    Analysis configuration providing the collapsing matrix (i_prime).
     * @param frac_cov    Fractional covariance on the full (uncollapsed) binning.
     * @param spec        Spectrum used for the diagonal scaling (full binning).
     * @return Collapsed m x m covariance matrix.
     */
    Eigen::MatrixXf CollapsedScaledCovariance(const PROconfig &inconfig, const Eigen::MatrixXf& frac_cov, const Eigen::VectorXf& spec);

    /**
     * @brief Collapse a full spectrum vector to the channel level using the primary variable.
     * @param inconfig    Analysis configuration providing the collapsing matrix.
     * @param full_vector Full spectrum vector (size m_num_variable_bins_total for primary variable).
     * @return Collapsed spectrum vector.
     */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector);

    /**
     * @brief Collapse a full covariance matrix to the channel level for a specified variable.
     * @param inconfig    Analysis configuration.
     * @param full_matrix Full covariance matrix.
     * @param other_index Variable index selecting which collapsing matrix to use.
     * @return Collapsed covariance matrix.
     */
    Eigen::MatrixXf CollapseMatrix(const PROconfig &inconfig, const Eigen::MatrixXf& full_matrix, int other_index);

    /**
     * @brief Collapse a full spectrum vector for a specified analysis variable.
     * @param inconfig    Analysis configuration.
     * @param full_vector Full spectrum vector for the given variable.
     * @param other_index Variable index.
     * @return Collapsed spectrum vector.
     */
    Eigen::VectorXf CollapseMatrix(const PROconfig &inconfig, const Eigen::VectorXf& full_vector, int other_index);

    /**
     * @brief Shape-only per-channel normalisation factors r_c = sum_c(data) / sum_c(pred).
     * @details One factor per collapsed (mode x detector x channel) block, returned expanded
     * to every collapsed bin of that block. A channel with sum_c(pred) <= 0 gets r_c = 1.
     * This is THE shape-only convention (v3.1+): the PREDICTION is rescaled onto the
     * data's channel integral and compared against the untouched data, so the chi^2 is
     * exactly invariant under pred -> k*pred and the statistical term never moves with
     * the fit. (Pre-v3.1 the data was rescaled instead, which made the Neyman/CNP/Poisson
     * stat term shrink with the prediction and biased fits toward lower total rate.)
     * @param inconfig       Analysis configuration.
     * @param collapsed_pred Prediction in the collapsed bin space of @p other_index.
     * @param data           Observed (collapsed) data spectrum.
     * @param other_index    Variable index.
     * @return Per-collapsed-bin factor vector.
     */
    Eigen::VectorXf ChannelNormFactors(const PROconfig &inconfig, const Eigen::VectorXf &collapsed_pred, const Eigen::VectorXf &data, int other_index);

    /**
     * @brief Shape-only: rescale a full-binning prediction onto the data per channel.
     * @details Returns spec .* (T r) with r = ChannelNormFactors(T^T spec, data): every
     * subchannel bin of channel c is multiplied by the same r_c. Optionally hands back r.
     * @param inconfig    Analysis configuration.
     * @param full_spec   Prediction on the full (uncollapsed) binning of @p other_index.
     * @param data        Observed (collapsed) data spectrum.
     * @param other_index Variable index.
     * @param r_out       If non-null, receives the collapsed-space factor vector.
     * @return The rescaled full-binning prediction.
     */
    Eigen::VectorXf ShapeRescaleToData(const PROconfig &inconfig, const Eigen::VectorXf &full_spec, const Eigen::VectorXf &data, int other_index, Eigen::VectorXf *r_out = nullptr);

    /**
     * @brief Return the PROfit ASCII-art icon string.
     * @return Icon string for display at programme startup.
     */
    std::string getIcon();

    /**
     * @brief Print a debug summary of all variables and their binning from PROconfig.
     * @param inconfig  Analysis configuration to inspect.
     */
    void PrintVariableInfo(const PROconfig &inconfig);


    /**
     * @brief Convert a numeric value to a fixed-precision string.
     * @tparam T  Arithmetic type.
     * @param a_value  Value to format.
     * @param n        Number of decimal places (default 6).
     * @return Fixed-precision string representation.
     */
    template <typename T>
    std::string to_string_prec(const T a_value, const int n = 6)    {
      std::ostringstream out;
      out <<std::fixed<< std::setprecision(n) << a_value;
      return out.str();
    }

    /**
     * @brief Compute the square-root (Cholesky or SVD) factor of a covariance matrix.
     * @details Returns L such that L * L^T ≈ covariance.  Uses LDLT decomposition by default;
     * falls back to SVD when the matrix is not positive definite or when @p svd_tol >= 0.
     * @param covariance  Input symmetric positive-semi-definite covariance matrix.
     * @param use_ldlt    If true (default), attempt LDLT factorisation first.
     * @param svd_tol     Singular-value tolerance for the SVD fallback; negative means do not use SVD.
     * @return Lower-triangular square-root factor L.
     */
    Eigen::MatrixXf ComputeSquareRootCovariance(const Eigen::MatrixXf& covariance, bool use_ldlt = true, float svd_tol = -1.0);

};

#endif
