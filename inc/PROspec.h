#ifndef PROSPEC_H_
#define PROSPEC_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ROOT
#include "TFile.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "THStack.h"
#include "TLegend.h"

// EIGEN
#include <Eigen/Dense>
#include <Eigen/SVD>

// PROfit
#include "PROconfig.h"

namespace PROfit{

    /* Class: Primary class for storing final spectra and associated errors.
     * Note: 
     *  A barebones class, as all binning, collapsing..etc.. is handeled by PROconfig. 
     *  Essentially two Eigen::VectorXd and associated functionality
     * Todo: 
     *  The toRoot/toTH1Dnot implemented. Do we need them?
     * */

    class PROspec {

        private:
            //Base
            size_t nbins;
            Eigen::VectorXd spec;
            Eigen::VectorXd error;


            //---- private helper function --------
            // Function: given two eigenvector of same dimension, calculate element-wise calculation of sqrt(a**2 + b**2) 
            Eigen::VectorXd eigenvector_sqrt_quadrature_sum(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise division a/b 
            Eigen::VectorXd eigenvector_division(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

            // Function: given two eigenvector of same dimension, calculate element-wise multiplication a*b 
            Eigen::VectorXd eigenvector_multiplication(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

        public:

            //Constructors
            PROspec():nbins(0) {}
            PROspec(const Eigen::VectorXd &in_spec, const Eigen::VectorXd &in_error) : nbins(in_spec.size()), spec(in_spec), error(in_error){}

            /* Function: create PROspec of given size */
            PROspec(size_t num_bins);

            //PROspec(PROconfig const & configin); //Load in config file EMPTY hists
            //PROspec(std::string &xmlname); //Load directly from XML 

            /* Function: Create a new PROspec with Poisson fluctuations applied. */
            static PROspec PoissonVariation(const PROspec &s);


            /* Function: given subchannel name/index, generate TH1D histogram of corresponding spectrum */
            TH1D toTH1D(const PROconfig& inconfig, int subchannel_index, int reco_dim=0) const;
            TH1D toTH1D(const PROconfig& inconfig, const std::string& subchannel_fullname, int reco_dim=0) const;


            /* Function: save TH1Ds of all subchannels into a root file */
            void toROOT(const PROconfig& inconfig, const std::string& output_name);

            /*Function: Create a pdf of all channels, with stacked subchannels*/
            void plotSpectrum(const PROconfig& inconfig, const std::string& output_name) const;

            /* Function: fill given bin with provided weight 
             * Note: 
             * 	 Both function do NOT check whether the given bin is out of range or not 
             * 	 Fill() updates the bin content and error, while QuickFill() only updates bin content and doesn't care error.
             * 	 Care when using QuickFill!! 
             *
             */
            void Fill(int bin_index, double weight);
            void QuickFill(int bin_index, double weight);

            /* Function: zero out the spectrum and error, but keep the dimension */
            void Zero();

            /* Function: Print out spec*/
            void Print() const;

            /*Return number of bins in spectrum */
            size_t GetNbins() const;



            /* Function:  Return the content of spectrum at given bin
             * Note: bin index starts at 0
             */
            inline
                double GetBinContent(int bin) const{
                    return spec(bin);
                }

            /* Function:  Return the bin error  at given bin
             * Note: bin index starts at 0
             */
            inline
                double GetBinError(int bin) const{
                    return error(bin);
                }

            /*Return reference to the core specturm */ 
            inline
                const Eigen::VectorXd& Spec() const{
                    return spec;
                }

            /*Return reference to the error specturm */ 
            inline
                const Eigen::VectorXd& Error() const{
                    return error;
                }


            /* Return true if two PROspec have the same dimension (number of bins */
            static bool SameDim(const PROspec& a, const PROspec& b);

            //----- Arithmetic Operations ---------
            //addition 
            PROspec operator+(const PROspec& b) const;
            //addition assignment
            PROspec& operator+=(const PROspec& b);
            //subtraction
            PROspec operator-(const PROspec& b) const;
            //subtraction assignment
            PROspec& operator-=(const PROspec& b);
            //division
            PROspec operator/(const PROspec& b) const;
            //division assignment
            PROspec& operator/=(const PROspec& b);
            //scaling (multiply with constant)
            PROspec operator*(double scale) const;
            //scaling assignmnet 
            PROspec& operator*=(double scale);
    };

}


#endif
