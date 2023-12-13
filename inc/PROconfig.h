#ifndef PROCONFIG_H_
#define PROCONFIG_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <unordered_map>
#include <climits>
#include <cstdlib>
#include <numeric>

// TINYXML2
#include "tinyxml2.h"

// EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

//PROfit
#include "PROlog.h"

//ROOT
#include "TTreeFormula.h"


/*eweight_type here to switch between uboone style "double" and SBNcode style "float"  */
#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
typedef float eweight_type;
#else
typedef double eweight_type;
#endif

namespace PROfit{

    /*typedef the base "eventweight_class" branch that reweightable systematics are stored in in uboonestyle systematics 
    */
    typedef std::map<std::string, std::vector<eweight_type>> eweight_map;


    /* Struct: Branch variable is a SBNfit era class to load using TTReeFormula a givem variable (or function of variables) 
     * Note: was originally split between float/int, but moved to TTreeFormula
     */

    struct BranchVariable{
        std::string name;
        std::string type;
        std::string associated_hist;
        std::string associated_systematic;
        bool central_value;

        std::shared_ptr<TTreeFormula> branch_formula=nullptr;
        std::shared_ptr<TTreeFormula> branch_monte_carlo_weight_formula = nullptr;
        std::shared_ptr<TTreeFormula> branch_true_value_formula=nullptr;
        std::shared_ptr<TTreeFormula> branch_true_L_formula=nullptr;
        std::shared_ptr<TTreeFormula> branch_true_pdg_formula=nullptr;

        bool oscillate;
        std::string true_param_name;
        std::string true_L_name;
        std::string pdg_name;

	//constructor
        BranchVariable(std::string n, std::string t, std::string a) : name(n), type(t), associated_hist(a), central_value(false), oscillate(false){}
        BranchVariable(std::string n, std::string t, std::string a_hist, std::string a_syst, bool cv) : name(n), type(t), associated_hist(a_hist), associated_systematic(a_syst), central_value(cv), oscillate(false){}

	/* Function: Return the TTreeformula for branch 'name', usually it's the reconstructed variable */
        std::shared_ptr<TTreeFormula> GetFormula(){
            return branch_formula;
        }

        void SetOscillate(bool inbool){ oscillate = inbool; return;}
        bool GetOscillate() const { return oscillate;}
	void SetTrueParam(const std::string& true_parameter_def){ true_param_name = true_parameter_def; return;}
   	void SetPDG(const std::string& pdg_def){ pdg_name = pdg_def; return;}
	void SetTrueL(const std::string& true_L_def){true_L_name = true_L_def; return;}

	//Function: evaluate branch "pdg", and return the value. Usually it's the pdg value of the particle
	//Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
	template <typename T = int>
	T GetTruePDG() const;


	// Function: evaluate additional weight setup in the branch and return in floating precision 
	// Note: if no additional weight is set, value of 1.0 will be returned.
	inline
	double GetMonteCarloWeight() const{
	    if(branch_monte_carlo_weight_formula){ 
		branch_monte_carlo_weight_formula->GetNdata();
		return (double)branch_monte_carlo_weight_formula->EvalInstance();
	    }
	    return 1.0;
	}


        //Function: evaluate branch 'name' and return the value. Usually its reconstructed quantity
	//Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
	template <typename T=double>
        T GetValue() const;


        //Function: evaluate formula 'true_L_name' and return the value. Usually it's true baseline.
	//Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
        template <typename T=double>
	T GetTrueL() const;


        //Function: evaluate formula 'true_param_name' and return the value. Usually it's true energy  
	//Note: when called, if the corresponding TreeFormula is not linked to a TTree, value of ZERO (0) will be returned.
        template <typename T=double>
	T GetTrueValue() const;
    };



    /* 
     * Class: Primary booking class for loading in info from XML and saving all information on mode_detector_channel_subchannel information.
     * Note:
     *  PROconfig to be created once and only once per PROfit executable and then passed by reference to various more complex functions, and ignored when not needed. 
     *  Contains information and collapsing matricies for collapsing subchannels->channels also. Filled once, on construction and then used by later classes when needed.
     */


    class PROconfig {
        private:

            //indicator of whether each channel/detector/subchannel is used
            std::vector<bool> m_mode_bool;
            std::vector<bool> m_detector_bool;
            std::vector<bool> m_channel_bool;
            std::vector<std::vector<bool>>  m_subchannel_bool;


            //map from subchannel name/index to global index and channel index
            std::unordered_map<std::string, int> m_map_fullname_subchannel_index;
	    std::vector<int> m_vec_subchannel_index; //vector of global subchannel index, in increasing order
	    std::vector<int> m_vec_channel_index;    //vector of corresponding channel index
	    std::vector<int> m_vec_global_reco_index_start;  //vector of global reco bin index, in increasing order
	    std::vector<int> m_vec_global_true_index_start;  //vector of global true bin index, in increasing order


            //---- PRIVATE FUNCTION ------

            /* Function: construct a matrix T, which will be used to collapse matrix and vectors */
            void construct_collapsing_matrix();

            /* Function: remove any mode/detector/channel/subchannels in the configuration xml that are not used from consideration
            */
            void remove_unused_channel();


            /* Function: ignore any file that is associated with unused channels 
            */
            void remove_unused_files();


            /* Function: fill in mapping between subchannel name/index to global indices */
            void generate_index_map();


	    /* Function: given an input vector that's sorted in ascending order, and input val, return the index of elmeent which is equal to val
 	     * Note: it gives exception when input value is not present in the vector 
	     */
	    size_t find_equal_index(const std::vector<int>& input_vec, int val) const;

	    /* Function: given an input vector that's sorted in ascending order, and input val, return the index of the closest element which is equal or smaller than val */
	    size_t find_less_or_equal_index(const std::vector<int>& input_vec, int val) const;

            /* Function: given global bin index, return associated global subchannel index 
 	     * Note: not used anymore 
 	     */
            int find_global_subchannel_index_from_global_bin(int global_index, const std::vector<int>& num_subchannel_in_channel, const std::vector<int>& num_bins_in_channel, int num_channels, int num_bins_total) const;


        public:

            PROconfig() {}; //always have an empty constructor?

            /* Constructor Function: Need a string passed which is the filename (with path) of the configuration xml */
            PROconfig(const std::string &xmlname);

            /*
             * Function: Use TinyXML2 to load XML */
            int LoadFromXML(const std::string & filename);


            std::string m_xmlname;	
            double m_plot_pot;
            std::vector<std::string> m_fullnames;

            int m_num_detectors;
            int m_num_channels;
            int m_num_modes;

            /*Vectors of length num_channels. Unless specificed all refer to fittable (reco) variables*/
            std::vector<int> m_num_subchannels; 
            std::vector<int> m_channel_num_bins;
            std::vector<std::vector<double> > m_channel_bin_edges;
            std::vector<std::vector<double> > m_channel_bin_widths;

            /* New true bins to save the truth level variables in addition.*/
            std::vector<int> m_channel_num_truebins;
            std::vector<std::vector<double> > m_channel_truebin_edges;
            std::vector<std::vector<double> > m_channel_truebin_widths;


            bool m_has_oscillation_patterns;


            //the xml names are the way we track which channels and subchannels we want to use later
            std::vector<std::string> m_mode_names; 			
            std::vector<std::string> m_mode_plotnames; 			

            std::vector<std::string> m_detector_names; 		
            std::vector<std::string> m_detector_plotnames; 		

            std::vector<std::string> m_channel_names; 		
            std::vector<std::string> m_channel_plotnames; 		
            std::vector<std::string> m_channel_units; 		


            std::vector<std::vector<std::string >> m_subchannel_names; 
            std::vector<std::vector<std::string >> m_subchannel_plotnames; 
            std::vector<std::vector<int >> m_subchannel_datas; 
            std::vector<std::vector<int> > m_subchannel_osc_patterns; 

            int m_num_bins_detector_block;
            int m_num_bins_mode_block;
            int m_num_bins_total;

            int m_num_truebins_detector_block;
            int m_num_truebins_mode_block;
            int m_num_truebins_total;

            int m_num_bins_detector_block_collapsed;
            int m_num_bins_mode_block_collapsed;
            int m_num_bins_total_collapsed;

            /* Eigen Matrix for collapsing subchannels->channels*/
            Eigen::MatrixXd collapsing_matrix;

            //This section entirely for montecarlo generation of a covariance matrix or PROspec 
            //For generating a covariance matrix from scratch, this contains the number of montecarlos (weights in weight vector) and their names.
            bool m_write_out_variation;
            bool m_form_covariance;
            std::string m_write_out_tag;


            int m_num_mcgen_files;
            std::vector<std::string> m_mcgen_tree_name;	
            std::vector<std::string> m_mcgen_file_name;	
            std::vector<long int> m_mcgen_maxevents;	
            std::vector<double> m_mcgen_pot;	
            std::vector<double> m_mcgen_scale;	
            std::vector<bool> m_mcgen_fake;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_map;
            std::map<std::string,std::vector<std::string>> m_mcgen_file_friend_treename_map;
            std::vector<std::vector<std::string>> m_mcgen_additional_weight_name;
            std::vector<std::vector<bool>> m_mcgen_additional_weight_bool;
            std::vector<std::vector<std::shared_ptr<BranchVariable>>> m_branch_variables;
            std::vector<std::vector<std::string>> m_mcgen_eventweight_branch_names;


            //specific bits for covariancegeneration
            std::vector<std::string> m_mcgen_weightmaps_formulas;
            std::vector<bool> m_mcgen_weightmaps_uses;
            std::vector<std::string> m_mcgen_weightmaps_patterns;
            std::vector<std::string> m_mcgen_weightmaps_mode;
            std::unordered_set<std::string> m_mcgen_variation_allowlist;
            std::unordered_set<std::string> m_mcgen_variation_denylist;
            std::map<std::string, std::vector<std::string>> m_mcgen_shapeonly_listmap; //a map of shape-only systematic and corresponding subchannels

            //FIX skepic
            std::vector<std::string> systematic_name;


            //----- PUBLIC FUNCTIONS ------
            //


            /* Function: return matrix T, of size (m_num_bins_total, m_num_bins_total_collapsed), which will be used to collapse matrix and vectors 
             * Note: To collapse a full matrix M, please do T.transpose() * M * T
             * 	     To collapse a full vector V, please do T.transpose() * V
             */
            inline 
                Eigen::MatrixXd GetCollapsingMatrix() const {return collapsing_matrix; }

            /* Function: Calculate how big each mode block and decector block are, for any given number of channels/subchannels, before and after the collapse
             * Note: only consider mode/detector/channel/subchannels that are actually used 
             */
            void CalcTotalBins();


            /* Function: given subchannel full name, return global subchannel index 
             * Note: index start from 0, not 1
             */
            int GetSubchannelIndex(const std::string& fullname) const;

            /* Function: given global index (in the full vector), return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            int GetSubchannelIndexFromGlobalBin(int global_index) const;

            /* Function: given global true index , return global subchannel index of associated subchannel
             * Note: returns a 0-based index 
             */
            int GetSubchannelIndexFromGlobalTrueBin(int global_trueindex) const;

            /* Function: given subchannel global index, return corresponding channel index 
             * Note: index start from 0, not 1
             */
            int GetChannelIndex(int subchannel_index) const;


            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            int GetGlobalBinStart(int subchannel_index) const;


            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<double>& GetChannelBinEdges(int channel_index) const;

            /* Function: given channel index, return number of true bins for this channel */
            int GetChannelNTrueBins(int channel_index) const;

            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            int GetGlobalTrueBinStart(int subchannel_index) const;

            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<double>& GetChannelTrueBinEdges(int channel_index) const;

    };


//----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------
//----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------
//----------- BELOW: Definition of BranchVariable templated member function. Please don't move it elsewhere !! ---------------

template <typename T>
T BranchVariable::GetTruePDG() const{
     if(branch_true_pdg_formula == NULL) return static_cast<T>(0);
    else{
        branch_true_pdg_formula->GetNdata();
        return static_cast<T>(branch_true_pdg_formula->EvalInstance());
    }
}


template <typename T>
T BranchVariable::GetValue() const{
    if(branch_formula == NULL) return static_cast<T>(0);
    else{
        branch_formula->GetNdata();
        return static_cast<T>(branch_formula->EvalInstance());
    }
}

template <typename T>
T BranchVariable::GetTrueL() const{
    if(branch_true_L_formula == NULL) return static_cast<T>(0);
    else{
        branch_true_L_formula->GetNdata();
        return static_cast<T>(branch_true_L_formula->EvalInstance());
    }
}

template <typename T>
T BranchVariable::GetTrueValue() const{
    if(branch_true_value_formula == NULL) return static_cast<T>(0);
    else{
        branch_true_value_formula->GetNdata();
        return static_cast<T>(branch_true_value_formula->EvalInstance());
    }
}
//----------- ABOVE: Definition of BranchVariable templated member function. END ---------------

}
#endif
