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

#define TYPE_FLOAT
#ifdef TYPE_FLOAT  
typedef float eweight_type;
#else
typedef double eweight_type;
#endif

namespace PROfit{

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

        bool oscillate;
        std::string true_param_name;
        std::string true_L_name;

        float value_f;
        float true_value_f;
        float true_L_f;

        double value_d;
        double true_value_d;
        double true_L_d;

        BranchVariable(std::string n, std::string t, std::string a) : name(n), type(t), associated_hist(a) {oscillate=false; associated_systematic=""; central_value =false;}
        BranchVariable(std::string n, std::string t, std::string a_hist, std::string a_syst, bool cv) : name(n), type(t), associated_hist(a_hist), associated_systematic(a_syst) { oscillate=false; central_value=cv;}
        virtual void* GetValue(){return nullptr;};  //Function: evaluate branch 'name' and return the value. Usually its reconstructed quantity
        virtual void* GetTrueValue(){return nullptr;};  //Function: evaluate formula 'true_param_name' and return the value. Usually it's true energy  
        virtual void* GetTrueL(){return nullptr;};      //Function: evaluate formula 'true_L_name' and return the value. Usually it's true baseline.


	/* Function: Return the TTreeformula for branch 'name', usually it's the reconstructed variable */
        std::shared_ptr<TTreeFormula> GetFormula(){
            return branch_formula;
        }

        void SetOscillate(bool inbool){ oscillate = inbool; return;}
        bool GetOscillate(){ return oscillate;}


	/* Function: evaluate additional weight setup in the branch and return in floating precision */
        double GetMonteCarloWeight(){
            if(branch_monte_carlo_weight_formula){ 
                branch_monte_carlo_weight_formula->GetNdata();
                return (double)branch_monte_carlo_weight_formula->EvalInstance();
            }
            return 1.0;
        }
    };


    struct BranchVariable_d: public BranchVariable{
        BranchVariable_d(std::string n, std::string t, std::string a) : BranchVariable(n,t,a) {value_d=0;true_value_d=0; true_L_d = 0;};
        BranchVariable_d(std::string n, std::string t, std::string a_hist, std::string a_syst, bool cv) : BranchVariable(n,t,a_hist, a_syst, cv) {value_d=0;true_value_d=0; true_L_d = 0;};
        void* GetValue(){ 
            if(branch_formula == NULL) return &value_d;
            else{   
                branch_formula->GetNdata();
                value_d = (double)branch_formula->EvalInstance();
                return &value_d;
            }
        }

        void* GetTrueL(){
            if(branch_true_L_formula == NULL) return &true_L_d;
            else{
                branch_true_L_formula->GetNdata();
                true_L_d = (double)branch_true_L_formula->EvalInstance();
                return &true_L_d;
            }
        }

        void* GetTrueValue(){ 
            if(branch_true_value_formula == NULL) return &true_value_d;
            else{
                branch_true_value_formula->GetNdata();
                true_value_d = (double)branch_true_value_formula->EvalInstance();
                return &true_value_d;
            }
        }
    };

    struct BranchVariable_f: public BranchVariable{
        BranchVariable_f(std::string n, std::string t, std::string a) : BranchVariable(n,t,a) {value_f=0;true_value_f=0; true_L_f = 0;};
        void* GetValue(){ 
            if(branch_formula == NULL) return &value_f;
            else{
                branch_formula->GetNdata();
                value_f = (float)branch_formula->EvalInstance();
                return &value_f;
            }
        }

        void* GetTrueValue(){ 
            if(branch_true_value_formula == NULL) return &true_value_f;
            else{
                branch_true_value_formula->GetNdata();
                true_value_f = (float)branch_true_value_formula->EvalInstance();
                return &true_value_f;
            }
        }

        void* GetTrueL(){
            if(branch_true_L_formula == NULL) return &true_L_f;
            else{
                branch_true_L_formula->GetNdata();
                true_L_f = (float)branch_true_L_formula->EvalInstance();
                return &true_L_f;
            }
        }
    };


    class PROconfig {
        private:

            //indicator of whether each channel/detector/subchannel is used
            std::vector<bool> m_mode_bool;
            std::vector<bool> m_detector_bool;
            std::vector<bool> m_channel_bool;
            std::vector<std::vector<bool>>  m_subchannel_bool;


            //map from subchannel name/index to global index and channel index
            std::unordered_map<std::string, int> m_map_fullname_subchannel_index;
            std::unordered_map<int, long int> m_map_subchannel_index_to_global_index_start;
            std::unordered_map<int, int> m_map_subchannel_index_to_channel_index;


            //---- PRIVATE FUNCTION ------


            /* Function: remove any mode/detector/channel/subchannels in the configuration xml that are not used from consideration
            */
            void remove_unused_channel();


            /* Function: ignore any file that is associated with unused channels 
            */
            void remove_unused_files();


            /* Function: fill in mapping between subchannel name/index to global indices */
            void generate_index_map();


        public:


            PROconfig() {}; //always have an empty?
            PROconfig(const std::string &xml);

            int LoadFromXML(const std::string & filename);

            std::string m_xmlname;	
            double m_plot_pot;
            std::vector<std::string> m_fullnames;

            int m_num_detectors;
            int m_num_channels;
            int m_num_modes;
            //vectors of length num_channels
            std::vector<int> m_num_subchannels; 

            std::vector<int> m_channel_num_bins;
            std::vector<std::vector<double> > m_channel_bin_edges;
            std::vector<std::vector<double> > m_channel_bin_widths;

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

            int m_num_bins_detector_block_collapsed;
            int m_num_bins_mode_block_collapsed;
            int m_num_bins_total_collapsed;

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

            /* Function: Calculate how big each mode block and decector block are, for any given number of channels/subchannels, before and after the collapse
             * Note: only consider mode/detector/channel/subchannels that are actually used 
             */
            void CalcTotalBins();


            /* Function: given subchannel full name, return global subchannel index 
             * Note: index start from 0, not 1
             */
            int GetSubchannelIndex(const std::string& fullname) const;

            /* Function: given subchannel global index, return corresponding channel index 
             * Note: index start from 0, not 1
             */
            int GetChannelIndex(int subchannel_index) const;

            /* Function: given subchannel global index, return corresponding global bin start
             * Note: global bin index start from 0, not 1
             */
            long int GetGlobalBinStart(int subchannel_index) const;


            /* Function: given channel index, return list of bin edges for this channel */
            const std::vector<double>& GetChannelBinEdges(int channel_index) const;
    };




}
#endif
