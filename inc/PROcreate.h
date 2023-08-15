#ifndef PROCREATE_H_
#define PROCREATE_H_

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>
#include <ctime>
#include <cmath>

// EIGEN
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

//PROfit
#include "PROlog.h"
#include "PROconfig.h"
#include "PROspec.h"
#include "PROtocall.h"

//CAFana
#include "sbnanaobj/StandardRecord/SRGlobal.h"
#include "sbnanaobj/StandardRecord/SRWeightPSet.h"

namespace PROfit{

    struct SystStruct {

        //members
        std::string systname;
        int n_univ;
        std::string mode;
        std::string weight_formula;
        std::vector<float> knobval;
        std::vector<int> knob_index;
        int index;
        std::vector<std::vector<std::array<double, 4>>> spline_coeffs;

        //std::vector<PROspec> m_multi_spec;

	// pointer to cv spectrum and multi-universe spectrum from systematic variation
	std::unique_ptr<PROspec> p_cv;	
        std::vector<std::unique_ptr<PROspec>> p_multi_spec;

        // functions 
        SystStruct(const std::string& in_systname, const int in_n_univ): SystStruct(in_systname, in_n_univ, "multisim", "1",{},0){}
        SystStruct(const std::string& in_systname, const int in_n_univ, const std::string& in_mode, const std::string& in_formula, const std::vector<float>& in_knobval, std::vector<int>& in_knob_index, const int in_index): systname(in_systname), n_univ(in_n_univ), mode(in_mode), weight_formula(in_formula), knobval(in_knobval), knob_index(in_knob_index), index(in_index){}


        inline
            void SetMode(const std::string& in_mode){mode = in_mode; return;}

        inline
            void SetWeightFormula(const std::string& in_formula){weight_formula = in_formula; return;}

        inline
            int GetNUniverse() const {return n_univ;}

        inline 
            const std::string& GetSysName() const {return systname;}

	inline 
	    bool HasWeightFormula() const {return weight_formula == "1";}

        inline 
            const std::string& GetWeightFormula() const {return weight_formula;}

        std::vector<std::vector<eweight_type>> GetCovVec();
        std::vector<eweight_type> GetKnobs(int index, std::string variation);


        /* Function: check if num of universes of this systematics matches with its type 
 	 * Note: multisim mode can have many universes, while minmax mode can only have 2
 	 */
        void SanityCheck() const;

	/* Function: clean up all the member spectra (but ONLY spectra) */
        void CleanSpecs();

        void Print() const;

	/* Function: create EMPTY spectra with given length 
 	 */ 
        void CreateSpecs(long int num_bins);

	/* Function: given global bin index, and event weight, fill the central value spectrum */
	void FillCV(long int global_bin, double event_weight);

	/* Function: given global bin index, and event weight, fill the spectrum of given universe */
	void FillUniverse(int universe, long int global_bin, double event_weight);

    /* Function: Fill spline_coeffs assuming p_cv and p_multi_spec have been filled */
    void FillSpline();

    /* Function: Get weight for bin for a given shift using spline */
    double GetSplineShift(long bin, double shift);

    /* Function: Get cv spectrum shifted using spline */
    PROspec GetSplineShiftedSpectrum(double shift);
    };


    struct CAFweightHelper{
        int i_wgt_univ_size ; //rec.mc.nu.wgt.univ..totarraysize
        int i_wgt_size ; //rec.slc..length
        int i_wgt_totsize ; //rec.mc.nu.wgt..totalarraysize

        float v_wgt_univ[30000];
        int v_wgt_univ_idx[30000];
        int v_wgt_idx[2000];
        int v_wgt_univ_length[2000];
        int v_truth_index[100] ;

        CAFweightHelper(){
            i_wgt_univ_size=0;
            i_wgt_size =0;
            i_wgt_totsize=0;
        };

        float GetUniverseWeight(int which_index , int which_uni){
            for(int s = 0; s<i_wgt_size;s++){
                if(v_truth_index[s]==0){
                    return v_wgt_univ[v_wgt_univ_idx[v_wgt_idx[s] + which_index] + which_uni];
                }
            }

            return 0;
        };

    };


    /* Function: given config, read files in the xml, and grab all systematic variations 
     * TODO: not finished yet
     */
    int PROcess_SBNfit(const PROconfig &inconfig);
    int PROcess_CAFana(const PROconfig &inconfig);

    int PROcess_CAFana_Event(const PROconfig &inconfig, std::vector<std::unique_ptr<TTreeFormula>> & formulas, std::vector<SystStruct> &syst_vector, CAFweightHelper &caf_helper, double add_weight, long int global_bin);

    /* Function: given configuration, generate spectrum at central value. 
     * Note: assume the input config has SBNfit-style files, TODO: check if compatible with CAF-style
     */
    PROspec CreatePROspecCV(const PROconfig& configin);



    /* Function: assume currently reading one entry of a file, update systematic variation spectrum 
     * Note: designed to be called internally by PROcess_SBNfit() function
     *
     * Arguments: 
     * 		branch: pointer to branch variable, each corresponding to one subchannel 
     * 		eventweight_map: a map between systematic string to list of variation weights
     * 		subchannel_index: index associated with current branch/subchannel
     *		syst_vector: list of SystStruct TO BE UPDATED, each stores all variation spectra of one systematic
     *		syst_additional_weight: additional weight applied to systematic variation
     */
    void process_sbnfit_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>>& eventweight_map, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<double>& syst_additional_weight);


    
};
#endif
