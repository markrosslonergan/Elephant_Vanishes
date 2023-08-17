#include "PROcreate.h"
#include "Eigen/src/Core/Matrix.h"
#include "TTree.h"
#include "TFile.h"
#include <algorithm>

namespace PROfit {

    void SystStruct::CleanSpecs(){
	if(p_cv)  p_cv.reset(nullptr);
	p_multi_spec.clear();
        return;
    }

    void SystStruct::CreateSpecs(int num_bins){
        this->CleanSpecs();
        log<LOG_INFO>(L"%1% || Creating multi-universe spectrum with dimension (%2% x %3%)") % __func__ % n_univ % num_bins;
	
        p_cv = std::make_unique<PROspec>(num_bins);
	for(int i = 0; i != n_univ; ++i){
	    p_multi_spec.push_back(std::make_unique<PROspec>(num_bins));
  	}
        return;
    }


    const PROspec& SystStruct::CV() const{
	return *p_cv;
    }

    const PROspec& SystStruct::Variation(int universe) const{
   	return *(p_multi_spec.at(universe));
    }

    void SystStruct::FillCV(int global_bin, double event_weight){
	p_cv->Fill(global_bin, event_weight);
	return;
    }

    void SystStruct::FillUniverse(int universe, int global_bin, double event_weight){
	p_multi_spec.at(universe)->QuickFill(global_bin, event_weight);
	return;
    }

    void SystStruct::SanityCheck() const{
        if(mode == "minmax" && n_univ != 2){
            log<LOG_ERROR>(L"%1% || Systematic variation %2% is tagged as minmax mode, but has %3% universes (can only be 2)") % __func__ % systname.c_str() % n_univ;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        log<LOG_INFO>(L"%1% || Systematic variation %2% passed sanity check!!") % __func__ % systname.c_str();
        log<LOG_INFO>(L"%1% || Systematic variation %2% has %3% universes, and is in %4% mode with weight formula: %5%") % __func__ % systname.c_str() % n_univ % mode.c_str() % weight_formula.c_str();
        return;
    }

    void SystStruct::Print() const{
        log<LOG_INFO>(L"%1% || Printing %2%") % __func__ % systname.c_str();
        int i=0;
        for(auto &spec: p_multi_spec){
             log<LOG_INFO>(L"%1% || On Universe %2% for knob %3% ") % __func__ % i % knobval[i];
             spec->Print();
             i++;
        }

        return;
    }

    void SystStruct::FillSpline() {
      std::vector<PROspec> ratios;
      ratios.reserve(p_multi_spec.size());
      for(size_t i = 0; i < p_multi_spec.size(); ++i) {
        ratios.push_back(*p_multi_spec[i] / *p_cv);
        if(knobval[i] == -1) ratios.push_back(*p_cv / *p_cv);
      }
      spline_coeffs.reserve(p_cv->GetNbins());
      for(int i = 0; i < p_cv->GetNbins(); ++i) {
        std::vector<std::array<double, 4>> spline;
        spline.reserve(knobval.size());

        // This is cubic interpolation. For each adjacent set of four points we
        // determine coefficients for a cubic which will be the curve between the
        // center two. We constrain the function to match the two center points
        // and to have the right mean gradient at them. This causes this patch to
        // match smoothly with the next one along. The resulting function is
        // continuous and first and second differentiable. At the ends of the
        // range we fit a quadratic instead with only one constraint on the
        // slope. The coordinate conventions are that point y1 sits at x=0 and y2
        // at x=1. The matrices are simply the inverses of writing out the
        // constraints expressed above.

        const double y1 = ratios[0].GetBinContent(i);
        const double y2 = ratios[1].GetBinContent(i);
        const double y3 = ratios[2].GetBinContent(i);
        const Eigen::Vector3d v{y1, y2, (y3-y1)/2};
        const Eigen::Matrix3d m{{ 1, -1,  1},
                                {-2,  2, -1},
                                { 1,  0,  0}};
        const Eigen::Vector3d res = m * v;
        spline.push_back({res(2), res(1), res(0), 0});

        for(unsigned int shiftIdx = 1; shiftIdx < ratios.size()-2; ++shiftIdx){
          const double y0 = ratios[shiftIdx-1].GetBinContent(i);
          const double y1 = ratios[shiftIdx  ].GetBinContent(i);
          const double y2 = ratios[shiftIdx+1].GetBinContent(i);
          const double y3 = ratios[shiftIdx+2].GetBinContent(i);
          const Eigen::Vector4d v{y1, y2, (y2-y0)/2, (y3-y1)/2};
          const Eigen::Matrix4d m{{ 2, -2,  1,  1},
                                  {-3,  3, -2, -1},
                                  { 0,  0,  1,  0},
                                  { 1,  0,  0,  0}};
          const Eigen::Vector4d res = m * v;
          spline.push_back({res(3), res(2), res(1), res(0)});
        }

        const double y4 = ratios[ratios.size() - 3].GetBinContent(i);
        const double y5 = ratios[ratios.size() - 2].GetBinContent(i);
        const double y6 = ratios[ratios.size() - 1].GetBinContent(i);
        const Eigen::Vector3d vp{y5, y6, (y6-y4)/2};
        const Eigen::Matrix3d mp{{-1,  1, -1},
                                 { 0,  0,  1},
                                 { 1,  0,  0}};
        const Eigen::Vector3d resp = mp * vp;
        spline.push_back({resp(2), resp(1), resp(0), 0});
        
        spline_coeffs.push_back(spline);
      }
    }

    double SystStruct::GetSplineShift(int bin, double shift) {
      if(bin < 0 || bin >= p_cv->GetNbins()) return -1;
      int shiftBin = (shift < knobval[0]) ? 0 : (int)(shift - knobval[0]);
      if(shiftBin > spline_coeffs[0].size() - 1) shiftBin = spline_coeffs[0].size() - 1;
      // We should use the line below if we switch to c++17
      // const long shiftBin = std::clamp((long)(shift - knobval[0]), 0, spline_coeffs[0].size() - 1);
      std::array<double, 4> coeffs = spline_coeffs[bin][shiftBin];
      if(shiftBin == spline_coeffs.size() - 1 || knobval[shiftBin] > 1) shift -= knobval[shiftBin - 1];
      else if(knobval[shiftBin] < 1) shift -= knobval[shiftBin];
      return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
    }

    PROspec SystStruct::GetSplineShiftedSpectrum(double shift) {
      PROspec ret(p_cv->GetNbins());
      for(int i = 0; i < p_cv->GetNbins(); ++i)
        ret.Fill(i, GetSplineShift(i, shift) * p_cv->GetBinContent(i));
      return ret;
    }


int PROcess_SBNfit(const PROconfig &inconfig, std::vector<SystStruct>& syst_vector){
 
    log<LOG_DEBUG>(L"%1% || Starting to construct CovarianceMatrixGeneration in EventWeight Mode  ") % __func__ ;

    int num_files = inconfig.m_num_mcgen_files;

    log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

    std::vector<long int> nentries(num_files,0);
    std::vector<std::unique_ptr<TFile>> files(num_files);
    std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
    std::vector<std::vector<std::map<std::string, std::vector<eweight_type>>* >> f_event_weights(num_files);
    std::map<std::string, int> map_systematic_num_universe;


    //open files, and link trees and branches
    int good_event = 0;
    for(int fid=0; fid < num_files; ++fid) {
        const auto& fn = inconfig.m_mcgen_file_name.at(fid);

        files[fid] = std::make_unique<TFile>(fn.c_str(),"read");
        trees[fid] = (TTree*)(files[fid]->Get(inconfig.m_mcgen_tree_name.at(fid).c_str()));
        nentries[fid]= (long int)trees.at(fid)->GetEntries();

	if(files[fid]->IsOpen()){
    	    log<LOG_INFO>(L"%1% || Root file succesfully opened: %2%") % __func__  % fn.c_str();
	}else{
    	    log<LOG_ERROR>(L"%1% || Fail to open root file: %2%") % __func__  % fn.c_str();
	    exit(EXIT_FAILURE);
	}
    	log<LOG_INFO>(L"%1% || Total Entries: %2%") % __func__ %  nentries[fid];


    	//first, grab friend trees
        auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
        if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {

            auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
            if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
    	    	log<LOG_ERROR>(L"%1% || Friend TTree provided but no friend file??") % __func__;
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

            for(size_t k=0; k < mcgen_file_friend_treename_iter->second.size(); k++){

                std::string treefriendname = mcgen_file_friend_treename_iter->second.at(k);
                std::string treefriendfile = mcgen_file_friend_iter->second.at(k);
                trees[fid]->AddFriend(treefriendname.c_str(),treefriendfile.c_str());
            }
        }

        // grab branches 
        int num_branch = inconfig.m_branch_variables[fid].size();
	f_event_weights[fid].resize(num_branch);
        for(int ib = 0; ib != num_branch; ++ib) {

            std::shared_ptr<BranchVariable> branch_variable = inconfig.m_branch_variables[fid][ib];

            //quick check that this branch associated subchannel is in the known chanels;
            int is_valid_subchannel = 0;
            for(const auto &name: inconfig.m_fullnames){
                if(branch_variable->associated_hist==name){
                    log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name.c_str();
                    ++is_valid_subchannel;
                }
            }
            if(is_valid_subchannel==0){
    	    	log<LOG_ERROR>(L"%1% || This branch did not match one defined in the .xml : %2%") % __func__ % inconfig.m_xmlname.c_str();
    	    	log<LOG_ERROR>(L"%1% || There is probably a typo somehwhere in xml!") % __func__;
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);

            }else if(is_valid_subchannel>1){
    	    	log<LOG_ERROR>(L"%1% || This branch matched more than 1 subchannel!: %2%") % __func__ %  branch_variable->associated_hist.c_str();
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

            branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), trees[fid]);
    	    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();


	    //grab monte carlo weight
	    if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),trees[fid]);
    	    	log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
            }


	    //grab eventweight branch
    	    log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2%") % __func__ %  inconfig.m_mcgen_eventweight_branch_names[fid][ib].c_str();
	    trees[fid]->SetBranchAddress(inconfig.m_mcgen_eventweight_branch_names[fid][ib].c_str(), &(f_event_weights[fid][ib]));

	    if(!f_event_weights[fid][ib]){
            	log<LOG_ERROR>(L"%1% || Could not read eventweight branch for file=%2%") % __func__ % fid ;
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
	    }
        } //end of branch loop


	//calculate how many "universes" each systematoc has.
    	log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
        trees.at(fid)->GetEntry(good_event);
	for(int ib = 0; ib != num_branch; ++ib) {
            const auto& branch_variable = inconfig.m_branch_variables[fid][ib];
	    auto& f_weight = f_event_weights[fid][ib];

            for(const auto& it : *f_weight){
    	    	log<LOG_DEBUG>(L"%1% || On systematic: %2%") % __func__ % it.first.c_str();

            	if(inconfig.m_mcgen_variation_allowlist.count(it.first)==0){
    	    	    log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as its not in the AllowList!!") % __func__ % it.first.c_str();
                    continue;
                }

            	if(inconfig.m_mcgen_variation_denylist.count(it.first)>0){
    	    	    log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as it is in the DenyList!!") % __func__ % it.first.c_str();
                    continue;
            	}

            	log<LOG_INFO>(L"%1% || %2% has %3% montecarlo variations in branch %4%") % __func__ % it.first.c_str() % it.second.size() % branch_variable->associated_hist.c_str();

		map_systematic_num_universe[it.first] = std::max((int)map_systematic_num_universe[it.first], (int)it.second.size());
	    }
        }
    } // end fid

    size_t total_num_systematics = map_systematic_num_universe.size();
    log<LOG_INFO>(L"%1% || Found %2% unique variations") % __func__ % total_num_systematics;
    for(auto& sys_pair : map_systematic_num_universe){
    	log<LOG_DEBUG>(L"%1% || Variation: %2% --> %3% universes") % __func__ % sys_pair.first.c_str() % sys_pair.second;
    }

    //constuct object for each systematic variation, and grab weight maps
    log<LOG_INFO>(L"%1% || Now start to grab related weightmaps") % __func__;
    for(auto& sys_pair : map_systematic_num_universe){

	const std::string& sys_name = sys_pair.first;
        syst_vector.emplace_back(sys_name, sys_pair.second);

            
	// Check to see if pattern is in this variation
	std::string sys_weight_formula = "1", sys_mode ="";

	for(size_t i = 0 ; i != inconfig.m_mcgen_weightmaps_patterns.size(); ++i){
            if (inconfig.m_mcgen_weightmaps_uses[i] && sys_name.find(inconfig.m_mcgen_weightmaps_patterns[i]) != std::string::npos) {
                sys_weight_formula = sys_weight_formula + "*(" + inconfig.m_mcgen_weightmaps_formulas[i]+")";
                sys_mode=inconfig.m_mcgen_weightmaps_mode[i];

		log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for patten %3%") % __func__ % sys_name.c_str() % inconfig.m_mcgen_weightmaps_patterns[i].c_str();
		log<LOG_INFO>(L"%1% || Corresponding weight is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_formulas[i].c_str();
		log<LOG_INFO>(L"%1% || Corresponding mode is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_mode[i].c_str();
            }
	}

	if(sys_weight_formula != "1" || sys_mode !=""){
	    syst_vector.back().SetWeightFormula(sys_weight_formula);
	    syst_vector.back().SetMode(sys_mode);
	}
    }


    //sanity check 
    for(const auto& s : syst_vector)
	s.SanityCheck();


    //create 2D multi-universe spec.
    for(auto& s : syst_vector){
 	s.CreateSpecs(inconfig.m_num_bins_total);	
    }


    time_t start_time = time(nullptr), time_stamp = time(nullptr);
    log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
    for(int fid=0; fid < num_files; ++fid) {
        const auto& fn = inconfig.m_mcgen_file_name.at(fid);
        long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
	log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;

	
	// set up systematic weight formula
	std::vector<double> sys_weight_value(total_num_systematics, 1.0);
    	std::vector<std::unique_ptr<TTreeFormula>> sys_weight_formula;
	for(const auto& s : syst_vector){
	    if(s.HasWeightFormula())
	    	sys_weight_formula.push_back(std::make_unique<TTreeFormula>(("weightMapsFormulas_"+std::to_string(fid)+"_"+ s.GetSysName()).c_str(), s.GetWeightFormula().c_str(),trees[fid]));
	    else
		sys_weight_formula.push_back(nullptr);
  	}
	log<LOG_DEBUG>(L"%1% || Finished setting up systematic weight formula") % __func__;


	// grab the subchannel index
	int num_branch = inconfig.m_branch_variables[fid].size();
	auto& branches = inconfig.m_branch_variables[fid];
	std::vector<int> subchannel_index(num_branch, 0); 
	log<LOG_DEBUG>(L"%1% || This file includes %2% branch/subchannels") % __func__ % num_branch;
        for(int ib = 0; ib != num_branch; ++ib) {

            const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
	    subchannel_index[ib] = inconfig.GetSubchannelIndex(subchannel_name);
	    log<LOG_DEBUG>(L"%1% || Subchannel: %2% maps to index: %3%") % __func__ % subchannel_name.c_str() % subchannel_index[ib];
	}


	// loop over all entries
        for(long int i=0; i < nevents; ++i) {
            if(i%1000==0){
	    	time_t time_passed = time(nullptr) - time_stamp;
		log<LOG_INFO>(L"%1% || File %2% -- uni : %3% / %4%  took %5% seconds") % __func__ % fid % i % nevents % time_passed;
	        time_stamp = time(nullptr);
	    }
	    trees[fid]->GetEntry(i);
            

	    //grab additional weight for systematics
	    for(size_t is = 0; is != total_num_systematics; ++is){
	    	if(syst_vector[is].HasWeightFormula()){
		    sys_weight_formula[is]->GetNdata();	
		    sys_weight_value[is] = sys_weight_formula[is]->EvalInstance();
		}
  	    }

	    //branch loop
            for(int ib = 0; ib != num_branch; ++ib) {
		process_sbnfit_event(inconfig, branches[ib], *f_event_weights[fid][ib], subchannel_index[ib], syst_vector, sys_weight_value);
	    } 

        } //end of entry loop

    } //end of file loop

    time_t time_took = time(nullptr) - start_time;
    log<LOG_INFO>(L"%1% || Finish reading files, it took %2% seconds..") % __func__ % time_took;
    log<LOG_INFO>(L"%1% || DONE") %__func__ ;

    return 0;
}


    int PROcess_CAFana(const PROconfig &inconfig, std::vector<SystStruct>& syst_vector){

        log<LOG_DEBUG>(L"%1% || Starting to construct CovarianceMatrixGeneration in EventWeight Mode  ") % __func__ ;


        std::vector<std::string> variations;
        std::vector<std::string> variations_tmp;

        int num_files = inconfig.m_num_mcgen_files;

        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<std::unique_ptr<TFile>> files(num_files);
        std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
        std::map<std::string, int> map_systematic_num_universe;

        //CAFANA related things
        std::vector<int> pset_indices_tmp;
        std::vector<int> pset_indices;
        std::vector<std::vector<float>> knobvals;
        std::vector<std::vector<float>> knob_index;
        std::vector<std::string> cafana_pset_names;

        int good_event = 0;

        std::vector<PROfit::CAFweightHelper> v_cafhelper(num_files);

        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);

            files[fid] = std::make_unique<TFile>(fn.c_str(),"read");
            trees[fid] = (TTree*)(files[fid]->Get(inconfig.m_mcgen_tree_name.at(fid).c_str()));
            TTree * globalTree = (TTree*)(files[fid]->Get("globalTree"));
            nentries[fid]= (long int)trees.at(fid)->GetEntries();

            if(files[fid]->IsOpen()){
                log<LOG_INFO>(L"%1% || Root file succesfully opened: %2%") % __func__  % fn.c_str();
            }else{
                log<LOG_ERROR>(L"%1% || Fail to open root file: %2%") % __func__  % fn.c_str();
                exit(EXIT_FAILURE);
            }
            log<LOG_INFO>(L"%1% || Total Entries: %2%") % __func__ %  nentries[fid];


            log<LOG_DEBUG>(L"%1% || On file %2% - %3% Getting SRglobal ") % __func__ % fid % inconfig.m_mcgen_file_name.at(fid).c_str()  ;
            caf::SRGlobal* global = NULL;
            globalTree->SetBranchAddress("global", &global);
            log<LOG_DEBUG>(L"%1% || On file %2% - %3% Getting first entry from global tree ") % __func__ % fid % fn.c_str()  ;
            globalTree->GetEntry(0);

            log<LOG_DEBUG>(L"%1% || On file %2% - %3% Grabbing Weights ") % __func__ % fid % fn.c_str()  ;
            for(unsigned int i = 0; i < global->wgts.size(); ++i) {
                const caf::SRWeightPSet& pset = global->wgts[i];
                pset_indices.push_back(i);
                cafana_pset_names.push_back(pset.name);
                map_systematic_num_universe[pset.name] = std::max(map_systematic_num_universe[pset.name], pset.nuniv);
                knob_index.push_back(pset.map.at(0).vals);
                knobvals.push_back(pset.map.at(0).vals);
                std::sort(knobvals.back().begin(), knobvals.back().end());
            }

            //Setup things for grabbing the CAFana weights
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt.univ..totarraysize", &(v_cafhelper[fid].i_wgt_univ_size));
            trees[fid]->SetBranchAddress("rec.mc.nu..length", &(v_cafhelper[fid].i_wgt_size));
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt..totarraysize",&(v_cafhelper[fid].i_wgt_totsize));
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt.univ", v_cafhelper[fid].v_wgt_univ);
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt.univ..idx", v_cafhelper[fid].v_wgt_univ_idx);
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt..idx",v_cafhelper[fid].v_wgt_idx);
            trees[fid]->SetBranchAddress("rec.mc.nu.wgt.univ..length",v_cafhelper[fid].v_wgt_univ_length);
            trees[fid]->SetBranchAddress("rec.mc.nu.index", v_cafhelper[fid].v_truth_index);

            // grab branches 
            int num_branch = inconfig.m_branch_variables[fid].size();
            for(int ib = 0; ib != num_branch; ++ib) {

                std::shared_ptr<BranchVariable> branch_variable = inconfig.m_branch_variables[fid][ib];
                //quick check that this branch associated subchannel is in the known chanels;
                int is_valid_subchannel = 0;
                for(const auto &name: inconfig.m_fullnames){
                    if(branch_variable->associated_hist==name){
                        log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name.c_str();
                        ++is_valid_subchannel;
                    }
                }
                if(is_valid_subchannel==0){
                    log<LOG_ERROR>(L"%1% || This branch did not match one defined in the .xml : %2%") % __func__ % inconfig.m_xmlname.c_str();
                    log<LOG_ERROR>(L"%1% || There is probably a typo somehwhere in xml!") % __func__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);

                }else if(is_valid_subchannel>1){
                    log<LOG_ERROR>(L"%1% || This branch matched more than 1 subchannel!: %2%") % __func__ %  branch_variable->associated_hist.c_str();
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }

                branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), trees[fid]);
                log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();


                //grab monte carlo weight
                if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                    branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),trees[fid]);
                    log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
                }

            } //end of branch loop


            //calculate how many "universes" each systematoc has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            trees.at(fid)->GetEntry(good_event);




            log<LOG_DEBUG>(L"%1% || On file %2% - %3% Starting on cafana pset loop to build SysVec ") % __func__ % fid % fn.c_str()  ;
            for(size_t v =0; v< cafana_pset_names.size(); v++){
                std::string varname = cafana_pset_names[v];

                log<LOG_DEBUG>(L"%1% || starting %2% ") % __func__ % varname.c_str()  ;

                if(inconfig.m_mcgen_variation_allowlist.size()> 0 ){
                    if(inconfig.m_mcgen_variation_allowlist.count(varname)==0){
                        log<LOG_INFO>(L"%1% || Skipping %2% as its not in allowlist ") % __func__ % varname.c_str();
                        continue;
                    }
                }

                if(inconfig.m_mcgen_variation_denylist.size()> 0 ){
                    if(inconfig.m_mcgen_variation_denylist.count(varname)>0){
                        log<LOG_INFO>(L"%1% || Skipping %2% as it in denylist ") % __func__ % varname.c_str();
                        continue;
                    }
                }

                //Variation Weight Maps Area
                int n_wei = inconfig.m_mcgen_weightmaps_patterns.size();
                std::string variation_mode = "multisim";
                if(varname.find("multisigma") != std::string::npos) variation_mode = "multisigma";
                std::string s_formula = "1";

                for(int i=0; i< n_wei; i++){
                    // Check to see if pattern is in this variation
                    if (varname.find(inconfig.m_mcgen_weightmaps_patterns[i]) != std::string::npos) {
                        log<LOG_DEBUG>(L"%1% || Variation %2% is a match for pattern %3%") % __func__ % varname.c_str() % inconfig.m_mcgen_weightmaps_patterns[i].c_str() ;

                        s_formula = s_formula + "*(" + inconfig.m_mcgen_weightmaps_formulas[i]+")";
                        log<LOG_DEBUG>(L"%1% || Weight is thus %2% and mode is %3% ") % __func__ % s_formula.c_str() % inconfig.m_mcgen_weightmaps_mode[i].c_str();
                    }
                }


                log<LOG_DEBUG>(L"%1% || %2% has %3% montecarlos in fie %4% ") % __func__ % varname.c_str() % map_systematic_num_universe[varname] % fid  ;

                //map_systematic_num_universe[varname] = std::max((int)map_systematic_num_universe[varname], (int)knobvals[v].size());

                //Some code to check if this varname is already in the syst_vector. If it is, check if things are the same, otherwise PANIC!
                log<LOG_DEBUG>(L"%1% || emplace syst_vector") % __func__  ;
                log<LOG_DEBUG>(L"%1% || varname: %2% , map: %3% , sform: %5% , varmode %4%, knobvals[v]size %6%, psetindex %7%  ") % __func__  % varname.c_str() % map_systematic_num_universe[varname] % variation_mode.c_str() % s_formula.c_str() % knobvals[v].size() % pset_indices[v] ;

                syst_vector.emplace_back(varname, map_systematic_num_universe[varname], variation_mode, s_formula, knobvals[v], knob_index[v], pset_indices[v]);
            }

        } // end fid

        log<LOG_INFO>(L"%1% || Found %2% unique variations") % __func__ % map_systematic_num_universe.size();
        for(auto& sys_pair : map_systematic_num_universe){
            log<LOG_DEBUG>(L"%1% || Variation: %2% --> %3% universes") % __func__ % sys_pair.first.c_str() % sys_pair.second;
        }


        //sanity check 
        for(const auto& s : syst_vector)
            s.SanityCheck();

        //create 2D multi-universe spec.
        for(auto& s : syst_vector){
            s.CreateSpecs(inconfig.m_num_bins_total);	
        }


        time_t start_time = time(nullptr);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;


            std::vector<std::unique_ptr<TTreeFormula>> sys_weight_formula;
            for(const auto& s : syst_vector){
                sys_weight_formula.push_back(std::make_unique<TTreeFormula>(("weightMapsFormulas_"+std::to_string(fid)+"_"+ s.GetSysName()).c_str(), s.GetWeightFormula().c_str(),trees[fid]));
            }
            log<LOG_DEBUG>(L"%1% || Finished setting up systematic weight formula") % __func__;

            // grab the subchannel index
            int num_branch = inconfig.m_branch_variables[fid].size();
            auto& branches = inconfig.m_branch_variables[fid];
            std::vector<int> subchannel_index(num_branch, 0); 
            log<LOG_DEBUG>(L"%1% || This file includes %2% branch/subchannels") % __func__ % num_branch;
            for(int ib = 0; ib != num_branch; ++ib) {
                const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
                subchannel_index[ib] = inconfig.GetSubchannelIndex(subchannel_name);
                log<LOG_DEBUG>(L"%1% || Subchannel: %2% maps to index: %3%") % __func__ % subchannel_name.c_str() % subchannel_index[ib];
            }

            for(long int i=0; i < nevents; ++i) {

                trees[fid]->GetEntry(i);
                if(i%1000==0)log<LOG_DEBUG>(L"%1% || ---- universe %2%/%3% ") % __func__  % files[fid]->GetName() % nevents ;

                for(int ib = 0; ib != num_branch; ++ib) {
                    double reco_value = *(static_cast<double*>(branches[ib]->GetValue()));
                    double additional_weight = branches[ib]->GetMonteCarloWeight();
                    //additional_weight *= pot_scale[fid]; POT NOT YET FIX
                    int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index[ib]);
                    if(global_bin < 0 )  //out or range
                        continue;
                    PROcess_CAFana_Event(inconfig, sys_weight_formula, syst_vector, v_cafhelper[fid], additional_weight, global_bin);

                }//end of branch 

            } //end of entry loop

        } //end of file loop


        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Finish reading files, it took %2% seconds..") % __func__ % time_took;

        log<LOG_INFO>(L"%1% || Some useful info:") %__func__ ;
        //OK some printing of syst_vector
        for(auto &syst: syst_vector){
            syst.Print();
        }

        log<LOG_INFO>(L"%1% || DONE") %__func__ ;
        return 0;
    }

    int PROcess_CAFana_Event(const PROconfig &inconfig, std::vector<std::unique_ptr<TTreeFormula>> & formulas, std::vector<SystStruct> &syst_vector, CAFweightHelper &caf_helper, double add_weight, int global_bin){

        int is = 0;
        for(SystStruct & syst : syst_vector){
            syst.FillCV(global_bin, add_weight);
            formulas[is]->GetNdata();
            double sys_weight_value =formulas[is]->EvalInstance();

            if(std::isinf(sys_weight_value) || sys_weight_value != sys_weight_value){
                log<LOG_ERROR>(L"%1% || Input values to histogram is NAN or inf %2% !") % __func__  % sys_weight_value ;
                throw std::runtime_error("NAN or INF in put string");
            }

            int nuniv = syst.GetNUniverse();
            for(int u =0; u<nuniv; u++){
                int i = 0;
                if(syst.mode == "multisigma") {
                  for(; i < nuniv; ++i) {
                    if(syst.knob_index[i] == syst.knobval[u]) break;
                  }
                } else {
                  i = u;
                }
                float this_weight = caf_helper.GetUniverseWeight(syst.index, i);
		syst.FillUniverse(u, global_bin, add_weight*this_weight);
                //std::cout<<"WEI "<<is<<" "<<u<<global_bin<<" "<<add_weight<<" "<<this_weight<<std::endl;
            }
            is++;
        }

        return 0;
    }


PROspec CreatePROspecCV(const PROconfig& inconfig){

    double spec_pot = inconfig.m_plot_pot;
    log<LOG_INFO>(L"%1% || Start generating central value spectrum") % __func__ ;
    log<LOG_INFO>(L"%1% || Spectrum will be generated with %2% POT") % __func__ % spec_pot;

    int num_files = inconfig.m_num_mcgen_files;
    log<LOG_DEBUG>(L"%1% || Starting to read file and build spectrum!") % __func__ ;
    log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;


    std::vector<long int> nentries(num_files,0);
    std::vector<double> pot_scale(num_files, 1.0);
    std::vector<std::unique_ptr<TFile>> files(num_files);
    std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(


    for(int fid=0; fid < num_files; ++fid) {
        const auto& fn = inconfig.m_mcgen_file_name.at(fid);

        files[fid] = std::make_unique<TFile>(fn.c_str(),"read");
        trees[fid] = (TTree*)(files[fid]->Get(inconfig.m_mcgen_tree_name.at(fid).c_str()));
        nentries[fid]= (long int)trees.at(fid)->GetEntries();

	if(files[fid]->IsOpen()){
    	    log<LOG_INFO>(L"%1% || Root file succesfully opened: %2%") % __func__  % fn.c_str();
	}else{
    	    log<LOG_ERROR>(L"%1% || Fail to open root file: %2%") % __func__  % fn.c_str();
	    exit(EXIT_FAILURE);
	}
    	log<LOG_INFO>(L"%1% || Total Entries: %2%") % __func__ %  nentries[fid];

	//calculate POT scale factor
        if(inconfig.m_mcgen_pot.at(fid) != -1){
            pot_scale[fid] = spec_pot/inconfig.m_mcgen_pot.at(fid);
        }
	pot_scale[fid] *= inconfig.m_mcgen_scale[fid];
    	log<LOG_INFO>(L"%1% || File POT: %2%, additional scale: %3%") % __func__ %  inconfig.m_mcgen_pot.at(fid) % inconfig.m_mcgen_scale[fid];
    	log<LOG_INFO>(L"%1% || POT scale factor: %2%") % __func__ %  pot_scale[fid];

    	//first, grab friend trees
        auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
        if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {

            auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
            if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
    	    	log<LOG_ERROR>(L"%1% || Friend TTree provided but no friend file??") % __func__;
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

            for(size_t k=0; k < mcgen_file_friend_treename_iter->second.size(); k++){

                std::string treefriendname = mcgen_file_friend_treename_iter->second.at(k);
                std::string treefriendfile = mcgen_file_friend_iter->second.at(k);
                trees[fid]->AddFriend(treefriendname.c_str(),treefriendfile.c_str());
            }
        }

        // grab branches 
        int num_branch = inconfig.m_branch_variables[fid].size();
        for(int ib = 0; ib != num_branch; ++ib) {

            std::shared_ptr<BranchVariable> branch_variable = inconfig.m_branch_variables[fid][ib];

            //quick check that this branch associated subchannel is in the known chanels;
            int is_valid_subchannel = 0;
            for(const auto &name: inconfig.m_fullnames){
                if(branch_variable->associated_hist==name){
                    log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name.c_str();
                    ++is_valid_subchannel;
                }
            }
            if(is_valid_subchannel==0){
    	    	log<LOG_ERROR>(L"%1% || This branch did not match one defined in the .xml : %2%") % __func__ % inconfig.m_xmlname.c_str();
    	    	log<LOG_ERROR>(L"%1% || There is probably a typo somehwhere in xml!") % __func__;
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);

            }else if(is_valid_subchannel>1){
    	    	log<LOG_ERROR>(L"%1% || This branch matched more than 1 subchannel!: %2%") % __func__ %  branch_variable->associated_hist.c_str();
		log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }

            branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), trees[fid]);
    	    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();


	    //grab monte carlo weight
	    if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),trees[fid]);
    	    	log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
            }

        } //end of branch loop
    } // end fid


    time_t start_time = time(nullptr);
    PROspec spec(inconfig.m_num_bins_total);


    log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
    for(int fid=0; fid < num_files; ++fid) {
        const auto& fn = inconfig.m_mcgen_file_name.at(fid);
        long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
	log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;


	// grab the subchannel index
	int num_branch = inconfig.m_branch_variables[fid].size();
	auto& branches = inconfig.m_branch_variables[fid];
	std::vector<int> subchannel_index(num_branch, 0); 
	log<LOG_INFO>(L"%1% || This file includes %2% branch/subchannels") % __func__ % num_branch;
        for(int ib = 0; ib != num_branch; ++ib) {

            const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
	    subchannel_index[ib] = inconfig.GetSubchannelIndex(subchannel_name);
	    log<LOG_DEBUG>(L"%1% || Subchannel: %2% maps to index: %3%") % __func__ % subchannel_name.c_str() % subchannel_index[ib];
	}


	// loop over all entries
        for(long int i=0; i < nevents; ++i) {

            if(i%1000==0)	log<LOG_INFO>(L"%1% || -- uni : %2% / %3%") % __func__ % i % nevents;
	    trees[fid]->GetEntry(i);
            
	    //branch loop
            for(int ib = 0; ib != num_branch; ++ib) {

		//guanqun: why have different types for branch_variables 
		double reco_value = *(static_cast<double*>(branches[ib]->GetValue()));

		double additional_weight = branches[ib]->GetMonteCarloWeight();
		additional_weight *= pot_scale[fid];
		if(additional_weight == 0) //skip on event failing cuts
		    continue;

		//find bins
		int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index[ib]);
		if(global_bin < 0 )  //out of range
		    continue;

                if(i%100==0)	
		    log<LOG_DEBUG>(L"%1% || Subchannel %2% -- Reco variable value: %3%, MC event weight: %4%, correponds to global bin: %5%") % __func__ %  subchannel_index[ib] % reco_value % additional_weight % global_bin;

		spec.Fill(global_bin, additional_weight);

	    }  //end of branch loop

        } //end of entry loop

    } //end of file loop

    time_t time_took = time(nullptr) - start_time;
    log<LOG_INFO>(L"%1% || Generating central value spectrum took %2% seconds..") % __func__ % time_took;
    log<LOG_INFO>(L"%1% || DONE") %__func__ ;

    return spec;
}

void process_sbnfit_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>>& eventweight_map, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<double>& syst_additional_weight){

    int total_num_sys = syst_vector.size(); 
    double reco_value = *(static_cast<double*>(branch->GetValue()));
    double mc_weight = branch->GetMonteCarloWeight(); 
    int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index);
    if(global_bin < 0 )  //out of range
        return;

    for(int i = 0; i != total_num_sys; ++i){
	SystStruct& syst_obj = syst_vector[i];
	double additional_weight = syst_additional_weight.at(i);

	syst_obj.FillCV(global_bin, mc_weight);

	auto map_iter = eventweight_map.find(syst_obj.GetSysName());
	int map_variation_size = (map_iter == eventweight_map.end()) ? 0 : map_iter->second.size();
	int iuni = 0;
	for(; iuni != std::min(map_variation_size, syst_obj.GetNUniverse()); ++iuni){
	    syst_obj.FillUniverse(iuni, global_bin, mc_weight * additional_weight * static_cast<double>(map_iter->second.at(iuni)));
	}

	while(iuni != syst_obj.GetNUniverse()){
	    //syst_obj.FillUniverse(iuni, global_bin, mc_weight);
	    syst_obj.FillUniverse(iuni, global_bin, mc_weight * additional_weight);
	    ++iuni;
 	}
    }

    return;
}


Eigen::MatrixXd SystStruct::GenerateCovarMatrix(const SystStruct& sys_obj){
    int n_universe = sys_obj.GetNUniverse(); 
    std::string sys_name = sys_obj.GetSysName();
    
    const PROspec& cv_spec = sys_obj.CV();
    int nbins = cv_spec.GetNbins();
    log<LOG_INFO>(L"%1% || Generating covariance matrix.. size: %2% x %3%") % __func__ % nbins % nbins;

    //build full covariance matrix 
    Eigen::MatrixXd full_covar_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
    for(int i = 0; i != n_universe; ++i){
	PROspec spec_diff  = cv_spec - sys_obj.Variation(i);
	full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<double>(n_universe);
    }
 
    //build fractional covariance matrix 
    //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
    Eigen::MatrixXd cv_spec_matrix =  Eigen::MatrixXd::Identity(nbins, nbins);
    for(int i =0; i != nbins; ++i)
	cv_spec_matrix(i, i) = 1.0/cv_spec.GetBinContent(i);

    //second, get fractioal covar
    Eigen::MatrixXd frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;

    return frac_covar_matrix;
}

Eigen::MatrixXd SystStruct::GenerateCovarMatrix() const{
    
    int nbins = p_cv->GetNbins();
    log<LOG_INFO>(L"%1% || Generating covariance matrix.. size: %2% x %3%") % __func__ % nbins % nbins;

    //build full covariance matrix 
    Eigen::MatrixXd full_covar_matrix = Eigen::MatrixXd::Zero(nbins, nbins);
    for(int i = 0; i != n_univ; ++i){
	PROspec spec_diff  = *p_cv - *(p_multi_spec.at(i));
	full_covar_matrix += (spec_diff.Spec() * spec_diff.Spec().transpose() ) / static_cast<double>(n_univ);
    }
 
    //build fractional covariance matrix 
    //first, get the matrix with diagonal being reciprocal of CV spectrum prdiction
    Eigen::MatrixXd cv_spec_matrix =  Eigen::MatrixXd::Identity(nbins, nbins);
    for(int i =0; i != nbins; ++i)
	cv_spec_matrix(i, i) = 1.0/p_cv->GetBinContent(i);

    //second, get fractioal covar
    Eigen::MatrixXd frac_covar_matrix = cv_spec_matrix * full_covar_matrix * cv_spec_matrix;

    return frac_covar_matrix;
}

bool SystStruct::isPositiveSemiDefinite(const Eigen::MatrixXd& in_matrix){

    //first, check if it's symmetric 
    if(!in_matrix.isApprox(in_matrix.transpose(), Eigen::NumTraits<double>::dummy_precision())){
	log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % Eigen::NumTraits<double>::dummy_precision();
	return false;
    }

    //second, check if it's positive semi-definite;
    Eigen::LDLT<Eigen::MatrixXd> llt(in_matrix);
    if((llt.info() == Eigen::NumericalIssue ) || (!llt.isPositive()) )
	return false;

    return true;

}

bool SystStruct::isPositiveSemiDefinite_WithTolerance(const Eigen::MatrixXd& in_matrix, double tolerance ){

    //first, check if it's symmetric 
    if(!in_matrix.isApprox(in_matrix.transpose(), Eigen::NumTraits<double>::dummy_precision())){
	log<LOG_ERROR>(L"%1% || Covariance matrix is not symmetric, with tolerance of %2%") % __func__ % Eigen::NumTraits<double>::dummy_precision();
	return false;
    }
   

    //second, check if it's positive semi-definite;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(in_matrix);
    if(eigensolver.info() != Eigen::Success){
	log<LOG_ERROR>(L"%1% || Failing to get eigenvalues..") % __func__ ;
	return false;
    }
   
    Eigen::VectorXd eigenvals = eigensolver.eigenvalues();
    for(auto val : eigenvals ){
	if(val < 0 && fabs(val) > tolerance){
	   log<LOG_ERROR>(L"%1% || Found negative eigenvalues beyond tolerance (%2%): %3%") % __func__ % tolerance % val;
	   return false;
	}
    }
    return true;

}
}//namespace

