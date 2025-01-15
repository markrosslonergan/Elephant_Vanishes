#include "PROcreate.h"
#include "Eigen/src/Core/Matrix.h"
#include "PROlog.h"
#include "PROpeller.h"
#include "PROtocall.h"
#include "TTree.h"
#include "TFile.h"
#include "TFriendElement.h"
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
            if (inconfig.m_mcgen_numfriends[fid]>0){
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

                for (const std::string &name: branch_variable->names) {
                    branch_variable->branch_formulas.push_back(std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), name.c_str(), trees[fid]));
                    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  name.c_str();
                }

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


            //calculate how many "universes" each systematic has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            trees.at(fid)->GetEntry(good_event);
            for(int ib = 0; ib != num_branch; ++ib) {
                const auto& branch_variable = inconfig.m_branch_variables[fid][ib];
                auto& f_weight = f_event_weights[fid][ib];

                for(const auto& it : *f_weight){
                    log<LOG_DEBUG>(L"%1% || On systematic: %2%") % __func__ % it.first.c_str();
                    int ncount = std::count(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), it.first);

                    if(ncount==0){
                        log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as its not in the AllowList!!") % __func__ % it.first.c_str();
                        continue;
                    }
                    ncount = std::count(inconfig.m_mcgen_variation_denylist.begin(), inconfig.m_mcgen_variation_denylist.end(), it.first);
                    if(ncount>0){
                        log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as it is in the DenyList!!") % __func__ % it.first.c_str();
                        continue;
                    }

                    log<LOG_INFO>(L"%1% || %2% has %3% montecarlo variations in branch %4%") % __func__ % it.first.c_str() % it.second.size() % branch_variable->associated_hist.c_str();

                    map_systematic_num_universe[it.first]   = std::max((int)map_systematic_num_universe[it.first], (int)it.second.size());

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

    int PROcess_CAFAna(const PROconfig &inconfig, std::vector<SystStruct>& syst_vector, PROpeller& inprop){

        log<LOG_DEBUG>(L"%1% || Loading Monte Carlo Files") % __func__ ;

        int num_files = inconfig.m_num_mcgen_files;

        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<std::unique_ptr<TFile>> files(num_files);
        std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
        std::vector<std::vector<std::map<std::string, std::vector<eweight_type>*>>> f_event_weights(num_files);
        std::vector<std::vector<std::map<std::string, std::vector<float>*>>> f_knob_vals(num_files);
        std::map<std::string, int> map_systematic_num_universe;
        std::map<std::string, std::vector<float>> map_systematic_knob_vals;
        std::map<std::string, std::string> map_systematic_type;

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
                log<LOG_ERROR>(L"%1% || Failed to open root file: %2%") % __func__  % fn.c_str();
                exit(EXIT_FAILURE);
            }


            //first, grab friend trees
            if(inconfig.m_mcgen_numfriends[fid]>0){
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
                        log<LOG_DEBUG>(L"%1% || Adding friend tree %2% from file %3%") % __func__ %  treefriendname.c_str() % treefriendfile.c_str();
                        trees[fid]->AddFriend(treefriendname.c_str(),treefriendfile.c_str());
                    }
                }
            }
            // grab branches 
            int num_branch = inconfig.m_branch_variables[fid].size();
            f_event_weights[fid].resize(num_branch);
            f_knob_vals[fid].resize(num_branch);
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

                for (const std::string &name: branch_variable->names) {
                    branch_variable->branch_formulas.push_back(std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), name.c_str(), trees[fid]));
                    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  name.c_str();
                }

                branch_variable->branch_true_L_formula = std::make_shared<TTreeFormula>(("branch_L_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_L_name.c_str(), trees[fid]);
                log<LOG_INFO>(L"%1% || Setting up L variable for this branch: %2%") % __func__ %  branch_variable->true_L_name.c_str();
                branch_variable->branch_true_value_formula = std::make_shared<TTreeFormula>(("branch_true_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_param_name.c_str(), trees[fid]);
                log<LOG_INFO>(L"%1% || Setting up true E variable for this branch: %2%") % __func__ %  branch_variable->true_param_name.c_str();
                branch_variable->branch_true_pdg_formula = std::make_shared<TTreeFormula>(("branch_pdg_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->pdg_name.c_str(), trees[fid]);
                log<LOG_INFO>(L"%1% || Setting up PDG variable for this branch: %2%") % __func__ %  branch_variable->pdg_name.c_str();


                //grab monte carlo weight
                if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                    branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),trees[fid]);
                    log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
                }


                //grab eventweight branch
                for(const TObject* branch: *trees[fid]->GetListOfBranches()) {
                    log<LOG_DEBUG>(L"%1% || Checking if branch %2% is in allowlist") % __func__ %  branch->GetName();

                    if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                        log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();
                        trees[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][ib][branch->GetName()]));
                    } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {
                        log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2%") % __func__ % branch->GetName();
                        trees[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][ib][branch->GetName()]));
                    }
                }
                if(inconfig.m_mcgen_numfriends[fid]>0){
                    for(const TObject* friend_: *trees[fid]->GetListOfFriends()) {
                        for(const TObject* branch: *((TFriendElement*)friend_)->GetTree()->GetListOfBranches()) {
                            log<LOG_DEBUG>(L"%1% || Checking if branch %2% is in allowlist") % __func__ %  branch->GetName();

                            if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                                if(branch_variable->GetIncludeSystematics()){
                                    log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();
                                    trees[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][ib][branch->GetName()]));
                                }else{
                                    log<LOG_INFO>(L"%1% || EXPLICITLY NOT Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();
                                }
                            } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {
                                log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2%") % __func__ % branch->GetName();
                                trees[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][ib][branch->GetName()]));
                            }
                        }
                    }
                }
                log<LOG_INFO>(L"%1% || This mcgen file has %2% friends.") % __func__ %  inconfig.m_mcgen_numfriends[fid];

            } //end of branch loop


            //calculate how many universes each systematic has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            trees.at(fid)->GetEntry(good_event);

            for(int ib = 0; ib != num_branch; ++ib) {
                const auto& branch_variable = inconfig.m_branch_variables[fid][ib];
                auto& f_weight = f_event_weights[fid][ib];
                auto& f_knob = f_knob_vals[fid][ib];

                if(branch_variable->GetIncludeSystematics()){
                    for(const auto& it : f_weight){
                        log<LOG_DEBUG>(L"%1% || On systematic: %2%") % __func__ % it.first.c_str();

                    int count = std::count(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), it.first);
                    if(count==0){
                        log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as its not in the AllowList!!") % __func__ % it.first.c_str();
                        continue;
                    }

                    count = std::count(inconfig.m_mcgen_variation_denylist.begin(), inconfig.m_mcgen_variation_denylist.end(), it.first);
                    if(count>0){
                        log<LOG_DEBUG>(L"%1% || Skip systematic: %2% as it is in the DenyList!!") % __func__ % it.first.c_str();
                        continue;
                    }

                        log<LOG_INFO>(L"%1% || %2% has %3% montecarlo variations in branch %4%") % __func__ % it.first.c_str() % it.second->size() % branch_variable->associated_hist.c_str();

                        map_systematic_num_universe[it.first] = std::max((int)map_systematic_num_universe[it.first], (int)it.second->size());
                        if(f_knob.find(it.first+"_sigma") != std::end(f_knob)) {
                            map_systematic_knob_vals[it.first] = *f_knob[it.first+"_sigma"];
                        }
                    }
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
            std::string sys_weight_formula = "1";
            std::string sys_mode = inconfig.m_mcgen_variation_type_map.at(sys_name);

            log<LOG_ERROR>(L"%1% || found mode %2% for systematic %3%: ") % __func__ % sys_mode.c_str() % sys_name.c_str();
            if(sys_weight_formula != "1" || sys_mode !=""){
                syst_vector.back().SetWeightFormula(sys_weight_formula);
                syst_vector.back().SetMode(sys_mode);
            }
            if(sys_mode == "spline") {
                if(map_systematic_knob_vals.find(sys_name) == map_systematic_knob_vals.end()) {
                  log<LOG_ERROR>(L"%1% || Expected %2% to have knob vals associated with it, but couldn't find any. Will use -3 to +3 as default.") % __func__ % sys_name.c_str();
                  map_systematic_knob_vals[sys_name] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
                }
                syst_vector.back().knob_index = map_systematic_knob_vals[sys_name];
                syst_vector.back().knobval = syst_vector.back().knob_index;
                std::sort(syst_vector.back().knobval.begin(), syst_vector.back().knobval.end());
            }
           
            for(size_t i = 0 ; i != inconfig.m_mcgen_weightmaps_patterns.size(); ++i){
                if (inconfig.m_mcgen_weightmaps_uses[i] && sys_name.find(inconfig.m_mcgen_weightmaps_patterns[i]) != std::string::npos) {
                    sys_weight_formula = sys_weight_formula + "*(" + inconfig.m_mcgen_weightmaps_formulas[i]+")";
                    sys_mode           = inconfig.m_mcgen_weightmaps_mode[i];


                    log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for pattern %3%") % __func__ % sys_name.c_str() % inconfig.m_mcgen_weightmaps_patterns[i].c_str();
                    log<LOG_INFO>(L"%1% || Corresponding weight is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_formulas[i].c_str();
                    log<LOG_INFO>(L"%1% || Corresponding mode is : %2%") % __func__ % inconfig.m_mcgen_weightmaps_mode[i].c_str();
                }
            }

       }


        //sanity check 
        for(const auto& s : syst_vector)
            s.SanityCheck();


        //create 2D multi-universe spec.
        for(auto& s : syst_vector){
            s.CreateSpecs(s.mode == "spline" ? inconfig.m_num_truebins_total : inconfig.m_num_bins_total);	
        }

        inprop.hist = Eigen::MatrixXd::Constant(inconfig.m_num_truebins_total, inconfig.m_num_bins_total, 0);
        inprop.histLE = Eigen::VectorXd::Constant(inconfig.m_num_truebins_total, 0);
        size_t LE_bin = 0;
        for(size_t i = 0; i < inconfig.m_num_channels; ++i) {
            const std::vector<double> &edges = inconfig.m_channel_truebin_edges[i];
            for(size_t j = 0; j < edges.size() - 1; ++j){
                inprop.histLE(LE_bin++) = (edges[j+1] + edges[j])/2;
                //double kk = (edges[j+1] + edges[j])/2;
                //log<LOG_INFO>(L"%1% || AGGGGGH %2% %3% %4% %5% %6% %7%") % __func__ % i % j % edges[j+1] % edges[j] % LE_bin % kk ;
            }
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
                    process_cafana_event(inconfig, branches[ib], f_event_weights[fid][ib], inconfig.m_mcgen_pot[fid], subchannel_index[ib], syst_vector, sys_weight_value, inprop);
                } 

            } //end of entry loop

        } //end of file loop

        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Finish reading files, it took %2% seconds..") % __func__ % time_took;
        log<LOG_INFO>(L"%1% || DONE") %__func__ ;

        return 0;
    }

    int PROcess_CAFs(const PROconfig &inconfig, std::vector<SystStruct>& syst_vector, PROpeller &inprop){

        log<LOG_DEBUG>(L"%1% || Starting to process input root files  ") % __func__ ;

        std::vector<std::string> variations;
        std::vector<std::string> variations_tmp;

        int num_files = inconfig.m_num_mcgen_files;

        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<std::unique_ptr<TFile>> files(num_files);
        std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
        std::map<std::string, int> map_systematic_num_universe;
        std::vector<double> pot_scale(num_files, 1.0);

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

            //calculate POT scale factor
            //if(inconfig.m_mcgen_pot.at(fid) != -1){
            //    pot_scale[fid] = spec_pot/inconfig.m_mcgen_pot.at(fid);
            //                                       }
            //pot_scale[fid] *= inconfig.m_mcgen_scale[fid];

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

                for (const std::string &name: branch_variable->names) {
                    branch_variable->branch_formulas.push_back(std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), name.c_str(), trees[fid]));
                    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  name.c_str();
                }

                branch_variable->branch_true_pdg_formula  =  std::make_shared<TTreeFormula>(("branch_add_pdg_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->pdg_name.c_str(),trees[fid]);
                branch_variable->branch_true_value_formula  =  std::make_shared<TTreeFormula>(("branch_add_trueE_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_param_name.c_str(),trees[fid]);
                branch_variable->branch_true_L_formula  =  std::make_shared<TTreeFormula>(("branch_add_trueL_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_L_name.c_str(),trees[fid]);

                //grab monte carlo weight
                if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                    branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),trees[fid]);
                    log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
                }

            } //end of branch loop

            //calculate how many "universes" each systematic has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            trees.at(fid)->GetEntry(good_event);


            log<LOG_DEBUG>(L"%1% || On file %2% - %3% Starting on cafana pset loop to build SysVec ") % __func__ % fid % fn.c_str()  ;
            for(size_t v =0; v< cafana_pset_names.size(); v++){
                std::string varname = cafana_pset_names[v];

                log<LOG_DEBUG>(L"%1% || starting %2% ") % __func__ % varname.c_str()  ;

                if(inconfig.m_mcgen_variation_allowlist.size()> 0 ){
                    int ncount = std::count(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), varname);
                    if(ncount==0){
                        log<LOG_INFO>(L"%1% || Skipping %2% as its not in allowlist ") % __func__ % varname.c_str();
                        continue;
                    }
                }

                if(inconfig.m_mcgen_variation_denylist.size()> 0 ){
                    int ncount = std::count(inconfig.m_mcgen_variation_denylist.begin(), inconfig.m_mcgen_variation_denylist.end(), varname);
                    if(ncount>0){
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


                log<LOG_DEBUG>(L"%1% || %2% has %3% montecarlos in file %4% ") % __func__ % varname.c_str() % map_systematic_num_universe[varname] % fid  ;

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
            s.CreateSpecs(s.mode == "multisigma" ? inconfig.m_num_truebins_total : inconfig.m_num_bins_total);	
        }

        inprop.hist = Eigen::MatrixXd::Constant(inconfig.m_num_truebins_total, inconfig.m_num_bins_total, 0);
        inprop.histLE = Eigen::VectorXd::Constant(inconfig.m_num_truebins_total, 0);
        size_t LE_bin = 0;
        for(size_t i = 0; i < inconfig.m_num_channels; ++i) {
            const std::vector<double> &edges = inconfig.m_channel_truebin_edges[i];
            for(size_t j = 0; j < edges.size() - 1; ++j)
                inprop.histLE(LE_bin++) = (edges[j+1] - edges[j])/2;
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
                    std::vector<double> reco_value = branches[ib]->GetValues<double>();
                    float additional_weight = branches[ib]->GetMonteCarloWeight();
                    //additional_weight *= pot_scale[fid]; POT NOT YET FIX
                    int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index[ib]);
                    int pdg_id = branches[ib]->GetTruePDG<int>();
                    double true_param = branches[ib]->GetTrueValue<double>();
                    double baseline = branches[ib]->GetTrueL<double>();
                    int global_true_bin = FindGlobalTrueBin(inconfig, baseline / true_param, subchannel_index[ib]);
                    int model_rule = branches[ib]->GetModelRule();

                    log<LOG_DEBUG>(L"%1% || Reco and True E values: %2% and  %3%") % __func__ % reco_value[0] % true_param;

                    if(additional_weight == 0 || global_bin < 0)
                        continue;

                    for (unsigned ir = 0; ir < reco_value.size(); ir++) {
                      if (inprop.recos.size() <= ir) inprop.recos.push_back({});
                      inprop.recos[ir].push_back(reco_value[ir]);
                    }

                    inprop.added_weights.push_back(additional_weight);
                    inprop.bin_indices.push_back(global_bin);
                    inprop.pdg.push_back(pdg_id);
                    inprop.truth.push_back((float)true_param);
                    inprop.baseline.push_back((float)baseline);
                    inprop.model_rule.push_back((int)model_rule);
                    inprop.true_bin_indices.push_back((int)global_true_bin);
                    inprop.hist(global_true_bin, global_bin) += additional_weight;
                    PROcess_CAF_Event(sys_weight_formula, syst_vector, v_cafhelper[fid], additional_weight, global_bin, global_true_bin);

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


    int PROcess_CAF_Event(std::vector<std::unique_ptr<TTreeFormula>> & formulas, std::vector<SystStruct> &syst_vector, CAFweightHelper &caf_helper, double add_weight, long int global_bin, long int global_true_bin){


        int is = 0;
        for(SystStruct & syst : syst_vector){
            long bin = ((syst.mode == "multisigma") ? global_true_bin : global_bin);
            syst.FillCV(bin, add_weight);
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
                syst.FillUniverse(u, bin, add_weight*this_weight);
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

            if (inconfig.m_mcgen_numfriends[fid]>0){
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

                for (const std::string &name: branch_variable->names) {
                    branch_variable->branch_formulas.push_back(std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), name.c_str(), trees[fid]));
                    log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  name.c_str();
                }

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
                    std::vector<double> reco_value = branches[ib]->GetValues<double>();
                    double additional_weight = branches[ib]->GetMonteCarloWeight();
                    additional_weight *= pot_scale[fid];

                    if(additional_weight == 0) //skip on event failing cuts
                        continue;

                    //find bins
                    int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index[ib]);
                    if(global_bin < 0 )  //out of range
                        continue;

                    if(i%100==0)	
                        log<LOG_DEBUG>(L"%1% || Subchannel %2% -- Reco variable value: %3%, MC event weight: %4%, correponds to global bin: %5%") % __func__ %  subchannel_index[ib] % reco_value[0] % additional_weight % global_bin;

                    spec.Fill(global_bin, additional_weight);

                }  //end of branch loop

            } //end of entry loop

        } //end of file loop

        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Generating central value spectrum took %2% seconds..") % __func__ % time_took;
        log<LOG_INFO>(L"%1% || DONE") %__func__ ;

        return spec;
    }

    void process_cafana_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>*>& eventweight_map, double mcpot, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<double>& syst_additional_weight, PROpeller& inprop){

        int total_num_sys = syst_vector.size(); 
      	std::vector<double> reco_value = branch->GetValues<double>();
        double true_param = branch->GetTrueValue<double>();
        double baseline = branch->GetTrueL<double>();
        double true_value = baseline / true_param;
        double pdg_id = branch->GetTruePDG();//No need, depreciated
        int run_syst = branch->GetIncludeSystematics();
        double mc_weight = branch->GetMonteCarloWeight();
        mc_weight *= inconfig.m_plot_pot / mcpot;
        int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index);
        int global_true_bin = run_syst ? FindGlobalTrueBin(inconfig, true_value, subchannel_index) : 0 ;//seems werid, but restricts ALL cosmics to one bin. 
        int model_rule = branch->GetModelRule();
        
        if(global_bin < 0 )  //out of range
            return;
        if(global_true_bin < 0)
            return;

        for (unsigned ir = 0; ir < reco_value.size(); ir++) {
            if (inprop.recos.size() <= ir) inprop.recos.push_back({});
            inprop.recos[ir].push_back(reco_value[ir]);
        }

        inprop.added_weights.push_back(mc_weight);
        inprop.bin_indices.push_back(global_bin);
        inprop.pdg.push_back(pdg_id);
        inprop.truth.push_back((float)true_param);
        inprop.baseline.push_back((float)baseline);
        inprop.model_rule.push_back((int)model_rule);
        inprop.true_bin_indices.push_back((int)global_true_bin);
        inprop.hist(global_true_bin, global_bin) += mc_weight;

        if(!run_syst) return;

        for(int i = 0; i != total_num_sys; ++i){
            SystStruct& syst_obj = syst_vector[i];
            double additional_weight = syst_additional_weight.at(i);

            auto map_iter = eventweight_map.find(syst_obj.GetSysName());

            if(syst_obj.mode == "spline") {
                syst_obj.FillCV(global_true_bin, mc_weight);
                for(int i = 0; i < syst_obj.GetNUniverse(); ++i){
                    size_t u = 0;
                    for(; u < syst_obj.knobval.size(); ++u)
                        if(syst_obj.knobval[u] == syst_obj.knob_index[i]) break;
                    syst_obj.FillUniverse(u, global_true_bin, mc_weight * additional_weight * static_cast<double>(map_iter->second->at(i)));
                }
                continue;
            }else{

                syst_obj.FillCV(global_bin, mc_weight);
                for(int iuni = 0; iuni < syst_obj.GetNUniverse(); ++iuni){
                    double sys_wei = run_syst ? additional_weight * static_cast<double>(map_iter->second->at(iuni) ) :  1.0;
                    syst_obj.FillUniverse(iuni, global_bin, mc_weight *sys_wei );
                }
            }

        }


        return;
    }

    void process_sbnfit_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>>& eventweight_map, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<double>& syst_additional_weight){

        int total_num_sys = syst_vector.size(); 
        std::vector<double> reco_value = branch->GetValues<double>();
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


}//namespace

