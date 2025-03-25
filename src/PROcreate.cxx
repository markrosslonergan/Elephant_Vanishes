#include "PROcreate.h"
#include "PROdata.h"
#include "PROlog.h"
#include "PROpeller.h"
#include "PROtocall.h"
#include "TTree.h"
#include "TFile.h"
#include "TFriendElement.h"
#include "TChain.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
namespace PROfit {




    void saveSystStructVector(const std::vector<SystStruct> &structs, const std::string &filename) {
        std::ofstream ofs(filename, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa & structs;  
    }

    void loadSystStructVector(std::vector<SystStruct> &structs, const std::string &filename) {
        std::ifstream ifs(filename, std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia & structs;  
    }


    std::string convertToXRootD(std::string fname_orig){
        std::string fname_use = fname_orig;
        if(fname_orig.find("pnfs")!=std::string::npos ){
            std::string p = "/pnfs";
            std::string::size_type i = fname_orig.find(p);
            fname_orig.erase(i,p.length());
            fname_use = "root://fndca1.fnal.gov:1094/pnfs/fnal.gov/usr"+fname_orig;
        }
        return fname_use;
    }


    void SystStruct::CleanSpecs(){
        if(p_cv)  p_cv.reset();
        p_multi_spec.clear();
        return;
    }

    void SystStruct::CreateSpecs(int num_bins){
        this->CleanSpecs();
        log<LOG_DEBUG>(L"%1% || Creating multi-universe spectrum with dimension (%2% x %3%)") % __func__ % n_univ % num_bins;

        p_cv = std::make_shared<PROspec>(num_bins);
        for(int i = 0; i != n_univ; ++i){
            p_multi_spec.push_back(std::make_shared<PROspec>(num_bins));
        }
        return;
    }


    const PROspec& SystStruct::CV() const{
        return *p_cv;
    }

    const PROspec& SystStruct::Variation(int universe) const{
        return *(p_multi_spec.at(universe));
    }

    void SystStruct::FillCV(int global_bin, float event_weight){
        p_cv->Fill(global_bin, event_weight);
        return;
    }

    void SystStruct::FillUniverse(int universe, int global_bin, float event_weight){
        log<LOG_INFO>(L"%1% || p_multi_spec size %2% and universe %3%") % __func__ % p_multi_spec.size() % universe;
        p_multi_spec.at(universe)->QuickFill(global_bin, event_weight);
        return;
    }

    void SystStruct::SanityCheck() const{
        if(mode == "minmax" && n_univ != 2){
            log<LOG_ERROR>(L"%1% || Systematic variation %2% is tagged as minmax mode, but has %3% universes (can only be 2)") % __func__ % systname.c_str() % n_univ;
            log<LOG_ERROR>(L"Terminating.");
            exit(EXIT_FAILURE);
        }

        log<LOG_DEBUG>(L"%1% || Systematic variation %2% has %3% universes, and is in %4% mode with weight formula: %5%") % __func__ % systname.c_str() % n_univ % mode.c_str() % weight_formula.c_str();
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


        //create 2D multi-universe spec for covariances
        for(auto& s : syst_vector){
            if(s.mode=="covariance")
                s.CreateSpecs(inconfig.m_num_bins_total);	
        }


        time_t start_time = time(nullptr), time_stamp = time(nullptr);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;


            // set up systematic weight formula
            std::vector<float> sys_weight_value(total_num_systematics, 1.0);
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

    int PROcess_CAFAna(const PROconfig &inconfig, std::vector<SystStruct>& syst_vector, PROpeller& inprop,bool noxrootd){

        log<LOG_DEBUG>(L"%1% || Loading Monte Carlo Files") % __func__ ;

        int num_files = inconfig.m_num_mcgen_files;

        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<std::unique_ptr<TFile>> files(num_files);
        //std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
        std::vector<TChain*> chains(num_files,nullptr);
        std::vector<std::vector<TChain*>> friendChains(num_files) ;


        std::vector<std::vector<std::map<std::string, std::vector<eweight_type>*>>> f_event_weights(num_files);
        std::vector<std::vector<std::map<std::string, std::vector<float>*>>> f_knob_vals(num_files);
        std::map<std::string, int> map_systematic_num_universe;
        std::map<std::string, std::vector<float>> map_systematic_knob_vals;
        std::map<std::string, std::string> map_systematic_type;

        //open files, and link trees and branches
        int good_event = 0;
        bool useXrootD = !noxrootd;

        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);


            std::vector<std::string> filesForChain;

            if (fn.find(".root") != std::string::npos) {
                log<LOG_INFO>(L"%1% || Starting a (single) TCHain, loading root file %2%") % __func__  % fn.c_str();
                filesForChain.push_back(useXrootD ? convertToXRootD(fn) : fn);
            }else{

                log<LOG_INFO>(L"%1% || Starting a TCHain number %3%, loading from filelist %2%") % __func__  % fn.c_str() % fid;
                std::ifstream infile(fn.c_str()); 
                if (!infile) {
                    log<LOG_ERROR>(L"%1% || Failed to open input filelist %2%") % __func__  % fn.c_str();
                    exit(EXIT_FAILURE);
                }

                std::string line;
                while (std::getline(infile, line)) {
                    if(useXrootD) line = convertToXRootD(line);
                    log<LOG_INFO>(L"%1% || Loading file %2% into TChain number %3% ") %__func__ % line.c_str() % fid;
                    filesForChain.push_back(line);
                }

                infile.close();
            }

            chains[fid] = new TChain(inconfig.m_mcgen_tree_name.at(fid).c_str());
            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                friendChains[fid].resize(inconfig.m_mcgen_numfriends[fid], nullptr);
            }

            for(auto &file: filesForChain){ 

                chains[fid]->Add(file.c_str()); 

                if (inconfig.m_mcgen_numfriends[fid] > 0) {
                    auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
                    if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {
                        auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
                        if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
                            log<LOG_ERROR>(L"%1% || Friend TTree provided but no friend file??") % __func__;
                            log<LOG_ERROR>(L"Terminating.");
                            exit(EXIT_FAILURE);
                        }

                        for (size_t k = 0; k < mcgen_file_friend_treename_iter->second.size(); k++) {
                            std::string treefriendname = mcgen_file_friend_treename_iter->second.at(k);
                            //std::string treefriendfile = mcgen_file_friend_iter->second.at(k);//not used

                            if (!friendChains[fid][k]) {
                                friendChains[fid][k] = new TChain(treefriendname.c_str());
                            }

                            // Add the file to the friend chain
                            log<LOG_DEBUG>(L"%1% || Adding friend tree %2% from file %3%") % __func__ % treefriendname.c_str() % file.c_str();
                            friendChains[fid][k]->Add(file.c_str());
                        }

                    }
                } // End friend initialization
            } // End chain filling

        }//end loading the main files and PREparing the chains

        //add chains and summarize
        log<LOG_DEBUG>(L"%1% || Starting summary of friends and chains in total over all files and friends ") % __func__ ;
        for (size_t fid = 0; fid < num_files; fid++) {
            if (inconfig.m_mcgen_numfriends[fid] > 0) {
                for (size_t k = 0; k < friendChains[fid].size(); k++) {
                    log<LOG_DEBUG>(L"%1% || Adding friend chain %2% (%5%) to main chain %3% (%4%)") % __func__ % k % fid % chains[fid]->GetName() % friendChains[fid][k]->GetName();
                    chains[fid]->AddFriend(friendChains[fid][k]);
                }
            }
        }

        //new loop over the filled, and friended   
        for(int fid=0; fid < num_files; ++fid) {

            nentries[fid] = chains[fid]->GetEntries();

            // grab branches 
            int num_branch = inconfig.m_branch_variables[fid].size();
            f_event_weights[fid].resize(1);//was num_branch
            f_knob_vals[fid].resize(1);//was num_brach
            bool gotWeights = false;

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

                branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), chains[fid]);
                log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();
                branch_variable->branch_true_L_formula = std::make_shared<TTreeFormula>(("branch_L_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_L_name.c_str(), chains[fid]);
                log<LOG_INFO>(L"%1% || Setting up L variable for this branch: %2%") % __func__ %  branch_variable->true_L_name.c_str();
                branch_variable->branch_true_value_formula = std::make_shared<TTreeFormula>(("branch_true_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_param_name.c_str(), chains[fid]);
                log<LOG_INFO>(L"%1% || Setting up true E variable for this branch: %2%") % __func__ %  branch_variable->true_param_name.c_str();
                branch_variable->branch_true_pdg_formula = std::make_shared<TTreeFormula>(("branch_pdg_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->pdg_name.c_str(), chains[fid]);
                log<LOG_INFO>(L"%1% || Setting up PDG variable for this branch: %2%") % __func__ %  branch_variable->pdg_name.c_str();
                if (branch_variable->hist_reweight) {
                    branch_variable->branch_true_proton_mom_formula = std::make_shared<TTreeFormula>(("branch_pmom_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_proton_mom_name.c_str(), chains[fid]);
                    log<LOG_INFO>(L"%1% || Setting up leading proton momentum variable for this branch: %2%") % __func__ %  branch_variable->true_proton_mom_name.c_str();
                    branch_variable->branch_true_proton_costh_formula = std::make_shared<TTreeFormula>(("branch_costh_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->true_proton_costh_name.c_str(), chains[fid]);
                    log<LOG_INFO>(L"%1% || Setting up leading proton costh variable for this branch: %2%") % __func__ %  branch_variable->true_proton_costh_name.c_str();
                }


                //grab monte carlo weight
                if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                    branch_variable->branch_monte_carlo_weight_formula  =  std::make_shared<TTreeFormula>(("branch_add_weight_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),inconfig.m_mcgen_additional_weight_name[fid][ib].c_str(),chains[fid]);
                    log<LOG_INFO>(L"%1% || Setting up additional monte carlo weight for this branch: %2%") % __func__ %  inconfig.m_mcgen_additional_weight_name[fid][ib].c_str();
                }


                //grab eventweight branch. Little hacky. Make sure to only set each one one.
                std::unordered_map<std::string,bool> map_syst_added;

                if(!gotWeights){
                    for(const TObject* branch: *chains[fid]->GetListOfBranches()) {
                        log<LOG_DEBUG>(L"%1% || GotWeights? %3% .. Checking if branch %2% is in allowlist") % __func__ %  branch->GetName() % gotWeights;

                        if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                            if (map_syst_added.find(branch->GetName()) == map_syst_added.end()) {
                                log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2% for fid %3$") % __func__ %  branch->GetName() % fid;
                                chains[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][0][branch->GetName()]));
                                map_syst_added[branch->GetName()]=true;
                            }else{
                                log<LOG_DEBUG>(L"%1% || OK, skipping above setup as its already been added for the file. weird branch/file mismatch.") % __func__ ;
                            }

                        } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {
                            if (map_syst_added.find(branch->GetName()) == map_syst_added.end()) {
                                log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2% for fid %3%") % __func__ % branch->GetName() % fid;
                                chains[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][0][branch->GetName()]));
                                map_syst_added[branch->GetName()]=true;
                            }else{
                                log<LOG_DEBUG>(L"%1% || OK, skipping above setup as its already been added for the file. weird branch/file mismatch.") % __func__ ;
                            }
                        }
                    }
                    if(inconfig.m_mcgen_numfriends[fid]>0){
                        log<LOG_DEBUG>(L"%1% || Some Friend Processing") % __func__ ;
                        for(const TObject* friend_: *chains[fid]->GetListOfFriends()) {
                            for(const TObject* branch: *((TFriendElement*)friend_)->GetTree()->GetListOfBranches()) {

                                log<LOG_DEBUG>(L"%1% || (friend) Checking if branch %2% in friend %3% is in allowlist") % __func__ %  branch->GetName() % friend_->GetName();

                                if (std::find(inconfig.m_mcgen_variation_allowlist.begin(), inconfig.m_mcgen_variation_allowlist.end(), branch->GetName()) != inconfig.m_mcgen_variation_allowlist.end()) {
                                    //its in here
                                    if(branch_variable->GetIncludeSystematics()){
                                        if (map_syst_added.find(branch->GetName()) == map_syst_added.end()) {
                                            log<LOG_INFO>(L"%1% || Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();

                                            chains[fid]->SetBranchAddress(branch->GetName(), &(f_event_weights[fid][0][branch->GetName()]));
                                            
                                            map_syst_added[branch->GetName()]=true;
                                        }else{
                                            log<LOG_DEBUG>(L"%1% || OK, skipping above setup as its already been added for the file. weird branch/file mismatch.") % __func__ ;
                                        }



                                    }else{
                                        log<LOG_INFO>(L"%1% || EXPLICITLY NOT Setting up eventweight map for this branch: %2%") % __func__ %  branch->GetName();
                                    }
                                } else if(strlen(branch->GetName()) > 6 && strcmp(branch->GetName() + strlen(branch->GetName()) - 6, "_sigma") == 0) {

                                    if (map_syst_added.find(branch->GetName()) == map_syst_added.end()) {
                                        log<LOG_INFO>(L"%1% || Setting up knob val list using branch %2%") % __func__ % branch->GetName();
                                        chains[fid]->SetBranchAddress(branch->GetName(), &(f_knob_vals[fid][0][branch->GetName()]));
                                        map_syst_added[branch->GetName()]=true;
                                    }else{
                                        log<LOG_DEBUG>(L"%1% || OK, skipping above setup as its already been added for the file. weird branch/file mismatch.") % __func__ ;
                                    }

                                }
                            }
                        }
                    }
                    gotWeights = true;
                }//
                log<LOG_INFO>(L"%1% || This mcgen file has %2% friends.") % __func__ %  inconfig.m_mcgen_numfriends[fid];

            } //end of branch loop


            //calculate how many universes each systematic has.
            log<LOG_INFO>(L"%1% || Start calculating number of universes for systematics") % __func__;
            chains.at(fid)->GetEntry(good_event);

            for(int ib = 0; ib != num_branch; ++ib) {
                const auto& branch_variable = inconfig.m_branch_variables[fid][ib];
                auto& f_weight = f_event_weights[fid][0];
                auto& f_knob = f_knob_vals[fid][0];

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

        //Do we have any flat systeatics?
        if(inconfig.m_num_variation_type_flat>0){
            for(auto& allow_sys : inconfig.m_mcgen_variation_type_map){
                if(allow_sys.second=="flat"){
                    map_systematic_num_universe[allow_sys.first] = -1;
                }
            }
        }

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

            log<LOG_INFO>(L"%1% || found mode %2% for systematic %3%: ") % __func__ % sys_mode.c_str() % sys_name.c_str();
            if(sys_weight_formula != "1" || sys_mode !=""){
                syst_vector.back().SetWeightFormula(sys_weight_formula);
                syst_vector.back().SetMode(sys_mode);
            }
            if(sys_mode == "spline") {
                if(map_systematic_knob_vals.find(sys_name) == map_systematic_knob_vals.end()) {
                    log<LOG_WARNING>(L"%1% || Expected %2% to have knob vals associated with it, but couldn't find any. Will use -3 to +3 as default.") % __func__ % sys_name.c_str();
                    map_systematic_knob_vals[sys_name] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
                }
                syst_vector.back().knob_index = map_systematic_knob_vals[sys_name];
                syst_vector.back().knobval = syst_vector.back().knob_index;
                std::sort(syst_vector.back().knobval.begin(), syst_vector.back().knobval.end());
            }
            if(sys_mode == "flat"){

                log<LOG_INFO>(L"%1% || Systematic variation %2% is a match for a flat norm systematic. Processing a such. ") % __func__ % sys_name.c_str();

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
        for(auto& s : syst_vector){
            s.SanityCheck();
            s.SetHash(inconfig.hash);
        }


        //create 2D multi-universe spec.
        for(auto& s : syst_vector){
            if(s.mode=="flat")
                continue;	
            s.CreateSpecs(s.mode == "spline" ? inconfig.m_num_truebins_total : inconfig.m_num_bins_total);	
        }

        inprop.mcStatErr = Eigen::VectorXf::Constant(inconfig.m_num_bins_total, 0);
        inprop.hist = Eigen::MatrixXf::Constant(inconfig.m_num_truebins_total, inconfig.m_num_bins_total, 0);
        inprop.histLE = Eigen::VectorXf::Constant(inconfig.m_num_truebins_total, 0);
        size_t LE_bin = 0;
        for(size_t im = 0; im < inconfig.m_num_modes; im++){
            for(size_t id =0; id < inconfig.m_num_detectors; id++){
                for(size_t ic = 0; ic < inconfig.m_num_channels; ic++){
                    const std::vector<float> &edges = inconfig.m_channel_truebin_edges[ic];
                    for(size_t sc = 0; sc < inconfig.m_num_subchannels.at(ic); sc++){
                        for(size_t j = 0; j < edges.size() - 1; ++j){
                            inprop.histLE(LE_bin++) = (edges[j+1] + edges[j])/2;
                        }
                    }
                }
            }
        }

        time_t start_time = time(nullptr), time_stamp = time(nullptr);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;


            // set up systematic weight formula
            std::vector<float> sys_weight_value(total_num_systematics, 1.0);
            std::vector<std::unique_ptr<TTreeFormula>> sys_weight_formula;
            for(const auto& s : syst_vector){
                if(s.HasWeightFormula())
                    sys_weight_formula.push_back(std::make_unique<TTreeFormula>(("weightMapsFormulas_"+std::to_string(fid)+"_"+ s.GetSysName()).c_str(), s.GetWeightFormula().c_str(),chains[fid]));
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

            //chains[fid]->SetCacheSize(1000000000); // Set cache size to 10 MB
            //chains[fid]->AddBranchToCache("*", kTRUE); // Cache all branches
            TObjArray* tbranches = chains[fid]->GetListOfBranches();
            for (int i = 0; i < tbranches->GetEntries(); i++) {
                TBranch* branch = (TBranch*)tbranches->At(i);
                const char* branchName = branch->GetName();
                bool isActive = chains[fid]->GetBranchStatus(branchName);
                log<LOG_DEBUG>(L"%1% || BRANCH %2% is %3%") % __func__ % branchName % (isActive ? "active" : "inactive");
            }

            // loop over all entries
            size_t to_print = nevents / 5;
            if(to_print>50000)to_print=50000;
            int currentTreeNumber = -1;

            for(long int i=0; i < nevents; ++i) {
                if(i%to_print==0){
                    time_t time_passed = time(nullptr) - time_stamp;
                    log<LOG_INFO>(L"%1% || File %2% -- uni : %3% / %4%  took %5% seconds") % __func__ % fid % i % nevents % time_passed;
                    time_stamp = time(nullptr);
                }
                chains[fid]->GetEntry(i);


                if (chains[fid]->GetTreeNumber() != currentTreeNumber) {
                    currentTreeNumber = chains[fid]->GetTreeNumber();


                    for(int ib = 0; ib != num_branch; ++ib) {
                        log<LOG_INFO>(L"%1% || Bcheck  %2%  %3%  %4% %5% ") % __func__ % fid % ib % currentTreeNumber % i;

                        if(inconfig.m_mcgen_additional_weight_bool[fid][ib]){
                            branches[ib]->branch_monte_carlo_weight_formula->GetNdata();
                            branches[ib]->branch_monte_carlo_weight_formula->UpdateFormulaLeaves();
                        }
                        branches[ib]->branch_true_pdg_formula->GetNdata();
                        branches[ib]->branch_true_pdg_formula->UpdateFormulaLeaves();
                        branches[ib]->branch_formula->GetNdata();
                        branches[ib]->branch_formula->UpdateFormulaLeaves();
                        branches[ib]->branch_true_L_formula->GetNdata();
                        branches[ib]->branch_true_L_formula->UpdateFormulaLeaves();
                        branches[ib]->branch_true_value_formula->GetNdata();
                        branches[ib]->branch_true_value_formula->UpdateFormulaLeaves();
                        if (branches[ib]->hist_reweight) {
                            branches[ib]->branch_true_proton_mom_formula->GetNdata();
                            branches[ib]->branch_true_proton_mom_formula->UpdateFormulaLeaves();
                            branches[ib]->branch_true_proton_costh_formula->GetNdata();
                            branches[ib]->branch_true_proton_costh_formula->UpdateFormulaLeaves();
                        }
                    }

                    log<LOG_INFO>(L"%1% || B1 ") % __func__ ;
                    for(size_t is = 0; is != total_num_systematics; ++is){
                        if(syst_vector[is].HasWeightFormula()){
                            sys_weight_formula[is]->UpdateFormulaLeaves();
                            sys_weight_formula[is]->GetNdata();	
                        }
                    }//end sys
                    log<LOG_INFO>(L"%1% || B2 ") % __func__ ;
                    TObjArray* tbranches = chains[fid]->GetListOfBranches();
                    for (int i = 0; i < tbranches->GetEntries(); i++) {
                        TBranch* branch = (TBranch*)tbranches->At(i);
                        const char* branchName = branch->GetName();
                        bool isActive = chains[fid]->GetBranchStatus(branchName);
                        log<LOG_DEBUG>(L"%1% || BRANCH %2% is %3%") % __func__ % branchName % (isActive ? "active" : "inactive");
                    }

                    log<LOG_INFO>(L"%1% || B3 ") % __func__ ;
                }

                //grab additional weight for systematics
                for(size_t is = 0; is != total_num_systematics; ++is){
                    if(syst_vector[is].HasWeightFormula()){
                        sys_weight_value[is] = sys_weight_formula[is]->EvalInstance();
                    }
                }

                log<LOG_INFO>(L"%1% || B4 ") % __func__ ;

                //branch loop
                for(int ib = 0; ib != num_branch; ++ib) {
                    log<LOG_INFO>(L"%1% || B5 going inside %2% ") % __func__ % ib;
                    process_cafana_event(inconfig, branches[ib], f_event_weights[fid][0], inconfig.m_mcgen_pot[fid], subchannel_index[ib], syst_vector, sys_weight_value, inprop);
                } 

            } //end of entry loop

        } //end of file loop

        //ensure hash is correctly assigned
        inprop.hash=inconfig.hash;
        inprop.mcStatErr = inprop.mcStatErr.array().sqrt();

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
        std::vector<float> pot_scale(num_files, 1.0);

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

                branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form_"+std::to_string(fid) +"_" + std::to_string(ib)).c_str(), branch_variable->name.c_str(), trees[fid]);
                log<LOG_INFO>(L"%1% || Setting up reco variable for this branch: %2%") % __func__ %  branch_variable->name.c_str();

                branch_variable->branch_true_pdg_formula  =  std::make_shared<TTreeFormula>(("branch_add_pdg_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->pdg_name.c_str(),trees[fid]);
                branch_variable->branch_true_value_formula  =  std::make_shared<TTreeFormula>(("branch_add_trueE_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_param_name.c_str(),trees[fid]);
                branch_variable->branch_true_L_formula  =  std::make_shared<TTreeFormula>(("branch_add_trueL_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_L_name.c_str(),trees[fid]);
                branch_variable->branch_true_proton_mom_formula  =  std::make_shared<TTreeFormula>(("branch_add_true_pmom_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_proton_mom_name.c_str(),trees[fid]);
                branch_variable->branch_true_proton_costh_formula  =  std::make_shared<TTreeFormula>(("branch_add_true_pcosth_"+std::to_string(fid)+"_" + std::to_string(ib)).c_str(),branch_variable->true_proton_costh_name.c_str(),trees[fid]);

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

        inprop.hist = Eigen::MatrixXf::Constant(inconfig.m_num_truebins_total, inconfig.m_num_bins_total, 0);
        inprop.histLE = Eigen::VectorXf::Constant(inconfig.m_num_truebins_total, 0);
        size_t LE_bin = 0;
        for(size_t i = 0; i < inconfig.m_num_channels; ++i) {
            const std::vector<float> &edges = inconfig.m_channel_truebin_edges[i];
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
                    float reco_value = branches[ib]->GetValue<float>();
                    float additional_weight = branches[ib]->GetMonteCarloWeight();
                    //additional_weight *= pot_scale[fid]; POT NOT YET FIX

                    int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index[ib]);
                    float true_param = branches[ib]->GetTrueValue<float>();
                    float baseline = branches[ib]->GetTrueL<float>();
                    float pmom = branches[ib]->GetTrueLeadProtonMom<float>();
                    float pcosth = branches[ib]->GetTrueLeadProtonCosth<float>();
                    int global_true_bin = FindGlobalTrueBin(inconfig, baseline / true_param, subchannel_index[ib]);
                    int model_rule = branches[ib]->GetModelRule();


                    if(additional_weight == 0 || global_bin < 0)
                        continue;

                    inprop.added_weights.push_back(additional_weight);
                    inprop.bin_indices.push_back(global_bin);
                    inprop.trueLE.push_back((float)(baseline/true_param));
                    inprop.model_rule.push_back((int)model_rule);
                    inprop.pmom.push_back((float)pmom);
                    inprop.pcosth.push_back((float)pcosth);
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


    int PROcess_CAF_Event(std::vector<std::unique_ptr<TTreeFormula>> & formulas, std::vector<SystStruct> &syst_vector, CAFweightHelper &caf_helper, float add_weight, long int global_bin, long int global_true_bin){


        int is = 0;
        for(SystStruct & syst : syst_vector){
            long bin = ((syst.mode == "multisigma") ? global_true_bin : global_bin);
            syst.FillCV(bin, add_weight);
            formulas[is]->GetNdata();
            float sys_weight_value =formulas[is]->EvalInstance();

            if(std::isinf(sys_weight_value) || sys_weight_value != sys_weight_value){
                log<LOG_ERROR>(L"%1% || Input values to histogram is NAN or inf %2% !") % __func__  % sys_weight_value ;
                throw std::runtime_error("NAN or INF in pushed string");
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

    PROdata CreatePROdata(const PROconfig& inconfig){
        float spec_pot = inconfig.m_plot_pot;
        log<LOG_INFO>(L"%1% || Start generating data spectrum") % __func__ ;
        log<LOG_INFO>(L"%1% || Spectrum will be generated with %2% POT") % __func__ % spec_pot;

        int num_files = inconfig.m_num_mcgen_files;
        log<LOG_DEBUG>(L"%1% || Starting to read file and build spectrum!") % __func__ ;
        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

        std::vector<long int> nentries(num_files,0);
        std::vector<float> pot_scale(num_files, 1.0);
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
        PROdata data(inconfig.m_num_bins_total);
        log<LOG_INFO>(L"%1% || Start reading the files..") % __func__;
        for(int fid=0; fid < num_files; ++fid) {
            const auto& fn = inconfig.m_mcgen_file_name.at(fid);
            long int nevents = std::min(inconfig.m_mcgen_maxevents[fid], nentries[fid]);
            log<LOG_DEBUG>(L"%1% || Start @files: %2% which has %3% events") % __func__ % fn.c_str() % nevents;

            // grab the subchannel index
            int num_branch = inconfig.m_branch_variables[fid].size();
            auto& branches = inconfig.m_branch_variables[fid];
            std::vector<int> channel_index(num_branch, 0); 
            log<LOG_INFO>(L"%1% || This file includes %2% branch/channels") % __func__ % num_branch;
            for(int ib = 0; ib != num_branch; ++ib) {

                const std::string& subchannel_name = inconfig.m_branch_variables[fid][ib]->associated_hist;
                // Note: This will only work if there's only 1 subchannel per channel
                channel_index[ib] = inconfig.GetSubchannelIndex(subchannel_name);
                log<LOG_DEBUG>(L"%1% || Subchannel: %2% maps to index: %3%") % __func__ % subchannel_name.c_str() % channel_index[ib];
            }

            // loop over all entries
            for(long int i=0; i < nevents; ++i) {
                if(i%1000==0)	log<LOG_INFO>(L"%1% || -- uni : %2% / %3%") % __func__ % i % nevents;
                trees[fid]->GetEntry(i);

                //branch loop
                for(int ib = 0; ib != num_branch; ++ib) {
                    float reco_value = branches[ib]->GetValue<float>();
                    float additional_weight = branches[ib]->GetMonteCarloWeight();
                    additional_weight *= pot_scale[fid];

                    if(additional_weight == 0) //skip on event failing cuts
                        continue;

                    //find bins
                    int global_bin = FindGlobalBin(inconfig, reco_value, channel_index[ib]);
                    if(global_bin < 0 )  //out of range
                        continue;

                    if(i%100==0)	
                        log<LOG_DEBUG>(L"%1% || Subchannel %2% -- Reco variable value: %3%, MC event weight: %4%, correponds to global bin: %5%") % __func__ %  channel_index[ib] % reco_value % additional_weight % global_bin;

                    data.Fill(global_bin, additional_weight);
                }  //end of branch loop
            } //end of entry loop
        } //end of file loop
        time_t time_took = time(nullptr) - start_time;
        log<LOG_INFO>(L"%1% || Generating data spectrum took %2% seconds..") % __func__ % time_took;
        return data;
    }


    PROspec CreatePROspecCV(const PROconfig& inconfig){

        float spec_pot = inconfig.m_plot_pot;
        log<LOG_INFO>(L"%1% || Start generating central value spectrum") % __func__ ;
        log<LOG_INFO>(L"%1% || Spectrum will be generated with %2% POT") % __func__ % spec_pot;

        int num_files = inconfig.m_num_mcgen_files;
        log<LOG_DEBUG>(L"%1% || Starting to read file and build spectrum!") % __func__ ;
        log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;


        std::vector<long int> nentries(num_files,0);
        std::vector<float> pot_scale(num_files, 1.0);
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
                    float reco_value = branches[ib]->GetValue<float>();
                    float additional_weight = branches[ib]->GetMonteCarloWeight();
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

    void process_cafana_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>*>& eventweight_map, float mcpot, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<float>& syst_additional_weight, PROpeller& inprop){



        log<LOG_INFO>(L"%1% || A1  ") % __func__ ;
        int total_num_sys = syst_vector.size(); 
        float reco_value = branch->GetValue<float>();
        float true_param = branch->GetTrueValue<float>();
        float baseline = branch->GetTrueL<float>();
        float true_value = baseline / true_param;

        log<LOG_INFO>(L"%1% || A2  %2% ") % __func__ % reco_value;
        float pmom = branch->GetTrueLeadProtonMom<double>();
        float pcosth = branch->GetTrueLeadProtonCosth<double>();
        int run_syst = branch->GetIncludeSystematics();
        float mc_weight = branch->GetMonteCarloWeight();
        mc_weight *= inconfig.m_plot_pot / mcpot;

        log<LOG_INFO>(L"%1% || A3  %2%") % __func__ % mc_weight;
        int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index);
        int global_true_bin = run_syst ? FindGlobalTrueBin(inconfig, true_value, subchannel_index) : 0 ;//seems werid, but restricts ALL cosmics to one bin. 
        int model_rule = branch->GetModelRule();

        log<LOG_INFO>(L"%1% || A4  ") % __func__ ;
        if(global_bin < 0 )  //out of range
            return;
        if(global_true_bin < 0)
            return;
        if(mc_weight == 0)
            return;

        log<LOG_INFO>(L"%1% || A5  ") % __func__ ;
        inprop.added_weights.push_back(mc_weight);
        inprop.bin_indices.push_back(global_bin);
        inprop.trueLE.push_back((float)(baseline/true_param));
        inprop.model_rule.push_back((int)model_rule);
        inprop.true_bin_indices.push_back((int)global_true_bin);
        inprop.hist(global_true_bin, global_bin) += mc_weight;
        inprop.pmom.push_back((float)pmom);
        inprop.pcosth.push_back((float)pcosth);
        inprop.mcStatErr(global_bin) += 1;

        log<LOG_INFO>(L"%1% || A6  ") % __func__ ;
        if(!run_syst) return;

        for(int i = 0; i != total_num_sys; ++i){


            log<LOG_INFO>(L"%1% || C1 %2% ") % __func__ % i;
            SystStruct& syst_obj = syst_vector[i];
            float additional_weight = syst_additional_weight.at(i);
            auto map_iter = eventweight_map.find(syst_obj.GetSysName());
            log<LOG_INFO>(L"%1% || C2 %2% ") % __func__ % i;


            if(syst_obj.mode == "spline") {
                log<LOG_INFO>(L"%1% || Cspline1 %2% ") % __func__ % i;
                syst_obj.FillCV(global_true_bin, mc_weight);
                log<LOG_INFO>(L"%1% || Cspline2 %2% ") % __func__ % global_true_bin;

                for(int is = 0; is < syst_obj.GetNUniverse(); ++is){
                    log<LOG_INFO>(L"%1% || Cspline3 %2% ") % __func__ % is;
                    size_t u = 0;
                    for(; u < syst_obj.knobval.size(); ++u)
                        if(syst_obj.knobval[u] == syst_obj.knob_index[is]) break;
                    log<LOG_INFO>(L"%1% || Cspline4 u %4%, is %2% / %3% ") % __func__ % is % map_iter->second->size() % u;
                    syst_obj.FillUniverse(u, global_true_bin, mc_weight * additional_weight * static_cast<float>(map_iter->second->at(is)));
                    log<LOG_INFO>(L"%1% || BLARG_S  %2% %3% %4%") % __func__ % additional_weight % mc_weight % static_cast<float>(map_iter->second->at(is));

                }
                continue;
            }else if(syst_obj.mode == "covariance"){

                log<LOG_INFO>(L"%1% || Ccovar1 %2% ") % __func__ % i;
                syst_obj.FillCV(global_bin, mc_weight);
                for(int iuni = 0; iuni < syst_obj.GetNUniverse(); ++iuni){
                    log<LOG_INFO>(L"%1% || Ccovar2 %2% ") % __func__ % i;
                    float sys_wei = run_syst ? additional_weight * static_cast<float>(map_iter->second->at(iuni) ) :  1.0;
                    log<LOG_INFO>(L"%1% || Ccovar3 %2% ") % __func__ % i;
                    syst_obj.FillUniverse(iuni, global_bin, mc_weight *sys_wei );
                    log<LOG_INFO>(L"%1% || BLARG_C  %2% %3% %4%") % __func__ % additional_weight % mc_weight % static_cast<float>(map_iter->second->at(iuni));
                }
            }

        }


        return;
    }

    void process_sbnfit_event(const PROconfig &inconfig, const std::shared_ptr<BranchVariable>& branch, const std::map<std::string, std::vector<eweight_type>>& eventweight_map, int subchannel_index, std::vector<SystStruct>& syst_vector, const std::vector<float>& syst_additional_weight){

        int total_num_sys = syst_vector.size(); 
        float reco_value = branch->GetValue<float>();
        float mc_weight = branch->GetMonteCarloWeight();
        int global_bin = FindGlobalBin(inconfig, reco_value, subchannel_index);
        if(global_bin < 0 )  //out of range
            return;

        for(int i = 0; i != total_num_sys; ++i){
            SystStruct& syst_obj = syst_vector[i];
            float additional_weight = syst_additional_weight.at(i);

            syst_obj.FillCV(global_bin, mc_weight);

            auto map_iter = eventweight_map.find(syst_obj.GetSysName());
            int map_variation_size = (map_iter == eventweight_map.end()) ? 0 : map_iter->second.size();
            int iuni = 0;
            for(; iuni != std::min(map_variation_size, syst_obj.GetNUniverse()); ++iuni){
                syst_obj.FillUniverse(iuni, global_bin, mc_weight * additional_weight * static_cast<float>(map_iter->second.at(iuni)));
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

