#include "PROcess_weights.h"
#include "TTree.h"
#include "TFile.h"

using namespace PROfit;

int PROcess(const PROconfig &inconfig){
 
    log<LOG_DEBUG>(L"%1% || Starting to construct CovarianceMatrixGeneration in EventWeight Mode  ") % __func__ ;

    int universes_used = 0;
    //Is there a Global weight to be applied to ALL weights, CV and otherwise, inside the eventweight class? 
    std::string bnbcorrection_str = "NAN";//this has mainly beeen superseeded by XML creation

    std::vector<std::string> variations;
    std::vector<std::string> variations_tmp;

    int num_files = inconfig.m_num_mcgen_files;

    log<LOG_DEBUG>(L"%1% || Using a total of %2% individual files") % __func__  % num_files;

    std::vector<int> nentries(num_files,0);
    std::vector<int> used_montecarlos(num_files,0);

    std::vector<std::unique_ptr<TFile>> files(num_files);
    std::vector<TTree*> trees(num_files,nullptr);//keep as bare pointers because of ROOT :(
    std::vector<std::map<std::string, std::vector<eweight_type> >* > f_weights(num_files,nullptr);

    //inconfig.m_mcgen_additional_weight.resize(num_files,1.0); its const, not allowed

    int good_event = 0;

    for(int fid=0; fid < num_files; ++fid) {
        const auto& fn = inconfig.m_mcgen_file_name.at(fid);

        files[fid] = std::make_unique<TFile>(fn.c_str(),"read");
        trees[fid] = (TTree*)(files[fid]->Get(inconfig.m_mcgen_tree_name.at(fid).c_str()));
        nentries[fid]= (int)trees.at(fid)->GetEntries();

        //Some check to see if files open right?

        //Some POT counting (FIX)
        //double pot_scale = 1.0;
        //if(inconfig.m_mcgen_pot[fid]!=-1){
        //    pot_scale = FIX_plot_pot/inconfig.m_mcgen_pot[fid];
        //}
        //mcgen_scale[fid] = inconfig.m_mcgen_scale[fid]*pot_scale;

        auto mcgen_file_friend_treename_iter = inconfig.m_mcgen_file_friend_treename_map.find(fn);
        if (mcgen_file_friend_treename_iter != inconfig.m_mcgen_file_friend_treename_map.end()) {

            auto mcgen_file_friend_iter = inconfig.m_mcgen_file_friend_map.find(fn);
            if (mcgen_file_friend_iter == inconfig.m_mcgen_file_friend_map.end()) {
                std::stringstream ss;
                throw std::runtime_error(ss.str());
            }

            for(int k=0; k < (*mcgen_file_friend_iter).second.size(); k++){

                std::string treefriendname = (*mcgen_file_friend_treename_iter).second.at(k);
                std::string treefriendfile = (*mcgen_file_friend_iter).second.at(k);
                trees[fid]->AddFriend(treefriendname.c_str(),treefriendfile.c_str());
            }
        }

        //Important step
        trees.at(fid)->SetBranchAddress(inconfig.m_mcgen_eventweight_branch_names[fid].c_str(), &(f_weights[fid]));

        for(const auto branch_variable : inconfig.m_branch_variables[fid]) {
            //quick check that this branch associated subchannel is in the known chanels;
            int is_valid_subchannel = 0;
            for(const auto &name: inconfig.m_fullnames){
                if(branch_variable->associated_hist==name){
                    log<LOG_DEBUG>(L"%1% || Found a valid subchannel for this branch %2%") % __func__  % name;
                    is_valid_subchannel++;
                }
            }
            if(is_valid_subchannel==0){
                std::cout<<" ERROR ERROR: This branch did not match one defined in the .xml : " <<branch_variable->associated_hist<<std::endl;
                std::cout<<" ERROR ERROR: There is probably a typo somehwhere in xml! "<<std::endl;
                exit(EXIT_FAILURE);

            }else if(is_valid_subchannel>1){
                std::cout<<" ERROR ERROR: This branch matched more than 1 subchannel!: " <<branch_variable->associated_hist<<std::endl;
                exit(EXIT_FAILURE);
            }

            branch_variable->branch_formula = std::make_shared<TTreeFormula>(("branch_form"+std::to_string(fid)).c_str(), branch_variable->name.c_str(), trees[fid]);
        }

        if(inconfig.m_mcgen_additional_weight_bool[fid]){
            //we have an additional weight we want to apply at run time, otherwise its just set at 1.
            std::cout<<"Setting Additional weight of : "<< inconfig.m_mcgen_additional_weight_name[fid].c_str()<<std::endl; 
            //FIX FIX
            //additional_weight_formulas[fid] =  std::make_shared<TTreeFormula>(("a_w"+std::to_string(fid)).c_str(),inconfig.m_mcgen_additional_weight_name[fid].c_str(),trees[fid]);
        }

        std::cout<<"Total Entries: "<<trees.at(fid)->GetEntries()<<" good event "<<good_event<<std::endl;
        trees.at(fid)->GetEntry(good_event);

        const auto f_weight = f_weights[fid];
        if (f_weight == nullptr) {
            std::stringstream ss;
            ss << "Could not read weight branch for file=" << fid << std::endl;
            throw std::runtime_error(ss.str());
        }

        //This bit will calculate how many "universes" the file has. if ALL default is the inputted xml value

        std::cout<<"starting"<<std::endl;
        for(const auto& it : *f_weight){
            std::cout<<"On : "<<it.first<<std::endl;
            if(it.first == bnbcorrection_str) {
                std::cout<<"Found a variation consistent with "<<bnbcorrection_str<<" . This will be instead applied as a general weight"<<std::endl;
                continue;    
            }

            if(inconfig.m_mcgen_variation_allowlist.size()> 0 ){
                if(inconfig.m_mcgen_variation_allowlist.count(it.first)==0){
                    std::cout<<"Skipping "<<it.first<<" as its not in the AllowList!!"<<std::endl;
                    continue;
                }
            }

            if(inconfig.m_mcgen_variation_denylist.size()> 0 ){
                if(inconfig.m_mcgen_variation_denylist.count(it.first)>0){
                    std::cout<<"Skipping "<<it.first<<" as it is the DenyList!!"<<std::endl;
                    continue;
                }
            }

            std::cout << it.first << " has " << it.second.size() << " montecarlos in file " << fid << std::endl;
            used_montecarlos[fid] += it.second.size();
            variations_tmp.push_back(it.first);
        }
    } // end fid

    std::sort(variations_tmp.begin(),variations_tmp.end());
    auto unique_iter = std::unique(variations_tmp.begin(), variations_tmp.end());
    variations.insert(variations.begin(),variations_tmp.begin(),unique_iter);

    //Variation Weight Maps Area
    std::vector<int> variation_modes(variations.size(),0);
    std::vector<std::string> s_formulas(variations.size(),"1"); 
    int n_wei = inconfig.m_mcgen_weightmaps_patterns.size();

    for(int i=0; i< n_wei; i++){

        for(int v=0; v< variations.size(); v++){

            // Check to see if pattern is in this variation
            if (variations[v].find(inconfig.m_mcgen_weightmaps_patterns[i]) != std::string::npos) {
                std::cout << "Variation "<<variations[v]<<" is a match for pattern "<<inconfig.m_mcgen_weightmaps_patterns[i]<<std::endl;
                s_formulas[v] = s_formulas[v] + "*(" + inconfig.m_mcgen_weightmaps_formulas[i]+")";
                std::cout<<" -- weight is thus "<<s_formulas[v]<<std::endl;
                std::cout<<" -- mode is "<<inconfig.m_mcgen_weightmaps_mode[i]<<std::endl;
                if(inconfig.m_mcgen_weightmaps_mode[i]=="multisim") variation_modes[v] = 0;
                if(inconfig.m_mcgen_weightmaps_mode[i]=="minmax") variation_modes[v] = 1;
            }
        }
    }

    std::vector<std::vector<std::unique_ptr<TTreeFormula>>> additional_weight_formulas(num_files);//(num_files, std::vector<std::unique_ptr<TTreeFormula>>(variations.size()));FIX why does this fail?
    for (auto& innerVector : additional_weight_formulas) {
        innerVector.resize(variations.size());
    }

    for(int fid=0; fid < num_files; ++fid) {
        files[fid]->cd();
        for(int vid = 0; vid < variations.size(); vid++){ 
            additional_weight_formulas[fid][vid] =  std::make_unique<TTreeFormula>(("weightMapsFormulas_"+std::to_string(fid)+"_"+std::to_string(vid)).c_str(), s_formulas[vid].c_str(),trees[fid]);
        }
    }

    // make a map and start filling, before filling find if already in map, if it is check size.
    std::cout <<" Found " << variations.size() << " unique variations: " << std::endl;

    //CHeck This FIX
    std::vector<int> num_universes_per_variation;
    std::map<int, std::string> map_universe_to_var;
    std::vector<int> vec_universe_to_var;
    std::map<std::string, int> map_var_to_num_universe;

    map_universe_to_var.clear();
    vec_universe_to_var.clear();
    num_universes_per_variation.clear();

    for(size_t vid=0; vid<variations.size(); ++vid) {
        const auto &v =  variations[vid];
        int max_variation_length = 0;
        int in_n_files = 0;

        //Lets loop over all trees

        for(size_t fid=0; fid<num_files; fid++){
            trees[fid]->GetEntry(good_event);

            //is this variation in this tree?
            int is_found = (*(f_weights[fid])).count(v);

            if(is_found==0){
                std::cout<<"  WARNING @  variation " <<v<<"  in File " << inconfig.m_mcgen_file_name.at(fid)<<". Variation does not exist in file! "<<std::endl;
            }else{
                int thissize = (*(f_weights[fid])).at(v).size(); // map lookup
                in_n_files++;       
                max_variation_length = std::max(thissize,max_variation_length);

            }

        }

        std::cout<<v<<" is of max length: "<<max_variation_length<<" and in "<<in_n_files<<" of "<<num_files<<" total files"<<std::endl;

        for(int p=0; p < max_variation_length; p++){
            map_universe_to_var[num_universes_per_variation.size()] = v;
            vec_universe_to_var.push_back(vid);
            num_universes_per_variation.push_back(max_variation_length);
        }
        map_var_to_num_universe[v] = max_variation_length;
    }

    std::cout << " File: 0 | " << inconfig.m_mcgen_file_name.at(0) << " has " << used_montecarlos.at(0) << " montecarlos" << std::endl;
    for(int i=1; i<num_files; i++){
        std::cout << " File: " << i <<" |  "<<inconfig.m_mcgen_file_name[i]<< " has " << used_montecarlos.at(i) << " montecarlos" << std::endl;
        if(used_montecarlos.at(i)!= used_montecarlos.at(i-1)){
            std::cerr << " Warning, number of universes for are different between files" << std::endl;
            std::cerr << " The missing universes are Set to weights of 1. Make sure this is what you want!" << std::endl;
            for(int j=0; j<num_files; j++){
                if(universes_used < used_montecarlos.at(j)) 
                    universes_used = used_montecarlos.at(j);
                std::cerr <<"File " << j << " montecarlos: " << used_montecarlos.at(j) << std::endl;
            }
        }
    }

    ///Quick check on minmax
    for(int v=0; v< variations.size(); v++){
        if(variation_modes[v]==1 && map_var_to_num_universe[variations[v]]!=2){
            std::cerr <<"ERROR! variation "<<variations[v]<<" is tagged as minmax mode, but has "<<map_var_to_num_universe[variations[v]]<<" universes (can only be 2)."<<std::endl;
            std::cout <<"ERROR! variation "<<variations[v]<<" is tagged as minmax mode, but has "<<map_var_to_num_universe[variations[v]]<<" universes (can only be 2)."<<std::endl;
            exit(EXIT_FAILURE);
        }
    }

    //But in reality we want the max universes to be the sum of all max variaitons across all files, NOT the sum over all files max variations.
    universes_used = num_universes_per_variation.size();

    std::vector<SystStruct> syst_vector;
    for(int v=0; v< variations.size(); v++){
        // Check to see if pattern is in this variation
        syst_vector.emplace_back(variations[v], map_var_to_num_universe[variations[v]], variation_modes[v], s_formulas[v]);
    }

    std::cout << " -------------------------------------------------------------" << std::endl;
    std::cout << " Initilizing " << universes_used << " universes." << std::endl;
    std::cout << " -------------------------------------------------------------" << std::endl;

    std::vector<double> base_vec (inconfig.m_num_bins_total,0.0);

    std::cout << " Full concatanated vector has : " << inconfig.m_num_bins_total << std::endl;

    std::vector<std::vector<double>> multi_vecspec; ///FIX REplace with Eigen
    multi_vecspec.clear();
    multi_vecspec.resize(universes_used,base_vec);

    std::cout << " multi_vecspec now initilized of size :" << multi_vecspec.size() << std::endl;
    std::cout << " Reading the data files" << std::endl;
    //watch.Reset();
    //watch.Start();

    for(int j=0; j < num_files; j++){
        int nevents = std::min(inconfig.m_mcgen_maxevents[j], nentries[j]);
        std::cout << " Starting @ data file=" << files[j]->GetName() <<" which has "<<nevents<<" Events. "<<std::endl;
        size_t nbytes = 0;
        for(int i=0; i < nevents; i++) {
            if(i%100==0)std::cout<<" -- uni :"<<i<<" / "<<nevents<<std::endl;
            nbytes+= trees[j]->GetEntry(i);
            ProcessEvent(inconfig, *(f_weights[j]),j,i);
            //INPUT PROCESS FIX

        } //end of entry loop
        std::cout << " nbytes read=" << nbytes << std::endl;

    } //end of file loop



    //watch.Stop();
    //std::cout << " done CpuTime=" << watch.CpuTime() << " RealTime=" << watch.RealTime() << std::endl;

    /***************************************************************
     *		Now some clean-up and Writing
     * ************************************************************/

    //FIX check?
    //NOPE In fact, calling Close() on a TFile object managed by a std::unique_ptr can lead to undefined behavior because the std::unique_ptr may delete the object before the Close() method finishes executing. So, it is best to rely on the std::unique_ptr to manage the lifetime of the TFile object and not call Close() explicitly.
    //for(auto f: files){
    //    std::cout << " TFile::Close() file=" << f->GetName() <<  std::endl;
    //    f->Close();
    //}
    std::cout << " End" << std::endl;


    return 0;
}


/*
void ProcessEvent(const PROconfig &inconfig,
        const std::map<std::string, 
        std::vector<eweight_type> >& thisfWeight,
        size_t fileid,
        int entryid) {

    double abnormally_large_weight = 1e3;//1e20;//20.0;
    double global_weight = 1.0;

    if( montecarlo_additional_weight_bool[fileid]){
        montecarlo_additional_weight_formulas[fileid]->GetNdata();
        global_weight = montecarlo_additional_weight_formulas[fileid]->EvalInstance();
    };//this will be 1.0 unless specifi
    global_weight *= montecarlo_scale[fileid];

    double additional_CV_weight = 1.0;

    const auto bnbcorr_iter = thisfWeight.find(bnbcorrection_str);
    if (bnbcorr_iter != thisfWeight.end())    additional_CV_weight *= (*bnbcorr_iter).second.front();

    if(std::isinf(global_weight) or (global_weight != global_weight)){
        std::stringstream ss;
        ss << "ProcessEvent\t||\tERROR  error @ " << entryid
            << " in File " << montecarlo_file.at(fileid) 
            << " as its either inf/nan: " << global_weight << std::endl;
        throw std::runtime_error(ss.str());
    }

    if(!EventSelection(fileid)) return;

    // precompute the weight size
    std::vector<double> weights(universes_used,global_weight);

    //Loop over all variations
    std::map<std::string, std::vector<eweight_type> >::const_iterator var_iter;
    int woffset = 0;
    int vid = 0;
    for(const auto& var : variations){


        //check if variation is in this file, if it isn't: then just push back 1's of appropiate number to keep universes consistent
        //this is of length of whatever the maximum length that was found in ANY file
        auto expected_num_universe_sz = map_var_to_num_universe.at(var); 

        //is  
        var_iter = thisfWeight.find(var);
        int quick_fix = 0;
	double indiv_variation_weight = 1.0;

        if (var_iter == thisfWeight.end()) {
            //This we need to drop this for new version (where we add 1's)
            //std::cout<<var<<" is not in this universe, adding "<<expected_num_universe_sz<<" to woffset "<<woffset<<std::endl;
            //woffset += expected_num_universe_sz;
            //continue;
        }else {
            //first one is what we expect, second is whats directly inside the map.

            if (expected_num_universe_sz != (*var_iter).second.size()) {
                std::stringstream ss;
                //std::cout<< "Number of universes is not the max in this file" << std::endl;
                //throw std::runtime_error(ss.str());

                if( (*var_iter).second.size() > expected_num_universe_sz && var_iter != thisfWeight.end()){
                    ss << ". REALLY BAD!!  iter.size() " <<(*var_iter).second.size()<<" expected "<<expected_num_universe_sz<<" on var "<<var<<std::endl;
                    throw std::runtime_error(ss.str());
                }
            }
            //so if this file contains smaller number of variations
            quick_fix = (*var_iter).second.size();
            //std::cout<< "So setting quick fix to the true number of universes in this varaiion : "<<quick_fix<< std::endl;

	    if(!montecarlo_fake[fileid]){
            	//Grab newwer variation specfic weights;
             	m_variation_weight_formulas[fileid][vid]->GetNdata();
            	indiv_variation_weight = m_variation_weight_formulas[fileid][vid]->EvalInstance();
            	if((indiv_variation_weight!= indiv_variation_weight || indiv_variation_weight <0) && !montecarlo_fake[fileid]){
		    	std::cout << "varaition: " << var << std::endl;
                    	std::cout<<"ERROR! the additional wight is nan or negative "<<indiv_variation_weight<<" Breakign!"<<std::endl;
                    	exit(EXIT_FAILURE);
            	}
  	    }
            //std::cout<<var<<" "<<indiv_variation_weight<<" "<<fileid<<" "<<vid<<std::endl;

        }

        if (woffset >= weights.size()) {
            std::stringstream ss;
            ss << "woffset=" << woffset << " weights sz=" << weights.size() << " !" << std::endl;
            throw std::runtime_error(ss.str());
        }

        for(size_t wid0 = 0, wid1 = woffset; wid1 < (woffset + expected_num_universe_sz); ++wid0, ++wid1) {
            double wei = 1.0;
            //            std::cout<<"wid0 "<<wid0<<"/ "<<expected_num_universe_sz<<"  wid1 "<<wid1<<" / "<<weights.size()<<" woffset "<<woffset<<" quick_fix "<<quick_fix<<std::endl;

            if(wid0<quick_fix){
                wei = (*var_iter).second[wid0];
            }
       
            bool is_inf = std::isinf(wei);
            bool is_nan = (wei != wei);

            if(is_inf or is_nan){
                std::stringstream ss;
                ss << "ProcessEvent\t||\t ERROR! Killing :: event # " << entryid
                    << " in File " << montecarlo_file.at(fileid) << " weight: " << wei << " global bnb: " << global_weight << " in " << var << std::endl;
                //                throw std::runtime_error(ss.str());
                wei = 1.0;                
            }

            if(wei > abnormally_large_weight){
                std::cout<<"ProcessEvent\t||\tATTENTION!! HUGE weight: "<<wei<<" at "<<var<<" event "<<entryid<<" file "<<fileid<<std::endl;
                wei=1.0;
            }
		

	    weights[wid1] *= wei*indiv_variation_weight;
        }

        woffset += expected_num_universe_sz;
        vid++;
    }//end of all variations

    if (woffset != weights.size()) {
        std::stringstream ss;
        ss << "woffset=" << woffset << " weights sz=" << weights.size() << " !" << std::endl;
        throw std::runtime_error(ss.str());
    }

    //So the size of weights must equal global universes ya?
    if(universes_used != num_universes_per_variation.size()){
        std::stringstream ss;
        ss <<otag<<" ERROR "<<std::endl;
        ss <<"weights.size() "<<weights.size()<<std::endl;
        ss <<"universes_used "<<universes_used<<std::endl;
        ss <<"multi_vecspec.size() "<<multi_vecspec.size()<<std::endl;
        ss <<"num_universes_per_variation.size() "<<num_universes_per_variation.size()<<std::endl;
        throw std::runtime_error(ss.str());
    }
    for(int t=0; t < branch_variables[fileid].size(); t++) {


        const auto branch_var_jt = branch_variables[fileid][t];
        int ih = spec_central_value.map_hist.at(branch_var_jt->associated_hist);

        branch_var_jt->GetFormula()->GetNdata();
        double reco_var = branch_var_jt->GetFormula()->EvalInstance();
        // double reco_varold = *(static_cast<double*>(branch_var_jt->GetValue()));
        int reco_bin = spec_central_value.GetGlobalBinNumber(reco_var,ih);
        spec_central_value.hist[ih].Fill(reco_var, global_weight*additional_CV_weight);
        //std::cout<<reco_var<<" "<<reco_bin<<" "<<ih<<std::endl;

        for(int m=0; m<weights.size(); m++){
            if(reco_bin<0) continue;
            multi_vecspec[m][reco_bin] += weights[m];
        }
    }

    return;*/
//}

