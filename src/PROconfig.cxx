#include "PROconfig.h"
using namespace PROfit;


PROconfig::PROconfig(const std::string &xml): 
    m_xmlname(xml), 
    m_plot_pot(1.0),
    m_num_detectors(0),
    m_num_channels(0),
    m_num_modes(0),
    m_num_bins_detector_block(0),
    m_num_bins_mode_block(0),
    m_num_bins_total(0),
    m_num_truebins_detector_block(0),
    m_num_truebins_mode_block(0),
    m_num_truebins_total(0),
    m_num_bins_detector_block_collapsed(0),
    m_num_bins_mode_block_collapsed(0),
    m_num_bins_total_collapsed(0),
    m_write_out_variation(false), 
    m_form_covariance(true),
    m_write_out_tag("UNSET_DEFAULT"),
    m_num_mcgen_files(0)
{

    LoadFromXML(m_xmlname);

    construct_collapsing_matrix();
    //A matrix for collapsing the full-vector
    //left multiply this matrix by the full-vector to get collapsed vector
    //TODO: check corr=init for multi detector


    /*collapsingVector = Eigen::MatrixXd::Zero(num_bins_total,num_bins_total_collapsed);

      for(int im = 0; im < num_modes; im++){
      for(int id =0; id < num_detectors; id++){
      int edge = id*num_bins_detector_block + num_bins_mode_block*im; // This is the starting index for this detector
      int corr = edge;
      for(int ic = 0; ic < num_channels; ic++){
      int corner=edge;
      for(int j=0; j< num_bins.at(ic); j++){
      for(int sc = 0; sc < num_subchannels.at(ic); sc++){
      int place = j+sc*num_bins[ic]+corner;
      collapsingVector(place,corr)=1;
      }
      }
      corr++;
      }
      }
      }*/

}


int PROconfig::LoadFromXML(const std::string &filename){


    //Setup TiXml documents
    tinyxml2::XMLDocument doc;
    doc.LoadFile(filename.c_str());
    bool loadOkay = !doc.ErrorID();

    bool use_universe = 1; //FIX
    try{
        if(loadOkay) log<LOG_INFO>(L"%1% || Correctly loaded and parsed XML, continuing") % __func__;
        else throw 404;    
    }
    catch (int ernum) {
        log<LOG_ERROR>(L"%1% || ERROR: Failed to load XML configuration file names %4%. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % filename.c_str();
        log<LOG_ERROR>(L"This generally means broken brackets or attribute syntax in xml itself.");
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    tinyxml2::XMLHandle hDoc(&doc);

    tinyxml2::XMLElement *pMode, *pDet, *pChan, *pPOT;



    //max subchannels 100? Can we avoid this
    m_subchannel_plotnames.resize(100);
    m_subchannel_datas.resize(100);
    m_subchannel_names.resize(100);
    m_subchannel_osc_patterns.resize(100);
    char *end;

    //Grab the first element. Note very little error checking here! make sure they exist.
    pMode = doc.FirstChildElement("mode");
    pDet =  doc.FirstChildElement("detector");
    pChan = doc.FirstChildElement("channel");
    pPOT = doc.FirstChildElement("plotpot");


    while(pPOT){
        const char* inplotpot = pPOT->Attribute("value");
        if(inplotpot){
            m_plot_pot = strtod(inplotpot,&end);
        }
        pPOT = pPOT->NextSiblingElement("plotpot");
    }


    if(!pMode){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 mode defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }else{
        while(pMode){
            // What modes are we running in (e.g nu, nu bar, horn current=XXvolts....) Can have as many as we want
            const char* mode_name= pMode->Attribute("name");
            if(mode_name==NULL){
                log<LOG_ERROR>(L"%1% || Modes need a name! Please define a name attribute for all modes. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mode_names.push_back(mode_name);
            }

            const char* mode_plotname= pMode->Attribute("plotname");
            if(mode_plotname==NULL){
                m_mode_plotnames.push_back(m_mode_names.back());
            }else{
                m_mode_plotnames.push_back(mode_plotname);
            }

            const char* mode_use = pMode->Attribute("use");
            if(mode_use == NULL || std::string(mode_use) == "true")
                m_mode_bool.push_back(true);
            else
                m_mode_bool.push_back(false);

            pMode = pMode->NextSiblingElement("mode");
            log<LOG_DEBUG>(L"%1% || Loading Mode %2%  ") % __func__ % m_mode_names.back().c_str() ;

        }
    }

    // How many detectors do we want!
    if(!pDet){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 detector defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);

    }else{

        while(pDet){

            const char* detector_name= pDet->Attribute("name");
            if(detector_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Need all detectors to have a name attribute @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_detector_names.push_back(detector_name);
            }

            const char* detector_plotname = pDet->Attribute("plotname");
            if(detector_plotname==NULL){
                m_detector_plotnames.push_back(m_detector_names.back());
            }else{
                m_detector_plotnames.push_back(detector_plotname);
            }

            const char* detector_use = pDet->Attribute("use");
            if(detector_use==NULL || std::string(detector_use) == "true")
                m_detector_bool.push_back(true);
            else
                m_detector_bool.push_back(false);

            pDet = pDet->NextSiblingElement("detector");
            log<LOG_DEBUG>(L"%1% || Loading Det %2%  ") % __func__ % m_detector_names.back().c_str();

        }
    }

    //How many channels do we want! At the moment each detector must have all channels
    int nchan = 0;
    if(!pChan){
        log<LOG_ERROR>(L"%1% || ERROR: Need at least 1 channel defined in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }else{


        while(pChan){
            // Read in how many bins this channel uses

            const char* channel_name= pChan->Attribute("name");
            if(channel_name==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: Need all channels to have names in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_channel_names.push_back(channel_name);
            }

            const char* channel_plotname= pChan->Attribute("plotname");
            if(channel_plotname==NULL){
                m_channel_plotnames.push_back(m_channel_names.back());
            }else{
                m_channel_plotnames.push_back(channel_plotname);
            }


            const char* channel_use = pChan->Attribute("use");
            if(channel_use==NULL || std::string(channel_use) == "true")
                m_channel_bool.push_back(true);
            else
                m_channel_bool.push_back(false);


            const char* channel_unit= pChan->Attribute("unit");
            if(channel_unit==NULL){
                m_channel_units.push_back("");
            }else{
                m_channel_units.push_back(channel_unit);
            }

            log<LOG_DEBUG>(L"%1% || Loading Channel %2% with   ") % __func__ % m_channel_names.back().c_str() ;


            // What are the bin edges and bin widths (bin widths just calculated from edges now)
            tinyxml2::XMLElement *pBin = pChan->FirstChildElement("bins");
            std::stringstream iss(pBin->Attribute("edges"));

            double number;
            std::vector<double> binedge;
            std::vector<double> binwidth;
            std::string binstring = "";
            while ( iss >> number ){
                binedge.push_back( number );
                binstring+=" "+std::to_string(number);
            }

            log<LOG_DEBUG>(L"%1% || Loading Bins with edges %2%  ") % __func__ % binstring.c_str();

            for(size_t b = 0; b<binedge.size()-1; b++){
                binwidth.push_back(fabs(binedge.at(b)-binedge.at(b+1)));
            }

            m_channel_num_bins.push_back(binedge.size()-1);

            m_channel_bin_edges.push_back(binedge);
            m_channel_bin_widths.push_back(binwidth);


            tinyxml2::XMLElement *pBinT = pChan->FirstChildElement("truebins");
            if(pBinT){
                const char* tmin = pBinT->Attribute("min");
                const char* tmax = pBinT->Attribute("max");
                const char* tnbins = pBinT->Attribute("nbins");
                const char* tedges =pBinT->Attribute("edges");
                if(tmin==NULL && tmax==NULL && tnbins==NULL && tedges == NULL)
                {
                    log<LOG_DEBUG>(L"%1% || This variable has a NO truth binning (or attribute min,max,nbins)  ") % __func__ ;
                    m_channel_num_truebins.push_back(0);
                    m_channel_truebin_edges.push_back(std::vector<double>());
                    m_channel_truebin_widths.push_back(std::vector<double>());
                }else{

                    log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning.   ") % __func__  ;

                    int nbinsp;
                    std::vector<double> binedge, binwidth;

                    // use edges if defined, otherwise use min-max-nbins 
                    if(tedges != NULL){

                        std::stringstream true_iss(tedges);
                        double number;
                        while ( true_iss >> number ){
                            binedge.push_back(number );
                        }

                        nbinsp = binedge.size() - 1;
                        for(int i = 0; i != nbinsp; ++i){
                            binwidth.push_back(binedge[i+1] - binedge[i]);
                        }

                    }else{
                        double minp = strtod(tmin, &end);
                        double maxp = strtod(tmax, &end);
                        nbinsp = (int)strtod(tnbins, &end);
                        double step = (maxp-minp)/(double)nbinsp;
                        for(int i=0; i<nbinsp; i++){
                            binedge.push_back(minp+i*step);
                        }
                        binedge.push_back(maxp);
                        binwidth.resize(nbinsp, step);
                        log<LOG_DEBUG>(L"%1% || This variable has a Truth Binning with min %2%, max %3% and nbins %4%   ") % __func__ % minp % maxp % nbinsp ;
                    }

                    m_channel_num_truebins.push_back(nbinsp);
                    m_channel_truebin_edges.push_back(binedge);
                    m_channel_truebin_widths.push_back(binwidth);


                }
            }else{
                m_channel_num_truebins.push_back(0);
                m_channel_truebin_edges.push_back(std::vector<double>());
                m_channel_truebin_widths.push_back(std::vector<double>());
            }

            // Now loop over all this channels subchanels. Not the names must be UNIQUE!!
            tinyxml2::XMLElement *pSubChan;
            m_subchannel_bool.push_back({});
            pSubChan = pChan->FirstChildElement("subchannel");
            while(pSubChan){

                const char* subchannel_name= pSubChan->Attribute("name");
                if(subchannel_name==NULL){
                    log<LOG_ERROR>(L"%1% || ERROR: Subchannels need a name in xml.@ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);

                }else{
                    m_subchannel_names[nchan].push_back(subchannel_name);
                    log<LOG_DEBUG>(L"%1% || Subchannel Starting:  %2%") % __func__ % m_subchannel_names.at(nchan).back().c_str() ;

                }


                const char* subchannel_plotname= pSubChan->Attribute("plotname");
                if(subchannel_plotname==NULL){
                    m_subchannel_plotnames[nchan].push_back(m_subchannel_names[nchan].back());
                }else{
                    m_subchannel_plotnames[nchan].push_back(subchannel_plotname);
                }

                const char* subchannel_use = pSubChan->Attribute("use");
                if(subchannel_use==NULL || std::string(subchannel_use) == "true")
                    m_subchannel_bool.back().push_back(true);
                else
                    m_subchannel_bool.back().push_back(false);

                const char* subchannel_data= pSubChan->Attribute("data");
                if(subchannel_data==NULL){
                    m_subchannel_datas[nchan].push_back(0);
                }else{
                    m_subchannel_datas[nchan].push_back(1);
                }

                //0 means dont oscillate, 11 means electron disapearance, -11 means antielectron dis..etc..
                if(pSubChan->Attribute("osc"))
                {
                    m_has_oscillation_patterns = true;
                    m_subchannel_osc_patterns.at(nchan).push_back(strtod(pSubChan->Attribute("osc"), &end));
                }else{
                    m_has_oscillation_patterns = false;
                    m_subchannel_osc_patterns.at(nchan).push_back(0);
                }

                log<LOG_DEBUG>(L"%1% || Subchannel %2% with and osc pattern %3% and isdata %4%") % __func__ % m_subchannel_names.at(nchan).back().c_str() % m_subchannel_osc_patterns.at(nchan).back() % m_subchannel_datas.at(nchan).back();


                pSubChan = pSubChan->NextSiblingElement("subchannel");
            }

            nchan++;
            pChan = pChan->NextSiblingElement("channel");
        }
    }//end channel loop



    //Now onto mcgen, for CV specs or for covariance generation
    tinyxml2::XMLElement *pMC, *pWeiMaps, *pList, *pSpec, *pShapeOnlyMap;
    pMC   = doc.FirstChildElement("MCFile");
    pWeiMaps = doc.FirstChildElement("WeightMaps");
    pList = doc.FirstChildElement("variation_list");
    pSpec = doc.FirstChildElement("varied_spectrum");
    pShapeOnlyMap = doc.FirstChildElement("ShapeOnlyUncertainty");



    if(pMC){//Skip if not in XML
        while(pMC)
        {
            const char* tree = pMC->Attribute("treename");
            if(tree==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: You must have an associated root TTree name for all MonteCarloFile tags.. eg. treename='events' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mcgen_tree_name.push_back(tree);
            }

            const char* file = pMC->Attribute("filename");
            if(file==NULL){
                log<LOG_ERROR>(L"%1% || ERROR: You must have an associated root TFile name for all MonteCarloFile tags.. eg. filename='my.root' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                log<LOG_ERROR>(L"Terminating.");
                exit(EXIT_FAILURE);
            }else{
                m_mcgen_file_name.push_back(file);
            }


            const char* maxevents = pMC->Attribute("maxevents");
            if(maxevents==NULL){
                m_mcgen_maxevents.push_back(1e16);
            }else{
                m_mcgen_maxevents.push_back(strtod(maxevents,&end) );
            }

            //arbitray scaling you can have
            const char* scale = pMC->Attribute("scale");
            if(scale==NULL){
                m_mcgen_scale.push_back(1.0);
            }else{
                m_mcgen_scale.push_back(strtod(scale,&end) );
            }

            const char* inpot = pMC->Attribute("pot");
            if(inpot==NULL){
                m_mcgen_pot.push_back(-1.0);
            }else{
                m_mcgen_pot.push_back(strtod(inpot,&end) );
            }

            //Is this useful? 
            const char* isfake = pMC->Attribute("fake");
            if(isfake==NULL){
                m_mcgen_fake.push_back(false);
            }else{
                m_mcgen_fake.push_back(true);
            }

            log<LOG_DEBUG>(L"%1% || MultisimFile %2%, treename: %3%  ") % __func__ % m_mcgen_file_name.back().c_str() % m_mcgen_tree_name.back().c_str();


            //Here we can grab some friend tree information
            tinyxml2::XMLElement *pFriend;
            pFriend = pMC->FirstChildElement("friend");
            while(pFriend){


                std::string ffname;
                const char* friend_filename = pFriend->Attribute("filename");
                if(friend_filename==NULL){
                    ffname = m_mcgen_file_name.back();
                }else{
                    ffname = friend_filename;
                }


                m_mcgen_file_friend_treename_map[m_mcgen_file_name.back()].push_back( pFriend->Attribute("treename") );
                m_mcgen_file_friend_map[m_mcgen_file_name.back()].push_back(ffname);

                pFriend = pFriend->NextSiblingElement("friend");
            }//END of friend loop


            tinyxml2::XMLElement *pBranch;
            pBranch = pMC->FirstChildElement("branch");


            std::vector<bool> TEMP_additional_weight_bool;
            std::vector<std::string> TEMP_additional_weight_name;
            std::vector<std::string> TEMP_eventweight_branch_names;
            std::vector<std::shared_ptr<BranchVariable>> TEMP_branch_variables;
            while(pBranch){

                const char* bnam = pBranch->Attribute("name");
                const char* btype = pBranch->Attribute("type");
                const char* bhist = pBranch->Attribute("associated_subchannel");
                const char* bsyst = pBranch->Attribute("associated_systematic");
                const char* bcentral = pBranch->Attribute("central_value");
                const char* bwname = pBranch->Attribute("eventweight_branch_name");
                const char* badditional_weight = pBranch->Attribute("additional_weight");

                if(bwname== NULL){
                    log<LOG_WARNING>(L"%1% || WARNING: No eventweight branch name passed, defaulting to 'weights' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    TEMP_eventweight_branch_names.push_back("weights");
                }else{
                    log<LOG_DEBUG>(L"%1% || Setting eventweight branch name %2%") %__func__ % bnam;
                    TEMP_eventweight_branch_names.push_back(std::string(bwname));
                }

                if(bnam == NULL){
                    log<LOG_ERROR>(L"%1% || ERROR!: Each branch must include the name of the branch to use. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"%1% || ERROR!: e.g name = 'ereco' @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }

                if(btype == NULL){
                    log<LOG_WARNING>(L"%1% || WARNING: No branch type has been specified, assuming double. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    btype= "double";
                }

                if(bhist == NULL){
                    log<LOG_ERROR>(L"%1% || Each branch must have an associated_subchannel to fill! On branch %4% : @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % bnam;
                    log<LOG_ERROR>(L"%1% || e.g associated_subchannel='mode_det_chan_subchannel ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }


                if(bsyst == NULL){
                    if(use_universe == false){
                        log<LOG_WARNING>(L"%1% || WARNING: No root file with unique systematic variation is provided ") % __func__;
                        log<LOG_ERROR>(L"%1% || ERROR! please provide what systematic variation this file correpsonds to!") % __func__;
                        log<LOG_ERROR>(L"Terminating.");
                        exit(EXIT_FAILURE);
                    }
                    systematic_name.push_back("");
                }else{
                    systematic_name.push_back(bsyst);	

                }


                if(badditional_weight == NULL){
                    TEMP_additional_weight_bool.push_back(0);
                    TEMP_additional_weight_name.push_back("");
                }else{
                    TEMP_additional_weight_name.push_back(badditional_weight);
                    TEMP_additional_weight_bool.push_back(1);
                    log<LOG_DEBUG>(L"%1% || Setting an additional weight for branch %2% using the branch %3% as a reweighting.") % __func__ % bnam %badditional_weight;

                }



                if((std::string)btype == "double"){
                    if(use_universe){
                        TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, btype, bhist ) );
                    } else  if((std::string)bcentral == "true"){
                        TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, btype, bhist,bsyst, true) );
                        log<LOG_DEBUG>(L"%1% || Setting as  CV for det sys.") % __func__ ;
                    } else {
                        TEMP_branch_variables.push_back( std::make_shared<BranchVariable>(bnam, btype, bhist,bsyst, false) );
                        log<LOG_DEBUG>(L"%1% || Setting as individual (not CV) for det sys.") % __func__ ;
                    }
                }else{
                    log<LOG_ERROR>(L"%1% || ERROR: currently only double, allowed for input branch variables (sorry!) i @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % bnam;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }

                std::string oscillate = "false";
                if(pBranch->Attribute("oscillate")!=NULL){
                    oscillate =pBranch->Attribute("oscillate");
                }	

                if(oscillate == "false"){
                    log<LOG_DEBUG>(L"%1% || Oscillations are OFF ") % __func__ ;
                    TEMP_branch_variables.back()->SetOscillate(false);
                }else if(oscillate == "true"){
                    log<LOG_DEBUG>(L"%1% || Oscillations are Set to  ON ") % __func__;
                    TEMP_branch_variables.back()->SetOscillate(true);
                    TEMP_branch_variables.back()->SetTrueParam(pBranch->Attribute("true_param_name"));
                    TEMP_branch_variables.back()->SetPDG(pBranch->Attribute("pdg_name"));
                    if(pBranch->Attribute("true_L_name") != NULL){
                        //for oscillation that needs both E and L
                        TEMP_branch_variables.back()->SetTrueL(pBranch->Attribute("true_L_name"));
                        log<LOG_DEBUG>(L"%1% || Oscillations using true param name:   %2% and baseline %3% ") % __func__ % pBranch->Attribute("true_param_name") % pBranch->Attribute("true_L_name") ;
                    }else{
                        //for oscillations that only needs E, such as an energy-dependent scaling for single photon NCpi0!
                        log<LOG_DEBUG>(L"%1% || Oscillations using  Energy only dependent oscillation ( or shift/normalization)  %2% ") % __func__ % pBranch->Attribute("true_param_name") ;
                    }
                }else{
                    log<LOG_DEBUG>(L"%1% || Do Not Oscillate  ") % __func__  ;
                    TEMP_branch_variables.back()->SetOscillate(false);
                }

                log<LOG_DEBUG>(L"%1% || Associated subchannel: %2% ") % __func__ % bhist;

                pBranch = pBranch->NextSiblingElement("branch");
            }
            m_mcgen_additional_weight_name.push_back(TEMP_additional_weight_name);
            m_mcgen_additional_weight_bool.push_back(TEMP_additional_weight_bool);
            m_branch_variables.push_back(TEMP_branch_variables);
            m_mcgen_eventweight_branch_names.push_back(TEMP_eventweight_branch_names);
            //next file
            pMC=pMC->NextSiblingElement("MCFile");
        }
    }

    if(!pList){
        log<LOG_DEBUG>(L"%1% || No Allowlist or Denylist set, including ALL variations by default.") % __func__  ;
    }else{
        while(pList){

            tinyxml2::XMLElement *pAllowList = pList->FirstChildElement("allowlist");
            while(pAllowList){
                std::string wt = std::string(pAllowList->GetText());
                m_mcgen_variation_allowlist.insert(wt); 
                log<LOG_DEBUG>(L"%1% || Allowlisting variations: %2%") % __func__ % wt.c_str() ;
                pAllowList = pAllowList->NextSiblingElement("allowlist");
            }

            tinyxml2::XMLElement *pDenyList = pList->FirstChildElement("denylist");
            while(pDenyList){
                std::string bt = std::string(pDenyList->GetText());
                m_mcgen_variation_denylist.insert(bt); 
                log<LOG_DEBUG>(L"%1% || Denylisting variations: %2%") % __func__ % bt.c_str() ;
                pDenyList = pDenyList->NextSiblingElement("denylist");
            }
            pList = pList->NextSiblingElement("variation_list");
        }
    }
    //weightMaps
    if(!pWeiMaps){
        log<LOG_DEBUG>(L"%1% || WeightMaps not set, all weights for all variations are 1 (individual branch weights still apply)") % __func__  ;
    }else{
        while(pWeiMaps){


            tinyxml2::XMLElement *pVariation;
            pVariation = pWeiMaps->FirstChildElement("variation");

            while(pVariation){


                const char* w_pattern = pVariation->Attribute("pattern");
                const char* w_formula = pVariation->Attribute("weight_formula");
                const char* w_use = pVariation->Attribute("use");
                const char* w_mode = pVariation->Attribute("mode");

                if(w_pattern== NULL){
                    log<LOG_ERROR>(L"%1% || ERROR! No pattern passed for this variation in WeightMaps. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__;
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                }else{
                    log<LOG_DEBUG>(L"%1% || Loading WeightMaps Variation Pattern: %2%") %__func__ % w_pattern;
                    m_mcgen_weightmaps_patterns.push_back(std::string(w_pattern));
                }


                if(w_formula== NULL){
                    log<LOG_WARNING>(L"%1% || Warning, No formula passed for this variation in WeightMaps. Setting to 1. Make sure this is wanted behaviour.") %__func__ ;
                    m_mcgen_weightmaps_formulas.push_back("1");
                }else{
                    log<LOG_DEBUG>(L"%1% || Loading WeightMaps Variation Formula: %2%") %__func__ % w_formula;
                    m_mcgen_weightmaps_formulas.push_back(std::string(w_formula));
                }

                if(w_use== NULL || std::string(w_use) == "true"){
                    m_mcgen_weightmaps_uses.push_back(true);
                }else{
                    m_mcgen_weightmaps_uses.push_back(false);
                }

                if(w_mode== NULL){
                    log<LOG_WARNING>(L"%1% || Warning, No mode passed for this variaiton in  WeightMaps. Assuming default multisim.  Make sure this is wanted behaviour.") %__func__ ;
                    m_mcgen_weightmaps_mode.push_back("multisim");
                }else{

                    std::string mode = std::string(w_mode);
                    if(mode=="multisim" || mode=="minmax"){
                        m_mcgen_weightmaps_mode.push_back(mode);
                    }else{
                        log<LOG_ERROR>(L"%1% || ERROR! The mode passed in is %4% but only allowed is multisim or minmax. @ line %2% in %3% ") % __func__ % __LINE__  % __FILE__ % w_mode;
                        log<LOG_ERROR>(L"Terminating.");
                        exit(EXIT_FAILURE);
                    }
                }

                pVariation = pVariation->NextSiblingElement("variation");
            }

            pWeiMaps=pWeiMaps->NextSiblingElement("WeightMaps");
        }
    }



    while(pShapeOnlyMap){

        log<LOG_WARNING>(L"%1% || Warning!  Setting up for shape-only covariance matrix generation. MAKE SURE this is what you want if you're generating covariance matrix!!!") % __func__;

        std::string pshapeonly_systematic_name = std::string(pShapeOnlyMap->Attribute("name"));
        const char* pshapeonly_systematic_use = pShapeOnlyMap->Attribute("use");
        bool pshapeonly_systematic_use_bool = true;

        if(pshapeonly_systematic_use == NULL || std::string(pshapeonly_systematic_use) == "true"){
            std::cout << "" << pshapeonly_systematic_name << std::endl;
            log<LOG_DEBUG>(L"%1% || Setting up shape-only covariance matrix for systematic: %2% ") % __func__ % pshapeonly_systematic_name.c_str();

        }else if(std::string(pshapeonly_systematic_use) == "false"){
            log<LOG_DEBUG>(L"%1% || Setting up shape-only covariance matrix for systematic: %2% ? False ") % __func__ % pshapeonly_systematic_name.c_str();
            pshapeonly_systematic_use_bool = false;
        }else{
            log<LOG_WARNING>(L"%1% || INVALID argument received for Attribute use of ShapeOnlyUncertainty element for systematic: %2% . Default it to true ") % __func__ % pshapeonly_systematic_name.c_str();
        }

        tinyxml2::XMLElement *pSubchannel;
        pSubchannel = pShapeOnlyMap->FirstChildElement("subchannel");	

        while(pshapeonly_systematic_use_bool && pSubchannel){

            std::string pshapeonly_subchannel_name = std::string(pSubchannel->Attribute("name"));
            std::string pshapeonly_subchannel_use = std::string(pSubchannel->Attribute("use"));

            if(pshapeonly_subchannel_use == "false" ){
                log<LOG_DEBUG>(L"%1% || Not include subchannel: %2% for shape-only covariance matrix") % __func__ % pshapeonly_subchannel_name.c_str();
            }else{
                log<LOG_DEBUG>(L"%1% || Include subchannel: %2% for shape-only covariance matrix") % __func__ % pshapeonly_subchannel_name.c_str();
                m_mcgen_shapeonly_listmap[pshapeonly_systematic_name].push_back(pshapeonly_subchannel_name);
            }

            pSubchannel = pSubchannel->NextSiblingElement("subchannel");
        }

        pShapeOnlyMap = pShapeOnlyMap->NextSiblingElement("ShapeOnlyUncertainty");

    }


    while(pSpec){
        const char* swrite_out = pSpec->Attribute("writeout");
        const char* swrite_out_tag = pSpec->Attribute("writeout_tag");
        const char* sform_matrix = pSpec->Attribute("form_matrix");	

        if( std::string(swrite_out) == "true"){
            m_write_out_variation = true;
            log<LOG_DEBUG>(L"%1% || Setting up to write out spectra for variations") % __func__;
        }

        if(m_write_out_variation){
            if(swrite_out_tag) 
                m_write_out_tag = std::string(swrite_out_tag);
        }

        if( std::string(sform_matrix) == "false"){
            m_form_covariance = false;
            log<LOG_DEBUG>(L"%1% || Explicitly to ask to not generate covariance matrix") % __func__;
        }
        pSpec = pSpec->NextSiblingElement("varied_spectrum");
    }


    this->CalcTotalBins();


    log<LOG_INFO>(L"%1% || Checking number of Mode/Detector/Channel/Subchannels and BINs") % __func__;
    log<LOG_INFO>(L"%1% || num_modes: %2% ") % __func__ % m_num_modes;
    log<LOG_INFO>(L"%1% || num_detectors: %2% ") % __func__ % m_num_detectors;
    log<LOG_INFO>(L"%1% || num_channels: %2% ") % __func__ % m_num_channels;
    for(int i = 0 ; i!=m_num_channels; ++i){
        log<LOG_INFO>(L"%1% || num of subchannels: %2% ") % __func__ % m_num_subchannels[i];
        log<LOG_INFO>(L"%1% || num of bins: %2% ") % __func__ % m_channel_num_bins[i];

    }
    log<LOG_INFO>(L"%1% || num_bins_detector_block: %2%") % __func__ % m_num_bins_detector_block;
    log<LOG_INFO>(L"%1% || num_truebins_detector_block: %2%") % __func__ % m_num_truebins_detector_block;
    log<LOG_INFO>(L"%1% || num_bins_detector_block_collapsed: %2%") % __func__ % m_num_bins_detector_block_collapsed;
    log<LOG_INFO>(L"%1% || num_bins_mode_block: %2%") % __func__ % m_num_bins_mode_block;
    log<LOG_INFO>(L"%1% || num_true_bins_mode_block: %2%") % __func__ % m_num_truebins_mode_block;
    log<LOG_INFO>(L"%1% || num_bins_mode_block_collapsed: %2%") % __func__ % m_num_bins_mode_block_collapsed;
    log<LOG_INFO>(L"%1% || num_bins_total: %2%") % __func__ % m_num_bins_total;
    log<LOG_INFO>(L"%1% || num_true_bins_total: %2%") % __func__ % m_num_truebins_total;
    log<LOG_INFO>(L"%1% || num_bins_total_collapsed: %2%") % __func__ % m_num_bins_total_collapsed;


    log<LOG_INFO>(L"%1% || Done reading the xmls") % __func__;
    return 0;
}


void PROconfig::CalcTotalBins(){
    this->remove_unused_channel();

    log<LOG_INFO>(L"%1% || Calculating number of bins involved") % __func__;
    for(int i = 0; i != m_num_channels; ++i){
        m_num_bins_detector_block += m_num_subchannels[i]*m_channel_num_bins[i];
        m_num_truebins_detector_block += m_num_subchannels[i]*m_channel_num_truebins[i];
        m_num_bins_detector_block_collapsed += m_channel_num_bins[i];
    }

    m_num_bins_mode_block = m_num_bins_detector_block *  m_num_detectors;
    m_num_truebins_mode_block = m_num_truebins_detector_block *  m_num_detectors;
    m_num_bins_mode_block_collapsed = m_num_bins_detector_block_collapsed * m_num_detectors;

    m_num_bins_total = m_num_bins_mode_block * m_num_modes;
    m_num_truebins_total = m_num_truebins_mode_block * m_num_modes;
    m_num_bins_total_collapsed = m_num_bins_mode_block_collapsed * m_num_modes;

    this->generate_index_map();
    return;
}

int PROconfig::GetSubchannelIndex(const std::string& fullname) const{
    auto pos_iter = m_map_fullname_subchannel_index.find(fullname);
    if(pos_iter == m_map_fullname_subchannel_index.end()){
        log<LOG_ERROR>(L"%1% || Subchannel name: %2% does not exist in the indexing map!") % __func__ % fullname.c_str();
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return pos_iter->second;
}

int PROconfig::GetChannelIndex(int subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_channel_index[index];
}

int PROconfig::GetGlobalBinStart(int subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_reco_index_start[index];
}

int PROconfig::GetGlobalTrueBinStart(int subchannel_index) const{
    size_t index = this->find_equal_index(m_vec_subchannel_index, subchannel_index);
    return m_vec_global_true_index_start[index];
}

int PROconfig::GetSubchannelIndexFromGlobalBin(int global_reco_index) const {
    size_t index = this->find_less_or_equal_index(m_vec_global_reco_index_start, global_reco_index); 
    return m_vec_subchannel_index[index];
}

int PROconfig::GetSubchannelIndexFromGlobalTrueBin(int global_trueindex) const{
    size_t index = this->find_less_or_equal_index(m_vec_global_true_index_start, global_trueindex);
    return m_vec_subchannel_index[index];
}

const std::vector<double>& PROconfig::GetChannelBinEdges(int channel_index) const{

    if(channel_index < 0 || channel_index >= m_num_channels){
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % m_num_channels;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    return m_channel_bin_edges[channel_index];
}

int PROconfig::GetChannelNTrueBins(int channel_index) const{
    if(channel_index < 0 || channel_index >= m_num_channels){
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % m_num_channels;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    return m_channel_num_truebins[channel_index];
}

const std::vector<double>& PROconfig::GetChannelTrueBinEdges(int channel_index) const{

    //check for out of bound
    if(channel_index < 0 || channel_index >= m_num_channels){
        log<LOG_ERROR>(L"%1% || Given channel index: %2% is out of bound") % __func__ % channel_index;
        log<LOG_ERROR>(L"%1% || Total number of channels : %2%") % __func__ % m_num_channels;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    return m_channel_truebin_edges[channel_index];
}
//------------ Start of private function ------------------
//------------ Start of private function ------------------
//------------ Start of private function ------------------

void PROconfig::remove_unused_channel(){

    log<LOG_INFO>(L"%1% || Remove any used channels and subchannels...") % __func__;

    m_num_modes = std::count(m_mode_bool.begin(), m_mode_bool.end(), true);
    m_num_detectors = std::count(m_detector_bool.begin(), m_detector_bool.end(), true);
    m_num_channels = std::count(m_channel_bool.begin(), m_channel_bool.end(), true);

    //update mode-info
    if(m_num_modes != (int)m_mode_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused modes!! Clean it up...") % __func__;
        std::vector<std::string> temp_mode_names(m_num_modes), temp_mode_plotnames(m_num_modes);
        for(size_t i = 0, mode_index = 0; i != m_mode_bool.size(); ++i){
            if(m_mode_bool[i]){
                temp_mode_names[mode_index] = m_mode_names[i];
                temp_mode_plotnames[mode_index] = m_mode_plotnames[i];

                ++mode_index;
            }    
        }
        m_mode_names = temp_mode_names;
        m_mode_plotnames = temp_mode_plotnames;
    }

    ///update detector-info
    if(m_num_detectors != (int)m_detector_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused detectors!! Clean it up...") % __func__;
        std::vector<std::string> temp_detector_names(m_num_detectors), temp_detector_plotnames(m_num_detectors);
        for(size_t i = 0, det_index = 0; i != m_detector_bool.size(); ++i){
            if(m_detector_bool[i]){
                temp_detector_names[det_index] = m_detector_names[i];
                temp_detector_plotnames[det_index] = m_detector_plotnames[i];

                ++det_index;
            }
        }
        m_detector_names = temp_detector_names;
        m_detector_plotnames = temp_detector_plotnames;
    }

    if(m_num_channels != (int)m_channel_bool.size()){
        log<LOG_DEBUG>(L"%1% || Found unused channels!! Clean the messs up...") % __func__;

        //update channel-related info
        std::vector<int> temp_channel_num_bins(m_num_channels, 0);
        std::vector<std::vector<double>> temp_channel_bin_edges(m_num_channels, std::vector<double>());
        std::vector<std::vector<double>> temp_channel_bin_widths(m_num_channels, std::vector<double>());

        std::vector<int> temp_channel_num_truebins(m_num_channels, 0);
        std::vector<std::vector<double>> temp_channel_truebin_edges(m_num_channels, std::vector<double>());
        std::vector<std::vector<double>> temp_channel_truebin_widths(m_num_channels, std::vector<double>());

        std::vector<std::string> temp_channel_names(m_num_channels);
        std::vector<std::string> temp_channel_plotnames(m_num_channels);
        std::vector<std::string> temp_channel_units(m_num_channels);
        for(size_t i=0, chan_index = 0; i< m_channel_bool.size(); ++i){
            if(m_channel_bool[i]){
                temp_channel_num_bins[chan_index] = m_channel_num_bins[i];
                temp_channel_bin_edges[chan_index] = m_channel_bin_edges[i];
                temp_channel_bin_widths[chan_index] = m_channel_bin_widths[i];

                temp_channel_num_truebins[chan_index] = m_channel_num_truebins[i];
                temp_channel_truebin_edges[chan_index] = m_channel_truebin_edges[i];
                temp_channel_truebin_widths[chan_index] = m_channel_truebin_widths[i];

                temp_channel_names[chan_index] = m_channel_names[i];
                temp_channel_plotnames[chan_index] = m_channel_plotnames[i];
                temp_channel_units[chan_index] = m_channel_units[i];

                ++chan_index;
            }
        }

        m_channel_num_bins = temp_channel_num_bins;
        m_channel_bin_edges = temp_channel_bin_edges;
        m_channel_bin_widths = temp_channel_bin_widths;
        m_channel_num_truebins = temp_channel_num_truebins;
        m_channel_truebin_edges = temp_channel_truebin_edges;
        m_channel_truebin_widths = temp_channel_truebin_widths;
        m_channel_names = temp_channel_names;
        m_channel_plotnames = temp_channel_plotnames;
        m_channel_units = temp_channel_units;
    }

    {

        //update subchannel-related info
        m_num_subchannels.resize(m_num_channels);
        std::vector<std::vector<std::string >> temp_subchannel_names(m_num_channels), temp_subchannel_plotnames(m_num_channels);
        std::vector<std::vector<int >> temp_subchannel_datas(m_num_channels), temp_subchannel_osc_patterns(m_num_channels);
        for(size_t i=0, chan_index = 0; i< m_channel_bool.size(); ++i){
            if(m_channel_bool.at(i)){
                m_num_subchannels[chan_index]= 0;
                for(size_t j=0; j< m_subchannel_bool[i].size(); ++j){ 
                    if(m_subchannel_bool[i][j]){
                        ++m_num_subchannels[chan_index];
                        temp_subchannel_names[chan_index].push_back(m_subchannel_names[i][j]);
                        temp_subchannel_plotnames[chan_index].push_back(m_subchannel_plotnames[i][j]);	
                        temp_subchannel_datas[chan_index].push_back(m_subchannel_datas[i][j]);
                        temp_subchannel_osc_patterns[chan_index].push_back(m_subchannel_osc_patterns[i][j]);

                    }
                }


                ++chan_index;
            }
        }

        m_subchannel_names = temp_subchannel_names;
        m_subchannel_plotnames = temp_subchannel_plotnames;
        m_subchannel_datas = temp_subchannel_datas;
        m_subchannel_osc_patterns = temp_subchannel_osc_patterns;

    }

    //grab list of fullnames used.
    log<LOG_DEBUG>(L"%1% || Sweet, now generating fullnames of all channels used...") % __func__;
    m_fullnames.clear();
    for(int im = 0; im < m_num_modes; im++){
        for(int id =0; id < m_num_detectors; id++){
            for(int ic = 0; ic < m_num_channels; ic++){
                for(int sc = 0; sc < m_num_subchannels.at(ic); sc++){

                    std::string temp_name  = m_mode_names.at(im) +"_" +m_detector_names.at(id)+"_"+m_channel_names.at(ic)+"_"+m_subchannel_names.at(ic).at(sc);
                    log<LOG_INFO>(L"%1% || fullname of subchannel: %2% ") % __func__ % temp_name.c_str();
                    m_fullnames.push_back(temp_name);
                }
            }
        }
    }

    this->remove_unused_files();
    return;
}


void PROconfig::remove_unused_files(){


    //ignore any files not associated with used channels 
    //clean up branches not associated with used channels 
    size_t num_all_branches = 0;
    for(auto& br : m_branch_variables)
        num_all_branches += br.size();

    log<LOG_DEBUG>(L"%1% || Deubg: BRANCH VARIABLE size: %2% ") % __func__ % m_branch_variables.size();;
    log<LOG_DEBUG>(L"%1% || Check for any files associated with unused subchannels ....") % __func__;
    log<LOG_DEBUG>(L"%1% || Total number of %2% active subchannels..") % __func__ % m_fullnames.size();
    log<LOG_DEBUG>(L"%1% || Total number of %2% branches listed in the xml....") % __func__ % num_all_branches;

    //update file info
    //loop over all branches, and ignore ones not used  
    if(num_all_branches != m_fullnames.size()){

        std::unordered_set<std::string> set_all_names(m_fullnames.begin(), m_fullnames.end());

        std::vector<std::string> temp_tree_name;
        std::vector<std::string> temp_file_name;
        std::vector<long int> temp_maxevents;
        std::vector<double> temp_pot;
        std::vector<double> temp_scale;
        std::vector<bool> temp_fake;
        std::map<std::string,std::vector<std::string>> temp_file_friend_map;
        std::map<std::string,std::vector<std::string>> temp_file_friend_treename_map;
        std::vector<std::vector<std::string>> temp_additional_weight_name;
        std::vector<std::vector<bool>> temp_additional_weight_bool;
        std::vector<std::vector<std::shared_ptr<BranchVariable>>> temp_branch_variables;
        std::vector<std::vector<std::string>> temp_eventweight_branch_names;

        for(size_t i = 0; i != m_mcgen_file_name.size(); ++i){
            log<LOG_DEBUG>(L"%1% || Check on @%2% th file: %3%...") % __func__ % i % m_mcgen_file_name[i].c_str();
            bool this_file_needed = false;

            std::vector<std::string> this_file_additional_weight_name;
            std::vector<bool> this_file_additional_weight_bool;
            std::vector<std::shared_ptr<BranchVariable>> this_file_branch_variables;
            std::vector<std::string> this_file_eventweight_branch_names;
            for(size_t j = 0; j != m_branch_variables[i].size(); ++j){

                if(set_all_names.find(m_branch_variables[i][j]->associated_hist) == set_all_names.end()){
                }else{

                    set_all_names.erase(m_branch_variables[i][j]->associated_hist);
                    this_file_needed = true;

                    this_file_additional_weight_name.push_back(m_mcgen_additional_weight_name[i][j]);
                    this_file_additional_weight_bool.push_back(m_mcgen_additional_weight_bool[i][j]);
                    this_file_branch_variables.push_back(m_branch_variables[i][j]);
                    this_file_eventweight_branch_names.push_back(m_mcgen_eventweight_branch_names[i][j]);
                }
            }

            if(this_file_needed){
                log<LOG_DEBUG>(L"%1% || This file is active, keep it!") % __func__ ;
                temp_tree_name.push_back(m_mcgen_tree_name[i]);
                temp_file_name.push_back(m_mcgen_file_name[i]);
                temp_maxevents.push_back(m_mcgen_maxevents[i]);
                temp_pot.push_back(m_mcgen_pot[i]);
                temp_scale.push_back(m_mcgen_scale[i]);
                temp_fake.push_back(m_mcgen_fake[i]);
                temp_file_friend_map[m_mcgen_file_name[i]] = m_mcgen_file_friend_map[m_mcgen_file_name[i]];		
                temp_file_friend_treename_map[m_mcgen_file_name[i]] = m_mcgen_file_friend_treename_map[m_mcgen_file_name[i]];

                temp_additional_weight_name.push_back(this_file_additional_weight_name);
                temp_additional_weight_bool.push_back(this_file_additional_weight_bool);
                temp_branch_variables.push_back(this_file_branch_variables);
                temp_eventweight_branch_names.push_back(this_file_eventweight_branch_names);
            }
        }

        m_mcgen_file_name = temp_file_name;
        m_mcgen_tree_name = temp_tree_name;
        m_mcgen_maxevents = temp_maxevents;
        m_mcgen_pot = temp_pot;
        m_mcgen_scale = temp_scale;
        m_mcgen_fake = temp_fake;
        m_mcgen_file_friend_map =temp_file_friend_map;
        m_mcgen_file_friend_treename_map = temp_file_friend_treename_map;
        m_mcgen_additional_weight_name = temp_additional_weight_name;
        m_mcgen_additional_weight_bool = temp_additional_weight_bool;
        m_branch_variables = temp_branch_variables;
        m_mcgen_eventweight_branch_names = temp_eventweight_branch_names;
    }

    m_num_mcgen_files = m_mcgen_file_name.size();
    log<LOG_DEBUG>(L"%1% || Finish cleaning up, total of %2% files left.") % __func__ % m_num_mcgen_files;
    return;
}

size_t PROconfig::find_equal_index(const std::vector<int>& input_vec, int val) const{
    auto pos_iter = std::lower_bound(input_vec.begin(), input_vec.end(), val);
    if(pos_iter == input_vec.end() || (*pos_iter) != val){
	log<LOG_ERROR>(L"%1% || Input value: %2% does not exist in the vector! Max element available: %3%") % __func__ % val % input_vec.back();
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }
    size_t index = pos_iter - input_vec.begin();
    return index;
}


size_t PROconfig::find_less_or_equal_index(const std::vector<int>& input_vec, int val) const{
    auto pos_iter = std::lower_bound(input_vec.begin(), input_vec.end(), val);
    if(pos_iter == input_vec.end() || (*pos_iter) != val){
        return pos_iter - input_vec.begin() -1;
    }
    else
	return pos_iter - input_vec.begin();
    return -1;
}


void PROconfig::generate_index_map(){
    log<LOG_INFO>(L"%1% || Generate map between subchannel and global indices..") % __func__;
    m_map_fullname_subchannel_index.clear();
    m_vec_subchannel_index.clear();
    m_vec_channel_index.clear();
    m_vec_global_reco_index_start.clear();
    m_vec_global_true_index_start.clear();


    int global_subchannel_index = 0;
    for(int im = 0; im < m_num_modes; im++){

        int mode_bin_start = im*m_num_bins_mode_block;
        int mode_truebin_start = im*m_num_truebins_mode_block;

        for(int id =0; id < m_num_detectors; id++){

            int detector_bin_start = id*m_num_bins_detector_block;
            int channel_bin_start = 0;

            int detector_truebin_start = id*m_num_truebins_detector_block;
            int channel_truebin_start = 0;


            for(int ic = 0; ic < m_num_channels; ic++){
                for(int sc = 0; sc < m_num_subchannels[ic]; sc++){

                    std::string temp_name  = m_mode_names[im] +"_" +m_detector_names[id]+"_"+m_channel_names[ic]+"_"+m_subchannel_names[ic][sc];
                    int global_bin_index = mode_bin_start + detector_bin_start + channel_bin_start + sc*m_channel_num_bins[ic];
                    int global_truebin_index = mode_truebin_start + detector_truebin_start + channel_truebin_start + sc*m_channel_num_truebins[ic];

                    m_map_fullname_subchannel_index[temp_name] = global_subchannel_index;
		    m_vec_subchannel_index.push_back(global_subchannel_index);
		    m_vec_channel_index.push_back(ic);
		    m_vec_global_reco_index_start.push_back(global_bin_index);
		    m_vec_global_true_index_start.push_back(global_truebin_index);

                    ++global_subchannel_index;
                }
                channel_bin_start += m_channel_num_bins[ic]*m_num_subchannels[ic];
                channel_truebin_start += m_channel_num_truebins[ic]*m_num_subchannels[ic];
            }
        }
    }
    return;
}

int PROconfig::find_global_subchannel_index_from_global_bin(int global_index, const std::vector<int>& num_subchannel_in_channel, const std::vector<int>& num_bins_in_channel, int num_channels, int num_bins_total) const{

    //check for out of bound
    if(global_index <0 || global_index >= num_bins_total){
        log<LOG_ERROR>(L"%1% || Given index: %2% is out of bound") % __func__ % global_index;
        log<LOG_ERROR>(L"%1% || Total number of bins : %2%") % __func__ % num_bins_total;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    // get number of bins per detector block 
    int num_bins_per_detector_block = 0;
    for( int ic = 0; ic != num_channels; ++ic)
        num_bins_per_detector_block += num_subchannel_in_channel[ic] * num_bins_in_channel[ic];
    if(num_bins_per_detector_block == 0){
        log<LOG_ERROR>(L"%1% || There is zero bins for each detector!! Provided global bin index: %2% ") % __func__ % global_index;
        log<LOG_ERROR>(L"Terminating.");
        exit(EXIT_FAILURE);
    }

    // get number of subchannels in detector block  
    int num_subchannel_in_detector_block = std::accumulate(num_subchannel_in_channel.begin(), num_subchannel_in_channel.end(), 0);
    int subchannel_index = (global_index / num_bins_per_detector_block) * num_subchannel_in_detector_block;

    global_index %= num_bins_per_detector_block;   //get the index inside a block 
    //check for each channel
    for( int ic = 0; ic != num_channels; ++ic){
        int total_bins_in_channel = num_subchannel_in_channel[ic] * num_bins_in_channel[ic];
        if(global_index >= total_bins_in_channel){
            global_index -= total_bins_in_channel;
            subchannel_index += num_subchannel_in_channel[ic];
        }
        else{
            subchannel_index += global_index / num_bins_in_channel[ic];
            break;
        }

    }
    return subchannel_index;
}

void PROconfig::construct_collapsing_matrix(){

    collapsing_matrix = Eigen::MatrixXd::Zero(m_num_bins_total, m_num_bins_total_collapsed);

    //construct the matrix by detector block
    Eigen::MatrixXd block_collapser = Eigen::MatrixXd::Zero(m_num_bins_detector_block, m_num_bins_detector_block_collapsed);

    int channel_row_start = 0, channel_col_start = 0;
    for(int ic =0; ic != m_num_channels; ++ic){

        //first, build matrix for each channel block
        int total_num_bins_channel = m_num_subchannels[ic] * m_channel_num_bins[ic];

        Eigen::MatrixXd channel_collapser = Eigen::MatrixXd::Zero(total_num_bins_channel, m_channel_num_bins[ic]);
        for(int col = 0; col != m_channel_num_bins[ic]; ++col){
            for(int subch = 0; subch != m_num_subchannels[ic]; ++subch){
                int row = subch * m_channel_num_bins[ic] + col;
                channel_collapser(row, col) = 1.0;
            }
        }

        // now, copy this matrix to detector block
        block_collapser(Eigen::seqN(channel_row_start, total_num_bins_channel), Eigen::seqN(channel_col_start, m_channel_num_bins[ic])) = channel_collapser;
        channel_row_start += total_num_bins_channel;
        channel_col_start += m_channel_num_bins[ic];
    }

    //okay! now stuff every detector block into the final collapse matrix
    for(int im = 0; im != m_num_modes; ++im){
        for(int id =0; id != m_num_detectors; ++id){
            int row_block_start = im * m_num_bins_mode_block + id * m_num_bins_detector_block;
            int col_block_start = im * m_num_bins_mode_block_collapsed + id * m_num_bins_detector_block_collapsed;
            collapsing_matrix(Eigen::seqN(row_block_start, m_num_bins_detector_block), Eigen::seqN(col_block_start, m_num_bins_detector_block_collapsed)) = block_collapser;
        }
    }
    return;
}
