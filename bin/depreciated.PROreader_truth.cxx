#include "PROconfig.h"
#include "PROspec.h"
#include "sbnanaobj/StandardRecord/SRGlobal.h"
#include "sbnanaobj/StandardRecord/SRWeightPSet.h"

#include "CLI11.h"
#include "LBFGSB.h"

#include "TFile.h"

#include <memory>
#include <string>

using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;

int main(int argc, char* argv[])
{

    CLI::App app{"Test for PROfit"}; 

    // Define options
    std::string filename = "NULL.flat.caf.root"; 
    int num = 0;

    app.add_option("-f,--file", filename, "File to read systematics from.");
    app.add_option("-n,--number", num, "umber input");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");

    CLI11_PARSE(app, argc, argv);

    auto file = std::make_unique<TFile>(filename.c_str());
    TTree* globalTree = (TTree*)file->Get("globalTree");


    //auto global = std::make_unique<SRGlobal>();
    caf::SRGlobal* global = NULL;
    globalTree->SetBranchAddress("global", &global);
    globalTree->GetEntry(0);

    int i = 0;
    for(const auto& pset: global->wgts) {
        //for(unsigned int i = 0; i < global->wgts.size(); ++i) {
        //const caf::SRWeightPSet& pset = global->wgts[i];
        //std::cout << "i is: " << i << std::endl;

        if(pset.map.size() != 1) continue;
        std::cout << pset.name << " (type " << pset.type << "): with " << pset.nuniv << " universes\n";
        std::cout << pset.map.at(0).param.name << std::endl;
        std::cout << " Mean: "<<pset.map.at(0).param.mean <<" Width: "<<pset.map.at(0).param.width <<  std::endl;

        for(const auto& val: pset.map.at(0).vals) {
            std::cout << val << ' ';
        }
        std::cout <<" On psetinext i "<<i<< std::endl;
        if(i>0)break;
        i++;
    }


    TTree* recTree = (TTree*)file->Get("recTree");

    int i_wgt_univ_size = 0; //rec.slc.truth.wgt.univ..totarraysize
    int i_wgt_size = 0; //rec.slc..length
    int i_wgt_totsize = 0; //rec.slc.truth.wgt..totalarraysize

    float v_wgt_univ[20000];
    int v_wgt_univ_idx[20000];
    int v_wgt_idx[2000];
    int v_wgt_univ_length[2000];
    int v_truth_index[2000] ;

    recTree->SetBranchAddress("rec.slc.truth.wgt.univ..totarraysize", &i_wgt_univ_size);
    recTree->SetBranchAddress("rec.slc..length", &i_wgt_size);
    recTree->SetBranchAddress("rec.slc.truth.wgt..totarraysize",&i_wgt_totsize);

    recTree->SetBranchAddress("rec.slc.truth.wgt.univ", v_wgt_univ);
    recTree->SetBranchAddress("rec.slc.truth.wgt.univ..idx", v_wgt_univ_idx);
    recTree->SetBranchAddress("rec.slc.truth.wgt..idx",v_wgt_idx);
    recTree->SetBranchAddress("rec.slc.truth.wgt.univ..length",v_wgt_univ_length);
    recTree->SetBranchAddress("rec.slc.truth.index", v_truth_index);
    recTree->GetEntry(num);


    std::cout<<"  rec.slc.truth.wgt.univ (val):  "<<v_wgt_univ[0]<<" (rec.slc.truth.wgt.univ..idx) (indx): "<<v_wgt_univ_idx[0]<<" both of length be: "<<i_wgt_univ_size<<" (rec.slc.truth.wgt..totarraysize )"<<std::endl; 
    std::cout<<"  rec.slc.truth.wgt..idx (idx):  "<<v_wgt_idx[0]<<" of length be: "<<i_wgt_size<<" (rec.slc..length)"<<std::endl; 
    for(int k=0; k< i_wgt_size; k++)std::cout<<v_wgt_idx[k]<<" ";
    std::cout<<std::endl;

    std::cout<<"  rec.slc.truth.wgt.univ..length (val):  "<<v_wgt_univ_length[0]<<" of length be: "<<i_wgt_totsize<<" (rec.slc.truth.wgt..totarraysize)" <<std::endl; 

    std::vector<double> ll = {-1, 1 ,-2, 2, -3, 3};
    //rec.slc.truth.wgh.univ[rec.slc.truth.wgt..idx[SLC]+I]+6
    for(int s = 0; s<i_wgt_size;s++){
        if(v_truth_index[s]==0){
            std::cout<<"Slice "<<s<<", truth "<<v_truth_index[s]<<", on pset "<<i<<"   ";
            int how_many =v_wgt_univ_length[i];
            for(int k=0; k<how_many; k++){ 
                std::cout<<"("<<ll[k]<<" , "<<v_wgt_univ[v_wgt_idx[s]+i+k]<<")  ";
            }std::cout<<std::endl;
        }
    }



    }


