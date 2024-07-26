#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <vector>
#include <map>
#include <string>
#include <iostream>

void analyzeROOTFile(const char* inputFilename, const char* outputFilename) {
    // Open the input ROOT file
    TFile* inputFile = TFile::Open(inputFilename);
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error opening file: " << inputFilename << std::endl;
        return;
    }

    // Get the multisimTree from the file
    TTree* multisimTree = (TTree*)inputFile->Get("events/multisimTree");
    if (!multisimTree) {
        std::cerr << "Error getting tree: events/multisimTree" << std::endl;
        return;
    }

    // Get the selectedNu tree from the file
    TTree* selectedNuTree = (TTree*)inputFile->Get("events/selectedNu");
    if (!selectedNuTree) {
        std::cerr << "Error getting tree: events/selectedNu" << std::endl;
        return;
    }

    // Variables to hold values from selectedNu tree
    double in_trueE, in_baseline;
    double in_nuPDG, in_isCC;
    double recoE;

    selectedNuTree->SetBranchAddress("trueE", &in_trueE);
    selectedNuTree->SetBranchAddress("trueL", &in_baseline);
    selectedNuTree->SetBranchAddress("truePDG", &in_nuPDG);
    selectedNuTree->SetBranchAddress("CC", &in_isCC);
    selectedNuTree->SetBranchAddress("recoE", &recoE);

    // Create maps to store systematic information
    std::map<std::string, int> map_systematic_num_universe;
    std::map<std::string, std::vector<float>> map_systematic_vector_weights;

    // Loop over each branch in the multisimTree
    TObjArray* branches = multisimTree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        std::string branchName = branch->GetName();

        // Skip the first four branches as mentioned
        if (i < 4) continue;

        // Create a vector to hold the values
        std::vector<float>* values = nullptr;
        multisimTree->SetBranchAddress(branchName.c_str(), &values);

        // Loop over each entry in the tree
        Long64_t nEntries = multisimTree->GetEntries();
        for (Long64_t j = 0; j < nEntries; ++j) {
            multisimTree->GetEntry(j);

            // Store the size of the vector in map_systematic_num_universe
            map_systematic_num_universe[branchName] = values->size();

            // Store the vector in map_systematic_vector_weights
            map_systematic_vector_weights[branchName] = *values;
        }
    }

    // Create the output ROOT file
    TFile* outputFile = new TFile(outputFilename, "RECREATE");
    TTree* outTree = new TTree("nutree", "nutree");

    // Variables for output branches
    double new_mc_weight = 1.0;

    // Set branch addresses for output tree
    outTree->Branch("trueE", &in_trueE);
    outTree->Branch("recoE", &recoE);
    outTree->Branch("baseline", &in_baseline);
    outTree->Branch("mc_weight", &new_mc_weight);
    outTree->Branch("PDG", &in_nuPDG);
    outTree->Branch("isCC", &in_isCC);
    outTree->Branch("event_weight", &map_systematic_vector_weights);

    // Loop over entries in selectedNuTree to fill the output tree
    Long64_t nEntries = selectedNuTree->GetEntries();
    for (Long64_t j = 0; j < nEntries; ++j) {
        selectedNuTree->GetEntry(j);
        outTree->Fill();
    }

    // Write the output tree to the file
    outTree->Write();
    outputFile->Close();

    // Close the input file
    inputFile->Close();
}

int main() {
    const char* inputFilename = "selected_events.root";
    const char* outputFilename = "processed_sbnfit_profit.root";
    analyzeROOTFile(inputFilename, outputFilename);
    return 0;
}

