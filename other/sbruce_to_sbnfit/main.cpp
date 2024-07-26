#include "analyzeROOTfile.h"

int main() {
    const char* inputFilename = "selected_events.root";
    const char* outputFilename = "selected_output.root";
    analyzeROOTFile(inputFilename, outputFilename);
    return 0;
}

