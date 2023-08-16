#include "PROconfig.h"
#include "PROspec.h"
#include "PROcovariancegen.h"
#include "PROmfa.h"

#include <algorithm>
#include <limits>

#include "CLI11.h"

#define FMT_HEADER_ONLY

using namespace PROfit;
log_level_t GLOBAL_LEVEL = LOG_DEBUG;





int main(int argc, char* argv[])
{


    CLI::App app{"PROfit: EpEm MFA4u"}; 

    // Define options
    std::string xmlname = "NULL.xml"; 
    int maxevents = 100;

    //doubles
    app.add_option("-x,--xml", xmlname, "Input PROfit XML config.");
    app.add_option("-m,--max", maxevents, "Max number of events to run over.");
    app.add_option("-v,--verbosity", GLOBAL_LEVEL, "Verbosity Level [1-4].");
    CLI11_PARSE(app, argc, argv);

    runMFA();
   
    return 0;
}

