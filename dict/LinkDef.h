#include <string>
#include <vector>
#include <map>

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace caf;
#pragma link C++ class caf::SRGlobal+;
#pragma link C++ class caf::SRWeightPSet+;
#pragma link C++ class caf::SRWeightMapEntry+;
#pragma link C++ class caf::SRWeightParam+;

#pragma link C++ class string+;
#pragma link C++ class vector<float>+;
#pragma link C++ class vector<double>+;
#pragma link C++ class std::pair<std::string, std::vector<float>>+;
#pragma link C++ class std::pair<std::string, std::vector<double>>+;
#pragma link C++ class std::map<std::string, std::vector<float>>+;
#pragma link C++ class std::map<std::string, std::vector<double>>+;
#endif
