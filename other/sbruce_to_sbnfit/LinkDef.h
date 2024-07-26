#include <map>
#include <string>
#include <vector>

// Forward declarations of the classes you want to generate dictionaries for
#ifdef __CINT__
#pragma link C++ class std::vector<float>+;
#pragma link C++ class std::map<std::string, std::vector<float>>+;
#endif

