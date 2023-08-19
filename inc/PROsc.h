#ifndef PROSC_H_
#define PROSC_H_

// STANDARD
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <cmath>

// TINYXML2
#include "tinyxml2.h"

//PROfit
#include "PROlog.h"

namespace PROfit{


    class PROsc {
        private:

        public:

            PROsc(){};

            float Pmue(float dmsq, float sinsq2thmue, float enu, float baseline) const              

                {
                float sinterm = std::sin(1.27*dmsq*(baseline/enu));
	        float prob    = sinsq2thmue*sinterm*sinterm;

                if(prob<0.0 || prob >1.0){
                    log<LOG_ERROR>(L"Your probability is outside the bounds of math.");
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                    }
                return prob;
                }

            float Pmumu(float dmsq, float sinsq2thmumu, float enu, float baseline) const 

                {
                float sinterm = std::sin(1.27*dmsq*(baseline/enu));
                float prob    = 1.0 - (sinsq2thmumu*sinterm*sinterm);

                if(prob<0.0 || prob >1.0){
                    log<LOG_ERROR>(L"Your probability is outside the bounds of math.");
                    log<LOG_ERROR>(L"Terminating.");
                    exit(EXIT_FAILURE);
                    }

	        return prob;
                }
        };

}
#endif
