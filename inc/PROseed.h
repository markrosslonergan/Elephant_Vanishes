#ifndef PROSEED_H
#define PROSEED_H

#include <random>
#include <thread>
#include <vector>
#include <limits>
#include <memory>

namespace PROfit{

    class PROseed {
        public:
            PROseed(size_t in_threads, int seed = -1) {
                if (seed == -1) {
                    global_rng.seed(rd());  
                } else {
                    global_rng.seed(seed);  // Use fixed seed
                }
                
                n_threads = in_threads;
                thread_seeds.resize(n_threads+1);
                std::uniform_int_distribution<uint32_t> dist(0, std::numeric_limits<uint32_t>::max());
                for(size_t s=0; s<n_threads+1;s++){
                    thread_seeds[s] = dist(global_rng);
                }
                log<LOG_INFO>(L"%1% || For %2%+1 threads we precalculated thread seeds as: %3% ") % __func__ % n_threads % thread_seeds ;

            }

            std::vector<uint32_t>* getThreadSeeds(){
                return &thread_seeds;
            }


            static inline std::mt19937 global_rng; 
        private:
            static inline std::random_device rd; 
            std::vector<uint32_t> thread_seeds;
            size_t n_threads;
    };

}
#endif
