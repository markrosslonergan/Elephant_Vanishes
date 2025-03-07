#ifndef PROSEED_H
#define PROSEED_H



#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>

namespace PROfit{

    class PROseed {
        public:
            PROseed(int seed = -1) {
                std::lock_guard<std::mutex> lock(global_rng_mutex);
                if (seed == -1) {
                    global_rng.seed(rd());  
                } else {
                    global_rng.seed(seed);  // Use fixed seed
                }
            }

            std::mt19937& get_thread_rng() {
                thread_local std::mt19937 rng(global_rng());

                return rng;
            }


        private:
            static inline std::random_device rd; 
            static inline std::mt19937 global_rng; 
            static inline std::mutex global_rng_mutex; 
    };

}
#endif
