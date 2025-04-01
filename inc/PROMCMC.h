#ifndef PROMCMC_H
#define PROMCMC_H

#include "PROmetric.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <optional>
namespace PROfit {

template<class Target_FN, class Proposal_FN>
class Metropolis {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform;
    uint32_t seed;

public:
    Target_FN target;
    Proposal_FN proposal;
    Eigen::VectorXf current;

    Metropolis(Target_FN target, Proposal_FN proposal, const Eigen::VectorXf &initial, uint32_t seed) 
        : seed(seed), target(target), proposal(proposal), current(initial) {
        rng.seed(seed);
    }

    bool step() {
        Eigen::VectorXf p = proposal(current);
        float acceptance = proposal.within_bound(p) ? std::min(1.0f, target(p)/target(current) * proposal.P(current, p)/proposal.P(p, current)) : 0;
        float u = uniform(rng);
        if(u <= acceptance) {
            current = p;
            return true;
        }
        return false;
    }

    void run(size_t burnin, size_t steps, std::optional<std::function<void(const Eigen::VectorXf&)>> action = {}) {
        for(size_t i = 0; i < burnin; i++) {
            if constexpr(Proposal_FN::has_tune) {
                proposal.tune(step());
            } else {
                step();
            }
        }
        for(size_t i = 0; i < steps; i++) {
            if(step() && action) (*action)(current);
        }
    }

};

struct simple_target {
    PROmetric &metric;

    float operator()(Eigen::VectorXf &value) {
        Eigen::VectorXf empty;
        return std::exp(-0.5f*metric(value, empty, false));
    }
};

struct simple_proposal {
    PROmetric &metric;
    uint32_t seed;
    float width;
    std::vector<int> fixed;
    std::mt19937 rng;
    static constexpr bool has_tune = true;
    std::vector<bool> accepted_list;
    size_t tune_calls = 0;
    float last_acceptance = -1;
    float last_shift;

    simple_proposal(PROmetric &metric, uint32_t seed, float width = 0.2, std::vector<int> fixed = {}) 
        : metric(metric), seed(seed), width(width), fixed(fixed), rng(seed), last_shift(width) {
        accepted_list = std::vector(1000, false);
    }

    Eigen::VectorXf operator()(Eigen::VectorXf &current) {
        Eigen::VectorXf ret = current;
        int nparams = metric.GetModel().nparams;
        for(int i = 0; i < ret.size(); ++i) {
            if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
            if(i < nparams) {
                // TODO: How to use width with a uniform distribution
                float lo = metric.GetModel().lb(i);
                if(std::isinf(lo)) lo = -5;
                float hi = metric.GetModel().ub(i);
                if(std::isinf(hi)) hi = 5;
                std::uniform_real_distribution<float> ud(lo, hi);
                ret(i) = ud(rng);
            } else if(metric.GetSysts().spline_lo[i-nparams] == 0) {
                // Currently there's some weird behavior with the 0-1 systematics
                // which using a uniform distribution seems to fix
                // TODO: How to use width with a uniform distribution
                float lo = metric.GetSysts().spline_lo[i-nparams];
                float hi = metric.GetSysts().spline_hi[i-nparams];
                std::uniform_real_distribution<float> ud(lo, hi);
                ret(i) = ud(rng);
            } else {
                std::normal_distribution<float> nd(current(i), width);
                float proposed_value = nd(rng);
                ret(i) = proposed_value;
                //ret(i) = std::clamp(proposed_value, metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
            }
        }
        return ret;
    }

    float P(const Eigen::VectorXf &value, const Eigen::VectorXf &given) {
        float prob = 1.0;
        int nparams = metric.GetModel().nparams;
        for(int i = 0; i < value.size(); ++i) {
            if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
            if(i < nparams) {
                float diff = metric.GetModel().ub(i) - metric.GetModel().lb(i);
                if(std::isinf(diff)) diff = 5;
                prob *= 1.0f / diff;
            } else {
                if(value(i) <= metric.GetSysts().spline_lo[i-nparams] || value(i) >= metric.GetSysts().spline_hi[i-nparams] || 
                   given(i) <= metric.GetSysts().spline_lo[i-nparams] || given(i) >= metric.GetSysts().spline_hi[i-nparams]) {
                    // Due to bounds, use CDF to get total probability value is <= bound
                    // Symmetry makes this work for upper bound as well
                    float v = std::clamp(value(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    float g = std::clamp(given(i), metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
                    prob *= 0.5f * (1.0f + std::erff((v - g)/(std::sqrt(2.0f)*width)));
                    //prob = 0;
                } else {
                    prob *= (1.0f / std::sqrt(2 * M_PI * width * width))
                            * std::exp(-(value(i) - given(i))*(value(i) - given(i))/(2 * width * width));
                }
            }
        }
        return prob;
    }

    bool within_bound(Eigen::VectorXf &value) {
        int nparams = metric.GetModel().nparams;
        for(int i = 0; i < value.size(); ++i) {
            if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
            if(i < nparams) {
                if(value(i) > metric.GetModel().ub(i) || value(i) < metric.GetModel().lb(i))
                    return false;
            } else {
                if(value(i) < metric.GetSysts().spline_lo[i-nparams] || value(i) > metric.GetSysts().spline_hi[i-nparams])
                    return false;
            }
        }
        return true;
    }

    void tune(bool accepted) {
        accepted_list[tune_calls % 1000] = accepted;
        if(++tune_calls % 1000 == 0) {
            float acceptance = std::count(accepted_list.begin(), accepted_list.end(), true) / 1000.0f;
            if(acceptance < 0.25 || acceptance > 0.35) {
                if(std::abs(acceptance - 0.3) < std::abs(last_acceptance - 0.3)) {
                    if(last_acceptance < 0) {
                        width *= 1.25; // Default first step
                        last_shift = 0.25 * width;
                    } else {
                        width += last_shift;
                    }
                } else { // Moved too far
                    width += -0.5 * last_shift;
                    last_shift *= -0.5;
                }
            }
            for(size_t i = 0; i < 1000; ++i) accepted_list[i] = false;
            last_acceptance = acceptance;
        }
    }
};

}

#endif

