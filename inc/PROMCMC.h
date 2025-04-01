#ifndef PROMCMC_H
#define PROMCMC_H

#include "PROmetric.h"
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <random>
#include <optional>
namespace PROfit {

template<class Target_FN, class Proposal_FN>
class Metropolis {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniform;

public:
    Target_FN target;
    Proposal_FN proposal;
    Eigen::VectorXf current;

    Metropolis(Target_FN target, Proposal_FN proposal, const Eigen::VectorXf &initial) 
        : target(target), proposal(proposal), current(initial) {
        std::random_device rd{};
        rng.seed(rd());
    }

    bool step() {
        Eigen::VectorXf p = proposal(current);
        float acceptance = std::min(1.0f, target(p)/target(current) * proposal.P(current, p)/proposal.P(p, current));
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
    float width;
    std::vector<int> fixed;
    std::mt19937 rng;
    static constexpr bool has_tune = true;
    std::vector<bool> accepted_list;
    size_t tune_calls = 0;
    float last_acceptance = -1;
    float last_shift;

    simple_proposal(PROmetric &metric, float width = 0.2, std::vector<int> fixed = {}) 
        : metric(metric), width(width), fixed(fixed), last_shift(width) {
        std::random_device rd{};
        rng.seed(rd());
        accepted_list = std::vector(1000, false);
    }

    Eigen::VectorXf operator()(Eigen::VectorXf &current) {
        Eigen::VectorXf ret = current;
        int nparams = metric.GetModel().nparams;
        for(int i = 0; i < ret.size(); ++i) {
            if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
            if(i < nparams) {
                float lo = metric.GetModel().lb(i);
                if(std::isinf(lo)) lo = -5;
                float hi = metric.GetModel().ub(i);
                if(std::isinf(hi)) hi = 5;
                std::uniform_real_distribution<float> ud(lo, hi);
                ret(i) = ud(rng);
            } else {
                std::normal_distribution<float> nd(current(i), width);
                float proposed_value = nd(rng);
                ret(i) = std::clamp(proposed_value, metric.GetSysts().spline_lo[i-nparams], metric.GetSysts().spline_hi[i-nparams]);
            }
        }
        return ret;
    }

    float P(Eigen::VectorXf &value, Eigen::VectorXf &given) {
        float prob = 1.0;
        for(int i = 0; i < value.size(); ++i) {
            if(std::find(fixed.begin(), fixed.end(), i) != std::end(fixed)) continue;
            if(i < metric.GetModel().nparams) {
                float diff = metric.GetModel().ub(i) - metric.GetModel().lb(i);
                if(std::isinf(diff)) diff = 5;
                prob *= 1.0f / diff;
            } else {
                prob *= (1.0f / std::sqrt(2 * M_PI * width * width))
                        * std::exp(-(value(i) - given(i))*(value(i) - given(i))/(2 * width * width));
            }
        }
        return prob;
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

