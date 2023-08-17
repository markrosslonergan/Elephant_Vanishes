#include "PROsyst.h"
#include "PROcreate.h"

namespace PROfit {

    PROsyst::PROsyst(const std::vector<SystStruct>& systs) {
        for(const auto& syst: systs) {
            if(syst.mode == "multisigma") {
                FillSpline(syst);
            } else {

            }
        }
    }

    void PROsyst::FillSpline(const SystStruct& syst) {
        std::vector<PROspec> ratios;
        ratios.reserve(syst.p_multi_spec.size());
        for(size_t i = 0; i < syst.p_multi_spec.size(); ++i) {
            ratios.push_back(*syst.p_multi_spec[i] / *syst.p_cv);
            if(syst.knobval[i] == -1) ratios.push_back(*syst.p_cv / *syst.p_cv);
        }
        Spline spline_coeffs;
        spline_coeffs.reserve(syst.p_cv->GetNbins());
        for(long i = 0; i < syst.p_cv->GetNbins(); ++i) {
            std::vector<std::pair<float, std::array<float, 4>>> spline;
            spline.reserve(syst.knobval.size());

            // This comment is copy-pasted from CAFAna:
            // This is cubic interpolation. For each adjacent set of four points we
            // determine coefficients for a cubic which will be the curve between the
            // center two. We constrain the function to match the two center points
            // and to have the right mean gradient at them. This causes this patch to
            // match smoothly with the next one along. The resulting function is
            // continuous and first and second differentiable. At the ends of the
            // range we fit a quadratic instead with only one constraint on the
            // slope. The coordinate conventions are that point y1 sits at x=0 and y2
            // at x=1. The matrices are simply the inverses of writing out the
            // constraints expressed above.

            const float y1 = ratios[0].GetBinContent(i);
            const float y2 = ratios[1].GetBinContent(i);
            const float y3 = ratios[2].GetBinContent(i);
            const Eigen::Vector3f v{y1, y2, (y3-y1)/2};
            const Eigen::Matrix3f m{{ 1, -1,  1},
                {-2,  2, -1},
                { 1,  0,  0}};
            const Eigen::Vector3f res = m * v;
            spline.push_back({syst.knobval[0], {res(2), res(1), res(0), 0}});

            for(unsigned int shiftIdx = 1; shiftIdx < ratios.size()-2; ++shiftIdx){
                const float y0 = ratios[shiftIdx-1].GetBinContent(i);
                const float y1 = ratios[shiftIdx  ].GetBinContent(i);
                const float y2 = ratios[shiftIdx+1].GetBinContent(i);
                const float y3 = ratios[shiftIdx+2].GetBinContent(i);
                const Eigen::Vector4f v{y1, y2, (y2-y0)/2, (y3-y1)/2};
                const Eigen::Matrix4f m{{ 2, -2,  1,  1},
                    {-3,  3, -2, -1},
                    { 0,  0,  1,  0},
                    { 1,  0,  0,  0}};
                const Eigen::Vector4f res = m * v;
                float knobval = syst.knobval[shiftIdx] <  0 ? syst.knobval[shiftIdx] :
                    syst.knobval[shiftIdx] == 1 ? 0 :
                    syst.knobval[shiftIdx - 1];
                spline.push_back({knobval, {res(3), res(2), res(1), res(0)}});
            }

            const float y4 = ratios[ratios.size() - 3].GetBinContent(i);
            const float y5 = ratios[ratios.size() - 2].GetBinContent(i);
            const float y6 = ratios[ratios.size() - 1].GetBinContent(i);
            const Eigen::Vector3f vp{y5, y6, (y6-y4)/2};
            const Eigen::Matrix3f mp{{-1,  1, -1},
                { 0,  0,  1},
                { 1,  0,  0}};
            const Eigen::Vector3f resp = mp * vp;
            spline.push_back({syst.knobval[syst.knobval.size() - 1], {resp(2), resp(1), resp(0), 0}});

            spline_coeffs.push_back(spline);
        }
        splines[syst.systname] = spline_coeffs;
    }

    float PROsyst::GetSplineShift(std::string name, float shift , int bin) {
        if(bin < 0 || bin >= splines[name].size()) return -1;
        const float lowest_knobval = splines[name][0][0].first;
        int shiftBin = (shift < lowest_knobval) ? 0 : (long)(shift - lowest_knobval);
        if(shiftBin > splines[name][0].size() - 1) shiftBin = splines[name][0].size() - 1;
        // We should use the line below if we switch to c++17
        // const long shiftBin = std::clamp((long)(shift - knobval[0]), 0, spline_coeffs[0].size() - 1);
        std::array<float, 4> coeffs = splines[name][bin][shiftBin].second;
        shift -= splines[name][bin][shiftBin].first;
        return coeffs[0] + coeffs[1]*shift + coeffs[2]*shift*shift + coeffs[3]*shift*shift*shift;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROspec& cv, std::string name, float shift) {
        assert(cv.GetNbins() == splines[name].size());
        PROspec ret(cv.GetNbins());
        for(int i = 0; i < cv.GetNbins(); ++i)
            ret.Fill(i, GetSplineShift(name, shift, i) * cv.GetBinContent(i));
        return ret;
    }

    PROspec PROsyst::GetSplineShiftedSpectrum(const PROspec& cv, std::vector<std::string> names, std::vector<float> shifts) {
        assert(cv.GetNbins() == splines[names[0]].size());
        assert(names.size() == shifts.size());
        PROspec ret(cv.GetNbins());
        for(int i = 0; i < cv.GetNbins(); ++i) {
            float weight = 1;
            for(size_t j = 0; j < names.size(); ++j) {
                weight *= GetSplineShift(names[j], shifts[j], i);
            }
            ret.Fill(i, weight * cv.GetBinContent(i));
        }
        return ret;
    }
};

