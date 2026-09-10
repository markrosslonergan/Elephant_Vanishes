#ifndef PROWATERMARK_H
#define PROWATERMARK_H

// PROfit version watermark. Every PDF/PNG page PROfit writes should carry the
// version string; call drawVersionWatermark on the TOP-LEVEL canvas (not a
// sub-pad) after the page content is drawn, immediately before Print/SaveAs.
// Multi-page loops that Clear() the canvas must re-stamp each page; a canvas
// printed to several formats back-to-back needs only one stamp.

#include <algorithm>
#include <string>

#include "TH2.h"
#include "TPad.h"
#include "TText.h"
#include "TVirtualPad.h"

#include "PROversion.h"

namespace PROfit {

    enum class WatermarkPos {
        TopRight,    // default: bottom-right of the text sits just above the frame's top-right corner
        BottomRight, // side-by-side panel pages where top-right NDC lands inside a sub-pad
        RightEdge    // dense Divide() grids: no free corner, so run vertically up the right-edge strip
    };

    inline void drawVersionWatermark(TVirtualPad *pad, WatermarkPos pos = WatermarkPos::TopRight) {
        if(!pad) return;
        TVirtualPad *prev = gPad;
        pad->cd();
        TText t;
        t.SetNDC();
        t.SetTextFont(42);
        const std::string pv = "PROfit v" + std::string(PROJECT_VERSION_STR);
        if(pos == WatermarkPos::TopRight) {
            // Anchor the text's bottom-right just above the top-right corner of the axis
            // frame. Default to the pad's own margins; if sub-pads are already drawn
            // (stacked main+ratio layouts), align to the top-most one's frame instead.
            double x = 1.0 - pad->GetRightMargin();
            double y = 1.0 - pad->GetTopMargin();
            double best_top = -1.0;
            TVirtualPad *frame_pad = pad;
            TIter next(pad->GetListOfPrimitives());
            while(TObject *o = next()) {
                if(auto *sub = dynamic_cast<TPad*>(o)) {
                    const double top = sub->GetYlowNDC() + sub->GetHNDC();
                    if(top > best_top) {
                        best_top = top;
                        frame_pad = sub;
                        x = sub->GetXlowNDC() + (1.0 - sub->GetRightMargin()) * sub->GetWNDC();
                        y = sub->GetYlowNDC() + (1.0 - sub->GetTopMargin()) * sub->GetHNDC();
                    }
                }
            }
            // colz pages draw the z-palette's exponent (x10^n) just above the frame's
            // top-right corner; lift the stamp clear of it when a TH2 is drawn WITH a
            // palette (draw option containing 'z') -- a TH2 used as a bare "AXIS"
            // frame has no exponent and needs no lift.
            bool has_palette = false;
            for(TObjLink *lnk = frame_pad->GetListOfPrimitives()->FirstLink(); lnk; lnk = lnk->Next()) {
                if(dynamic_cast<TH2*>(lnk->GetObject())) {
                    std::string opt = lnk->GetOption() ? lnk->GetOption() : "";
                    std::transform(opt.begin(), opt.end(), opt.begin(), ::tolower);
                    if(opt.find('z') != std::string::npos) { has_palette = true; break; }
                }
            }
            const double gap = has_palette ? 0.024 : 0.004;
            t.SetTextSize(0.022f);
            t.SetTextAlign(31); // right-bottom anchor
            // Keep the stamp on-canvas: the glyphs render against the pad's
            // SMALLER pixel dimension while y is a height fraction, so on tall
            // canvases the text is a small height fraction and the frame top
            // sits near 1.0 -- a fixed 0.97 cap would drag the stamp down into
            // the frame there. Cap at (1 - text height) instead.
            const double wh = std::max(1.0, (double)pad->GetWh());
            const double txt_h_ndc = 0.032 * std::min((double)pad->GetWw(), wh) / wh;
            t.DrawText(x, std::min(y + gap, 1.0 - txt_h_ndc - 0.002), pv.c_str()); // DrawText clones; the pad owns the clone
        } else if(pos == WatermarkPos::BottomRight) {
            t.SetTextSize(0.025f);
            t.SetTextAlign(31); // right-bottom anchor
            t.DrawText(0.995, 0.005, pv.c_str());
        } else { // RightEdge
            // Full Divide() grids own both corners (bottom-right cell's x title, top
            // row's panel titles); the right column's pad margins leave an empty strip
            // along the canvas edge — run the stamp vertically up it.
            t.SetTextSize(0.022f);
            t.SetTextAngle(90);
            t.SetTextAlign(11); // rotated: bottom-left anchor, text runs upward
            t.DrawText(0.998, 0.005, pv.c_str());
        }
        if(prev) prev->cd();
    }

}

#endif
