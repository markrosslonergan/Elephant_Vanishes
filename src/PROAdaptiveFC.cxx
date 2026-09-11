/**
 * @file PROAdaptiveFC.cxx
 * @brief Adaptive Feldman-Cousins pipeline — top-level mode dispatcher.
 *
 * run_adaptive_fc() routes each AdaptiveFCMode (build-mesh, init-bank,
 * print-bank, asimov, brazil) to the pipeline stages implemented in the
 * sibling translation units:
 *   - src/PROAdaptiveFCmesh.cxx  — throws, Wilks AMR prepass, meta-mesh
 *   - src/PROAdaptiveFCbank.cxx  — serialisation, PE worker/scheduler,
 *                                  asimov observables, classification
 *   - src/PROAdaptiveFCplot.cxx  — ROOT PDF / artifact output
 * Cross-TU internals are declared in inc/PROAdaptiveFCinternal.h
 * (namespace PROfit::afc). The AMR per-point fit body and the mesh drawing
 * are shared with PROsurf via inc/PROmeshEval.h and inc/PROmeshPlot.h.
 */
#include "PROAdaptiveFC.h"
#include "PROAdaptiveFCinternal.h"

#include "PROlog.h"
#include "PROmodel.h"
#include "PROspec.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "MurmurHash3.h"

#include <Eigen/Eigen>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace PROfit {

using namespace afc;

// ====================================================================
//  Helpers — axis name → parameter/spline index resolution.
//  Mirrors the lookup at bin/PROfit.cxx:1425-1457 used by the `surface`
//  command. Kept local because the parent function is large and not yet
//  factored out.
// ====================================================================

static size_t resolve_axis_index(const std::string &name,
                                 const PROmodel &model,
                                 const PROsyst  &systs,
                                 const PROconfig &config,
                                 size_t fallback)
{
    if (auto loc = std::find(model.param_names.begin(), model.param_names.end(), name);
        loc != model.param_names.end()) {
        return std::distance(model.param_names.begin(), loc);
    }
    // Spline axes index the FULL [physics | spline] parameter vector that the
    // callers pin/bound, so the spline position must be offset by nparams.
    if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), name);
        loc != systs.spline_names.end()) {
        return model.nparams + std::distance(systs.spline_names.begin(), loc);
    }
    for (const auto &[xml_name, plot_name] : config.m_mcgen_variation_plotname_map) {
        if (name == plot_name) {
            if (auto loc = std::find(systs.spline_names.begin(), systs.spline_names.end(), xml_name);
                loc != systs.spline_names.end()) {
                return model.nparams + std::distance(systs.spline_names.begin(), loc);
            }
        }
    }
    log<LOG_WARNING>(L"%1% || resolve_axis_index: axis variable '%2%' not found; falling back to %3%.")
        % __func__ % name.c_str() % fallback;
    return fallback;
}

// ====================================================================
//  Section 5b — Brazil throw aggregation (shared by brazil / brazil-cleanup)
// ====================================================================

namespace {
    /// Everything derived from (bank, archive) by re-classification:
    /// per-throw verdicts, the closed-contour filter decisions, and the
    /// per-cell inclusion fractions over kept throws.
    struct BrazilAggregation {
        std::vector<std::vector<std::vector<uint8_t>>> per_throw_verdicts; ///< [t][cl][cell]
        std::vector<std::vector<uint8_t>> throw_kept;                      ///< [cl][t]
        std::vector<std::vector<float>>   inclusion_frac;                  ///< [cl][cell]
        std::vector<int> n_kept_per_cl;
        std::vector<int> n_dropped_per_cl;
    };
    constexpr float kUndecidedSentinel = -1.0f;
}

// Re-classify every archived throw against the bank, apply the
// closed-contour filter, and aggregate per-cell inclusion fractions over
// kept throws. Pure function of (bank, archive, cl_targets, min_pes) — no
// RNG, no fits — used by --mode brazil (band outputs) and
// --mode brazil-cleanup (contour-targeted mesh refinement).
//
// Details preserved from the original in-line brazil implementation:
//  * `included` and `decidable` are tracked separately per throw so the
//    aggregation can distinguish "throw said outside" from "bank too sparse
//    to say" — treating undecidable as "outside" paints a phantom band ring
//    at the AMR refined↔baseline boundary.
//  * crit_dchi2/decidable depend only on the bank, so they are computed ONCE
//    (one sort per cell, compute_bank_crits) and each throw reduces to a
//    comparison; throws are classified in parallel (no RNG, disjoint writes
//    — bitwise deterministic for any thread count).
//  * Closed-contour filter: the band represents the spread of *exclusion
//    boundaries*; a throw whose allowed region is a closed island (the
//    bottom-left no-oscillation corner itself excluded) answers a different
//    question and is dropped for that CL. Decidability of the corner cell
//    depends only on its PE count, so the filter is fully active or — bank
//    too sparse there — fully inactive with a warning.
static BrazilAggregation aggregate_brazil_throws(
    const PEBank &bank,
    const BrazilArchive &arc,
    const std::vector<float> &cl_targets,
    int min_pes,
    int nthreads)
{
    const int n_total = (int)arc.per_throw_dchi2.size();
    const int n_cells = bank.n_cells;
    const int n_cl    = (int)cl_targets.size();

    BrazilAggregation agg;
    agg.per_throw_verdicts.assign(
        (size_t)n_total,
        std::vector<std::vector<uint8_t>>(
            (size_t)n_cl, std::vector<uint8_t>((size_t)n_cells, 0)));
    std::vector<std::vector<std::vector<uint8_t>>> per_throw_decidable(
        (size_t)n_total,
        std::vector<std::vector<uint8_t>>(
            (size_t)n_cl, std::vector<uint8_t>((size_t)n_cells, 0)));
    {
        const auto bank_crits = compute_bank_crits(bank, cl_targets, min_pes);
        std::atomic<int> next_throw{0};
        auto classify_worker = [&]() {
            while (true) {
                const int t = next_throw.fetch_add(1);
                if (t >= n_total) break;
                const auto &dchi2 = arc.per_throw_dchi2[(size_t)t];
                for (int k = 0; k < n_cl; ++k) {
                    for (int c = 0; c < n_cells; ++c) {
                        const auto &v = bank_crits[(size_t)k][(size_t)c];
                        per_throw_decidable[(size_t)t][(size_t)k][(size_t)c] = v.decidable ? 1 : 0;
                        agg.per_throw_verdicts[(size_t)t][(size_t)k][(size_t)c] =
                            (v.decidable && dchi2[(size_t)c] <= v.crit_dchi2) ? 1 : 0;
                    }
                }
            }
        };
        const int n_workers = std::max(1, std::min(nthreads, n_total));
        std::vector<std::thread> pool;
        pool.reserve((size_t)n_workers);
        for (int i = 0; i < n_workers; ++i) pool.emplace_back(classify_worker);
        for (auto &th : pool) th.join();
    }

    // Closed-contour filter (see block comment above).
    int corner_cell = -1;
    for (int c = 0; c < n_cells; ++c) {
        if (bank.cell_i_bl[(size_t)c] == 0 && bank.cell_j_bl[(size_t)c] == 0) {
            corner_cell = c;
            break;
        }
    }
    agg.throw_kept.assign((size_t)n_cl, std::vector<uint8_t>((size_t)n_total, 1));
    agg.n_kept_per_cl.assign((size_t)n_cl, n_total);
    agg.n_dropped_per_cl.assign((size_t)n_cl, 0);
    if (corner_cell < 0) {
        log<LOG_WARNING>(L"%1% || brazil: no meta-cell anchored at grid corner (0,0); closed-contour filter inactive.")
            % __func__;
    } else if (n_total > 0 && n_cl > 0
               && !per_throw_decidable[0][0][(size_t)corner_cell]) {
        log<LOG_WARNING>(L"%1% || brazil: corner cell %2% is undecidable (< %3% PEs in bank); closed-contour filter inactive -- top up the bank to enable it.")
            % __func__ % corner_cell % min_pes;
    } else {
        for (int k = 0; k < n_cl; ++k) {
            for (int t = 0; t < n_total; ++t) {
                if (!agg.per_throw_verdicts[(size_t)t][(size_t)k][(size_t)corner_cell]) {
                    agg.throw_kept[(size_t)k][(size_t)t] = 0;
                    ++agg.n_dropped_per_cl[(size_t)k];
                }
            }
            agg.n_kept_per_cl[(size_t)k] = n_total - agg.n_dropped_per_cl[(size_t)k];
            log<LOG_INFO>(L"%1% || brazil closed-contour filter CL=%2%: kept %3% / %4% throws (%5% dropped -- corner excluded, closed allowed contour).")
                % __func__ % cl_targets[(size_t)k]
                % agg.n_kept_per_cl[(size_t)k] % n_total % agg.n_dropped_per_cl[(size_t)k];
            if (agg.n_kept_per_cl[(size_t)k] == 0) {
                log<LOG_WARNING>(L"%1% || brazil: ALL %2% throws dropped at CL=%3% -- the band for this CL will be empty.")
                    % __func__ % n_total % cl_targets[(size_t)k];
            }
        }
    }

    // Aggregate inclusion fractions across kept throws only. Sentinel:
    // inclusion_frac < 0 marks a cell where no kept throw was ever decidable;
    // build_inclusion_th2d skips such cells from the IDW interpolation.
    agg.inclusion_frac.assign(
        (size_t)n_cl, std::vector<float>((size_t)n_cells, kUndecidedSentinel));
    int total_undecidable_cells = 0;
    for (int k = 0; k < n_cl; ++k) {
        int undecidable_this_cl = 0;
        for (int c = 0; c < n_cells; ++c) {
            int n_in = 0, n_decided = 0;
            for (int t = 0; t < n_total; ++t) {
                if (!agg.throw_kept[(size_t)k][(size_t)t]) continue;
                if (per_throw_decidable[(size_t)t][(size_t)k][(size_t)c]) {
                    ++n_decided;
                    if (agg.per_throw_verdicts[(size_t)t][(size_t)k][(size_t)c]) ++n_in;
                }
            }
            if (n_decided > 0) {
                agg.inclusion_frac[(size_t)k][(size_t)c] = (float)n_in / (float)n_decided;
            } else {
                agg.inclusion_frac[(size_t)k][(size_t)c] = kUndecidedSentinel;
                ++undecidable_this_cl;
            }
        }
        log<LOG_INFO>(L"%1% || brazil aggregation CL=%2%: %3% / %4% cells undecidable (bank too sparse).")
            % __func__ % cl_targets[(size_t)k] % undecidable_this_cl % n_cells;
        total_undecidable_cells += undecidable_this_cl;
    }
    if (total_undecidable_cells > 0) {
        log<LOG_INFO>(L"%1% || brazil: %2% undecidable (cell, CL) entries skipped from IDW interpolation; "
                      L"deep-basin / sparse-bank regions are filled by neighbouring decided cells.")
            % __func__ % total_undecidable_cells;
    }
    return agg;
}

// Band PDF + ROOT artifacts from an aggregated (bank, archive) pair —
// the shared output tail of --mode brazil and --mode merge-brazil.
static void emit_brazil_outputs(
    const PEBank &bank,
    const std::string &bank_in,
    const BrazilArchive &arc,
    const BrazilAggregation &agg,
    const AdaptiveFCConfig &acfg,
    const PROmodel &model,
    size_t xaxis_idx, size_t yaxis_idx,
    const Eigen::VectorXf &fakeDataParams)
{
    const std::string brazil_pdf  = acfg.output_tag + "_brazil_band.pdf";
    const std::string brazil_root = acfg.output_tag + "_brazil.root";
    const bool xlog_axis = (xaxis_idx < model.is_log10.size()) ? model.is_log10[xaxis_idx] : acfg.logx;
    const bool ylog_axis = (yaxis_idx < model.is_log10.size()) ? model.is_log10[yaxis_idx] : acfg.logy;
    const std::string xlabel = xaxis_idx < model.nparams
        ? model.pretty_param_names.at(xaxis_idx) : std::string("x");
    const std::string ylabel = yaxis_idx < model.nparams
        ? model.pretty_param_names.at(yaxis_idx) : std::string("y");
    const float truth_x_phys = xlog_axis
        ? std::pow(10.0f, fakeDataParams((int)xaxis_idx))
        : fakeDataParams((int)xaxis_idx);
    const float truth_y_phys = ylog_axis
        ? std::pow(10.0f, fakeDataParams((int)yaxis_idx))
        : fakeDataParams((int)yaxis_idx);

    plot_brazil_band_pdf(bank, agg.inclusion_frac, acfg.cl_targets, brazil_pdf, bank_in,
                          xlabel, ylabel, acfg.logx, acfg.logy,
                          xlog_axis, ylog_axis,
                          /*draw_truth_marker=*/ true,
                          truth_x_phys, truth_y_phys,
                          agg.n_kept_per_cl, agg.n_dropped_per_cl,
                          acfg.band_flag);

    save_brazil_root(bank, agg.per_throw_verdicts, arc.per_throw_dchi2,
                      arc.per_throw_global_chi2, agg.inclusion_frac, agg.throw_kept,
                      acfg.cl_targets, brazil_root, xlog_axis, ylog_axis);
}

// ====================================================================
//  Section 6 — run_adaptive_fc (top-level dispatcher)
// ====================================================================

AdaptiveFCResult run_adaptive_fc(
    const PROconfig &config,
    const PROpeller &prop,
    const PROsyst   &systs,
    const PROfitterConfig &fitconfig,
    PROseed         &proseed,
    const Eigen::VectorXf &fakeDataParams,
    const PROdata   &data,
    const AdaptiveFCConfig &acfg,
    int nthreads,
    MultiPROgressBar &progress)
{
    AdaptiveFCResult res;

    log<LOG_INFO>(L"%1% || mode=%2%, throws=%3%, p_thresh=%4%, baseline_level=%5%, "
                  L"prepass=%6%x%7%/levels=%8%, stat_only=%9%.")
        % __func__ % (int)acfg.mode % acfg.n_throws % acfg.p_thresh % acfg.baseline_level
        % acfg.prepass_amr_initial_x % acfg.prepass_amr_initial_y % acfg.prepass_amr_levels
        % (int)acfg.stat_only_throws;

    std::unique_ptr<PROmodel> model = get_model_from_string(config, prop);

    // Resolve axis names → indices using the same lookup the surface command uses.
    const size_t xaxis_idx = resolve_axis_index(acfg.xvar, *model, systs, config, 1);
    const size_t yaxis_idx = resolve_axis_index(acfg.yvar, *model, systs, config, 0);
    log<LOG_INFO>(L"%1% || resolved xvar='%2%' -> idx=%3%; yvar='%4%' -> idx=%5%.")
        % __func__ % acfg.xvar.c_str() % (int)xaxis_idx % acfg.yvar.c_str() % (int)yaxis_idx;

    // ---- Mode: merge-mesh ---------------------------------------------------
    // Union-merge >= 2 mesh binaries (--merge-input, filenames or globs
    // expanded by the CLI) into <tag>_mesh.bin. No fitting; typical use is
    // combining meshes produced independently (e.g. grid jobs) before
    // merge-bank + init-bank top-up.
    if (acfg.mode == AdaptiveFCMode::MergeMesh) {
        if (acfg.merge_inputs.size() < 2) {
            log<LOG_ERROR>(L"%1% || merge-mesh: need >= 2 --merge-input mesh binaries, got %2%.")
                % __func__ % (int)acfg.merge_inputs.size();
            return res;
        }
        std::vector<MetaMesh> inputs(acfg.merge_inputs.size());
        for (size_t m = 0; m < acfg.merge_inputs.size(); ++m) {
            if (!load_mesh(inputs[m], acfg.merge_inputs[m])) {
                log<LOG_ERROR>(L"%1% || merge-mesh: failed to load %2%.")
                    % __func__ % acfg.merge_inputs[m].c_str();
                return res;
            }
            log<LOG_INFO>(L"%1% || merge-mesh input %2%: %3% (%4% cells, finest=%5%x%6%, max_levels=%7%).")
                % __func__ % (int)m % acfg.merge_inputs[m].c_str()
                % (int)inputs[m].cells.size() % inputs[m].finest_nx
                % inputs[m].finest_ny % inputs[m].max_levels;
        }
        const MetaMesh &mref = inputs.front();
        for (size_t m = 1; m < inputs.size(); ++m) {
            const MetaMesh &mm2 = inputs[m];
            if (mm2.finest_nx != mref.finest_nx || mm2.finest_ny != mref.finest_ny
                || mm2.x_lo != mref.x_lo || mm2.x_hi != mref.x_hi
                || mm2.y_lo != mref.y_lo || mm2.y_hi != mref.y_hi) {
                log<LOG_ERROR>(L"%1% || merge-mesh: %2% has an incompatible grid "
                               L"(finest %3%x%4% box [%5%,%6%]x[%7%,%8%] vs reference %9%x%10% [%11%,%12%]x[%13%,%14%]). "
                               L"Meshes must share finest resolution and bounds; refusing to merge.")
                    % __func__ % acfg.merge_inputs[m].c_str()
                    % mm2.finest_nx % mm2.finest_ny % mm2.x_lo % mm2.x_hi % mm2.y_lo % mm2.y_hi
                    % mref.finest_nx % mref.finest_ny % mref.x_lo % mref.x_hi % mref.y_lo % mref.y_hi;
                return res;
            }
        }

        MetaMesh merged = merge_meta_meshes(inputs, acfg.baseline_level);
        if (merged.cells.empty()) {
            log<LOG_ERROR>(L"%1% || merge-mesh: merge failed (see errors above).") % __func__;
            return res;
        }
        const std::string mesh_out = acfg.output_tag + "_mesh.bin";
        if (save_mesh(merged, mesh_out)) {
            log<LOG_INFO>(L"%1% || merge-mesh: wrote %2% (%3% cells). "
                          L"Next: --mode merge-bank with the input banks, then --mode init-bank to top up.")
                % __func__ % mesh_out.c_str() % (int)merged.cells.size();
        }
        res.n_meta_cells     = (int)merged.cells.size();
        res.n_baseline_cells = merged.n_baseline_cells;
        res.n_refined_cells  = merged.n_refined_cells;
        return res;
    }

    // ---- Mode: merge-bank ---------------------------------------------------
    // Harvest PEs from >= 1 bank binaries (--merge-input) onto this tag's
    // <tag>_mesh.bin (produced by merge-mesh). A PE is a Δχ² sample thrown at
    // a cell CENTER, so it is carried over only where a merged-mesh cell has
    // the exact same footprint (i_bl, j_bl, step) — same footprint ⇒ same
    // center ⇒ same truth point ⇒ poolable. PEs from cells that changed
    // refinement are unusable at the new centers and are dropped (reported).
    // Bitwise-identical PEs (same per-PE seed AND dchi2 — i.e. two banks
    // generated with the same --seed) are deduplicated: pooling correlated
    // duplicates would silently inflate the effective statistics.
    //
    // NOTE: mesh/bank binaries carry no XML/fit-config provenance, so this
    // CANNOT verify the input banks came from the same config, metric,
    // preset, and grad-mode. Pooling banks with mismatched fit settings
    // mixes different Δχ² distributions — that discipline is on the caller.
    if (acfg.mode == AdaptiveFCMode::MergeBank) {
        if (acfg.merge_inputs.empty()) {
            log<LOG_ERROR>(L"%1% || merge-bank: need >= 1 --merge-input bank binaries.") % __func__;
            return res;
        }
        const std::string mesh_in = acfg.output_tag + "_mesh.bin";
        MetaMesh mm;
        if (!load_mesh(mm, mesh_in)) {
            log<LOG_ERROR>(L"%1% || merge-bank: required mesh artifact %2% not found. "
                           L"Run --mode merge-mesh (or build-mesh) first.")
                % __func__ % mesh_in.c_str();
            return res;
        }

        const bool xlog = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        std::vector<float> cell_x_model, cell_y_model;
        compute_cell_centers(mm, xlog, ylog, cell_x_model, cell_y_model);

        PEBank bank;
        bank.finest_nx  = mm.finest_nx;
        bank.finest_ny  = mm.finest_ny;
        bank.max_levels = mm.max_levels;
        bank.x_lo = mm.x_lo; bank.x_hi = mm.x_hi;
        bank.y_lo = mm.y_lo; bank.y_hi = mm.y_hi;
        bank.n_cells = (int)mm.cells.size();
        bank.cell_center_x = cell_x_model;
        bank.cell_center_y = cell_y_model;
        bank.cell_pes.assign(mm.cells.size(), {});
        std::unordered_map<uint64_t, int> footprint_to_idx;
        footprint_to_idx.reserve(mm.cells.size());
        auto footprint_key = [](int i_bl, int j_bl, int step) -> uint64_t {
            return ((uint64_t)(uint32_t)i_bl << 42)
                 | ((uint64_t)(uint32_t)j_bl << 21)
                 |  (uint64_t)(uint32_t)step;
        };
        for (size_t c = 0; c < mm.cells.size(); ++c) {
            const auto &mc = mm.cells[c];
            bank.cell_i_bl.push_back(mc.i_bl);
            bank.cell_j_bl.push_back(mc.j_bl);
            bank.cell_step.push_back(mc.step);
            bank.cell_level.push_back(mc.level);
            footprint_to_idx[footprint_key(mc.i_bl, mc.j_bl, mc.step)] = (int)c;
        }

        // Per-cell dedupe keys: (PE seed, bit pattern of dchi2).
        std::vector<std::unordered_set<uint64_t>> seen(mm.cells.size());
        auto pe_key = [](const PEBankRecord &r) -> uint64_t {
            uint32_t bits;
            std::memcpy(&bits, &r.dchi2, sizeof(bits));
            return ((uint64_t)r.seed << 32) | (uint64_t)bits;
        };

        int64_t total_carried = 0, total_dups = 0, total_dropped = 0;
        for (const auto &path : acfg.merge_inputs) {
            PEBank in;
            if (!load_bank(in, path)) {
                log<LOG_ERROR>(L"%1% || merge-bank: failed to load %2%.") % __func__ % path.c_str();
                return res;
            }
            if (in.finest_nx != mm.finest_nx || in.finest_ny != mm.finest_ny
                || in.x_lo != mm.x_lo || in.x_hi != mm.x_hi
                || in.y_lo != mm.y_lo || in.y_hi != mm.y_hi) {
                log<LOG_ERROR>(L"%1% || merge-bank: %2% has an incompatible grid footprint; refusing to merge.")
                    % __func__ % path.c_str();
                return res;
            }
            int64_t carried = 0, dups = 0, dropped_pes = 0;
            int matched_cells = 0, dropped_cells = 0;
            for (int c = 0; c < in.n_cells; ++c) {
                const auto it = footprint_to_idx.find(
                    footprint_key(in.cell_i_bl[(size_t)c], in.cell_j_bl[(size_t)c],
                                  in.cell_step[(size_t)c]));
                const auto &pes = in.cell_pes[(size_t)c];
                if (it == footprint_to_idx.end()) {
                    ++dropped_cells;
                    dropped_pes += (int64_t)pes.size();
                    continue;
                }
                ++matched_cells;
                const int mi = it->second;
                for (const auto &pe : pes) {
                    if (seen[(size_t)mi].insert(pe_key(pe)).second) {
                        bank.cell_pes[(size_t)mi].push_back(pe);
                        ++carried;
                    } else {
                        ++dups;
                    }
                }
            }
            log<LOG_INFO>(L"%1% || merge-bank input %2%: cells matched=%3% dropped=%4%; "
                          L"PEs carried=%5%, duplicate-seed dropped=%6%, unmatched-cell dropped=%7%.")
                % __func__ % path.c_str() % matched_cells % dropped_cells
                % (long long)carried % (long long)dups % (long long)dropped_pes;
            if (dups > 0) {
                log<LOG_WARNING>(L"%1% || merge-bank: %2% bitwise-identical PEs dropped from %3% -- "
                                 L"input banks share RNG seeds. Generate grid banks with distinct --seed values.")
                    % __func__ % (long long)dups % path.c_str();
            }
            total_carried += carried; total_dups += dups; total_dropped += dropped_pes;
        }

        int empty_cells = 0;
        for (const auto &v : bank.cell_pes) if (v.empty()) ++empty_cells;
        const std::string bank_out = acfg.output_tag + "_bank.bin";
        if (save_bank(bank, bank_out)) {
            res.bank_path = bank_out;
            log<LOG_INFO>(L"%1% || merge-bank: wrote %2% -- %3% cells, %4% PEs carried "
                          L"(%5% duplicates + %6% unmatched dropped), %7% cells still empty. "
                          L"Next: --mode init-bank -o %8% to top up to criterion.")
                % __func__ % bank_out.c_str() % bank.n_cells % (long long)total_carried
                % (long long)total_dups % (long long)total_dropped % empty_cells
                % acfg.output_tag.c_str();
        }
        res.n_meta_cells = bank.n_cells;
        res.total_pes_generated = total_carried;
        res.mean_pes_per_cell = bank.n_cells > 0 ? (float)total_carried / (float)bank.n_cells : 0.0f;
        return res;
    }

    // ---- Mode: merge-brazil -------------------------------------------------
    // Union throws from >= 1 brazil archives (--merge-input) produced by
    // independent --mode brazil runs (e.g. grid jobs) into <tag>_brazil.bin,
    // then re-classify everything against this tag's <tag>_bank.bin and emit
    // the band PDF + ROOT — the same outputs as --mode brazil, with zero fits.
    //
    // Unlike merge-bank there is no per-cell harvesting: a throw is a full
    // per-cell dchi2 vector tied to the cell ORDERING of the mesh it was
    // computed on, so each input archive must match the current bank's
    // footprint (finest grid + n_cells) exactly or it is refused whole.
    //
    // Dedupe: the archive stores no per-throw RNG seed, but throws are driven
    // by PROseed::global_rng, so two runs launched with the same --seed
    // produce bitwise-identical observable rows. The dedupe key is therefore
    // the bitwise fingerprint of (global chi2, dchi2 vector) — hash plus an
    // exact byte compare on hash match — the direct analogue of merge-bank's
    // (seed, dchi2) key. Generate grid archives with distinct --seed values;
    // duplicates are dropped here with a warning.
    //
    // NOTE: like merge-bank, archives carry no XML/fit-config provenance, so
    // pooling throws from mismatched configs/metrics/presets is on the caller.
    if (acfg.mode == AdaptiveFCMode::MergeBrazil) {
        if (acfg.merge_inputs.empty()) {
            log<LOG_ERROR>(L"%1% || merge-brazil: need >= 1 --merge-input brazil archives.") % __func__;
            return res;
        }
        const std::string bank_in    = acfg.output_tag + "_bank.bin";
        const std::string brazil_bin = acfg.output_tag + "_brazil.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || merge-brazil: failed to load %2%. "
                           L"Run --mode merge-bank / init-bank (same -o) first.")
                % __func__ % bank_in.c_str();
            return res;
        }
        const int n_cells = bank.n_cells;

        // The output archive is built from the inputs ONLY (mirrors
        // merge-bank). If this tag already has an archive that isn't among
        // the inputs, its throws would be silently replaced — warn.
        {
            std::ifstream test(brazil_bin, std::ios::binary);
            if (test.is_open()
                && std::find(acfg.merge_inputs.begin(), acfg.merge_inputs.end(),
                             brazil_bin) == acfg.merge_inputs.end()) {
                log<LOG_WARNING>(L"%1% || merge-brazil: existing %2% is NOT among the --merge-input files and will be overwritten; "
                                 L"add it to --merge-input to keep its throws.")
                    % __func__ % brazil_bin.c_str();
            }
        }

        BrazilArchive merged;
        merged.finest_nx = bank.finest_nx;
        merged.finest_ny = bank.finest_ny;
        merged.n_cells   = n_cells;

        // Dedupe: hash -> accepted throw indices with that hash; exact byte
        // compare on match so a hash collision can never drop a real throw.
        std::unordered_map<uint64_t, std::vector<size_t>> seen;
        auto throw_fingerprint = [](float global_chi2, const std::vector<float> &dchi2) -> uint64_t {
            uint64_t h[2] = {0, 0};
            MurmurHash3_x64_128(dchi2.data(), (int)(dchi2.size() * sizeof(float)),
                                BrazilArchive::MAGIC, h);
            uint32_t gbits;
            std::memcpy(&gbits, &global_chi2, sizeof(gbits));
            return h[0] ^ (h[1] << 1) ^ ((uint64_t)gbits * 0x9E3779B97F4A7C15ULL);
        };
        auto identical_throw = [&merged](size_t prev, float g, const std::vector<float> &d) -> bool {
            if (std::memcmp(&merged.per_throw_global_chi2[prev], &g, sizeof(float)) != 0)
                return false;
            const auto &pd = merged.per_throw_dchi2[prev];
            return pd.size() == d.size()
                && std::memcmp(pd.data(), d.data(), d.size() * sizeof(float)) == 0;
        };

        int64_t total_carried = 0, total_dups = 0;
        for (const auto &path : acfg.merge_inputs) {
            BrazilArchive in;
            if (!load_brazil_archive(in, path)) {
                log<LOG_ERROR>(L"%1% || merge-brazil: failed to load %2%.") % __func__ % path.c_str();
                return res;
            }
            if (in.finest_nx != bank.finest_nx || in.finest_ny != bank.finest_ny
                || in.n_cells != n_cells
                || in.per_throw_global_chi2.size() != in.per_throw_dchi2.size()) {
                log<LOG_ERROR>(L"%1% || merge-brazil: %2% has an incompatible footprint "
                               L"(finest %3%x%4%, n_cells=%5% vs bank %6%x%7%, n_cells=%8%). "
                               L"Throws are tied to the mesh's cell set; refusing to merge.")
                    % __func__ % path.c_str()
                    % in.finest_nx % in.finest_ny % in.n_cells
                    % bank.finest_nx % bank.finest_ny % n_cells;
                return res;
            }
            int64_t carried = 0, dups = 0;
            for (size_t t = 0; t < in.per_throw_dchi2.size(); ++t) {
                const auto &d = in.per_throw_dchi2[t];
                if ((int)d.size() != n_cells) {
                    log<LOG_ERROR>(L"%1% || merge-brazil: %2% throw %3% has %4% cells (expected %5%) -- corrupt archive; refusing to merge.")
                        % __func__ % path.c_str() % (int)t % (int)d.size() % n_cells;
                    return res;
                }
                const float g = in.per_throw_global_chi2[t];
                auto &bucket = seen[throw_fingerprint(g, d)];
                bool dup = false;
                for (size_t prev : bucket) {
                    if (identical_throw(prev, g, d)) { dup = true; break; }
                }
                if (dup) { ++dups; continue; }
                bucket.push_back(merged.per_throw_dchi2.size());
                merged.per_throw_global_chi2.push_back(g);
                merged.per_throw_dchi2.push_back(d);
                ++carried;
            }
            log<LOG_INFO>(L"%1% || merge-brazil input %2%: throws carried=%3%, duplicate dropped=%4%.")
                % __func__ % path.c_str() % (long long)carried % (long long)dups;
            if (dups > 0) {
                log<LOG_WARNING>(L"%1% || merge-brazil: %2% bitwise-identical throws dropped from %3% -- "
                                 L"input archives share RNG seeds. Generate grid archives with distinct --seed values.")
                    % __func__ % (long long)dups % path.c_str();
            }
            total_carried += carried; total_dups += dups;
        }
        const int n_total = (int)merged.per_throw_dchi2.size();
        if (n_total == 0) {
            log<LOG_ERROR>(L"%1% || merge-brazil: no throws survived the merge; nothing written.") % __func__;
            return res;
        }

        save_brazil_archive(merged, brazil_bin);

        // Re-classify all merged throws against the current bank and emit the
        // band outputs — same tail as --mode brazil (see aggregate_brazil_throws).
        const int min_pes = std::max(10, acfg.n_pe_min);
        BrazilAggregation agg =
            aggregate_brazil_throws(bank, merged, acfg.cl_targets, min_pes, nthreads);
        emit_brazil_outputs(bank, bank_in, merged, agg, acfg, *model,
                            xaxis_idx, yaxis_idx, fakeDataParams);

        // Populate result.
        res.bank_path     = bank_in;
        res.n_meta_cells  = n_cells;
        res.n_throws_done = n_total;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = n_cells > 0 ? (float)total_pes / (float)n_cells : 0.0f;

        log<LOG_INFO>(L"%1% || merge-brazil done: %2% throws carried from %3% archive(s) "
                      L"(%4% duplicates dropped) across %5% cells; outputs %6%, %7%_brazil_band.pdf, %7%_brazil.root.")
            % __func__ % (long long)total_carried % (int)acfg.merge_inputs.size()
            % (long long)total_dups % n_cells
            % brazil_bin.c_str() % acfg.output_tag.c_str();
        return res;
    }

    // ---- Mode: print-mesh ---------------------------------------------------
    // Plot mesh binaries as PDFs. No fitting. With --merge-input, plots each
    // given file (e.g. a _cleanup_mesh.bin or a merged mesh); otherwise plots
    // this tag's <tag>_mesh.bin. PDF name = input filename with .bin → .pdf.
    if (acfg.mode == AdaptiveFCMode::PrintMesh) {
        std::vector<std::string> mesh_files = acfg.merge_inputs;
        if (mesh_files.empty()) mesh_files.push_back(acfg.output_tag + "_mesh.bin");
        int n_plotted = 0;
        for (const auto &path : mesh_files) {
            MetaMesh mm;
            if (!load_mesh(mm, path)) {
                log<LOG_ERROR>(L"%1% || print-mesh: failed to load %2%.") % __func__ % path.c_str();
                continue;
            }
            std::string pdf = path;
            if (pdf.size() > 4 && pdf.compare(pdf.size() - 4, 4, ".bin") == 0)
                pdf.resize(pdf.size() - 4);
            pdf += ".pdf";
            // n_throws <= 0: a loaded mesh carries no reliable throw/p_thresh
            // provenance (and derived merge/cleanup meshes have none at all).
            plot_metamesh_pdf(mm, *model, pdf, /*n_throws=*/0, acfg.p_thresh,
                              acfg.baseline_level, acfg.logx, acfg.logy,
                              xaxis_idx, yaxis_idx);
            res.n_meta_cells     = (int)mm.cells.size();
            res.n_baseline_cells = mm.n_baseline_cells;
            res.n_refined_cells  = mm.n_refined_cells;
            ++n_plotted;
        }
        log<LOG_INFO>(L"%1% || print-mesh: plotted %2% / %3% mesh file(s).")
            % __func__ % n_plotted % (int)mesh_files.size();
        return res;
    }

    // ---- Mode: print-bank ---------------------------------------------------
    // Loads an existing bank artifact and writes a summary PDF. No fitting.
    if (acfg.mode == AdaptiveFCMode::PrintBank) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || print-bank: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        const std::string out_pdf = acfg.output_tag + "_bank_summary.pdf";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        const std::string xlabel = xaxis_idx < model->nparams
            ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
        const std::string ylabel = yaxis_idx < model->nparams
            ? model->pretty_param_names.at(yaxis_idx) : std::string("y");

        plot_pebank_summary_pdf(bank, out_pdf, bank_in, xlabel, ylabel,
                                acfg.logx, acfg.logy, xlog_axis, ylog_axis);

        const std::string per_cell_pdf = acfg.output_tag + "_bank_per_cell.pdf";
        plot_pebank_pes_multipage_pdf(bank, per_cell_pdf, xlabel, ylabel,
                                      xlog_axis, ylog_axis);

        // Populate result for the caller (no PE generation in this mode).
        res.bank_path        = bank_in;
        res.n_meta_cells     = bank.n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = bank.n_cells > 0 ? (float)total_pes / (float)bank.n_cells : 0.0f;
        return res;
    }

    // ---- Mode: asimov -------------------------------------------------------
    // Loads an existing bank, builds the asimov dataset from fakeDataParams
    // (noise-free expected counts under the injected truth — same convention
    // as surface/fc/profile), classifies every cell against the bank's
    // per-cell critical Δχ² at each requested CL, and writes a verdict PDF.
    // No PE generation. Bank is read-only.
    if (acfg.mode == AdaptiveFCMode::Asimov) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || asimov: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        // Use the data spectrum PROfit assembled — that's where --inject,
        // --use-fake-data, --poisson-throw, --pseudo-experiment, and real data
        // all flow into via bin/PROfit.cxx:993 (`data = variable_data[i_prime]`).
        // Classifying against that here gives the user "FC contour for whatever
        // dataset is in the chain". For null-asimov use, the user runs PROfit
        // with --use-fake-data and no --inject (so data == CV-spectrum
        // collapsed).
        const PROdata &asimov_data = data;
        log<LOG_INFO>(L"%1% || asimov: classifying %2% cells against bank %3%; data nbins=%4%.")
            % __func__ % bank.n_cells % bank_in.c_str() % (int)asimov_data.Spec().size();

        // Outer throws bar wasn't displayed in asimov mode (see PROfit.cxx),
        // so finish_all(false) — no display refresh on a never-shown bar.
        progress.finish_all(false);
        std::vector<std::pair<int, std::string>> ab_cfg;
        ab_cfg.push_back({bank.n_cells, "AFC asimov cells"});
        MultiPROgressBar asimov_progress(ab_cfg);
        asimov_progress.initialize_display();
        asimov_progress.start_display_thread();

        AsimovObs obs = compute_asimov_obs(
            config, prop, systs, *model, fitconfig, asimov_data,
            acfg.chi2, acfg.binned, acfg.shape_only, xaxis_idx, yaxis_idx,
            bank.cell_center_x, bank.cell_center_y,
            proseed, nthreads, asimov_progress, 0);

        asimov_progress.finish_all(true);

        // Classify against the bank for every requested CL.
        const int min_pes_for_decision = std::max(10, acfg.n_pe_min);
        std::vector<std::vector<CellVerdict>> verdicts =
            classify_against_bank(bank, obs, acfg.cl_targets, min_pes_for_decision);

        // PDF (multipage, one page per CL).
        const std::string out_pdf = acfg.output_tag + "_asimov_verdict.pdf";
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        const std::string xlabel = xaxis_idx < model->nparams
            ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
        const std::string ylabel = yaxis_idx < model->nparams
            ? model->pretty_param_names.at(yaxis_idx) : std::string("y");
        plot_asimov_verdict_pdf(bank, obs, acfg.cl_targets, verdicts, out_pdf, bank_in,
                                xlabel, ylabel, acfg.logx, acfg.logy, xlog_axis, ylog_axis);

        // Clean publication-style contour overlay — the main asimov deliverable.
        // Inject-truth marker: pulled from fakeDataParams in model space, mapped
        // to physical for the canvas (same convention as the bank summary).
        const float truth_x_phys = xlog_axis
            ? std::pow(10.0f, fakeDataParams((int)xaxis_idx))
            : fakeDataParams((int)xaxis_idx);
        const float truth_y_phys = ylog_axis
            ? std::pow(10.0f, fakeDataParams((int)yaxis_idx))
            : fakeDataParams((int)yaxis_idx);
        const std::string contour_pdf = acfg.output_tag + "_asimov_contour.pdf";
        plot_asimov_contour_pdf(bank, obs, acfg.cl_targets, verdicts, contour_pdf,
                                xlabel, ylabel, acfg.logx, acfg.logy,
                                xlog_axis, ylog_axis,
                                /*draw_truth_marker=*/ true,
                                truth_x_phys, truth_y_phys);

        // ROOT artifact: contour TGraphs + per-cell TTree for downstream use.
        const std::string asimov_root = acfg.output_tag + "_asimov_contours.root";
        save_asimov_root(bank, obs, acfg.cl_targets, verdicts, asimov_root,
                         xlog_axis, ylog_axis);

        // Populate result.
        res.bank_path     = bank_in;
        res.n_meta_cells  = bank.n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = bank.n_cells > 0 ? (float)total_pes / (float)bank.n_cells : 0.0f;
        // Summary log lines per CL.
        for (size_t k = 0; k < acfg.cl_targets.size(); ++k) {
            int n_in = 0, n_out = 0, n_undec = 0;
            for (const auto &v : verdicts[k]) {
                if (!v.decidable) ++n_undec;
                else if (v.included) ++n_in;
                else ++n_out;
            }
            log<LOG_INFO>(L"%1% || asimov verdict CL=%2%: inside=%3%, outside=%4%, undecidable=%5%.")
                % __func__ % acfg.cl_targets[k] % n_in % n_out % n_undec;
        }
        return res;
    }

    // ---- Mode: brazil -------------------------------------------------------
    // Additive Brazil-band construction. Each invocation:
    //   1. Loads <tag>_brazil.bin if it exists and matches the bank footprint.
    //   2. Generates --n-brazil-throws NEW throws, appending observables to the
    //      archive (chi2_osc_global + per-cell dchi2_obs per throw).
    //   3. Saves the merged archive.
    //   4. Re-classifies ALL throws (old + new) against the current bank +
    //      cl_targets to derive per-throw verdicts. The archive deliberately
    //      stores only observables, not verdicts, so growing the bank or
    //      changing CLs between brazil runs reuses existing throw data.
    //   5. Aggregates per-cell inclusion fractions across all throws and emits
    //      the brazil-band PDF + ROOT TGraphs.
    //
    // (Substage 2 — adaptive bank top-up triggered by undecidable verdicts —
    // is a separate slice on top of this.)
    if (acfg.mode == AdaptiveFCMode::Brazil) {
        const std::string bank_in = acfg.output_tag + "_bank.bin";
        const std::string brazil_bin = acfg.output_tag + "_brazil.bin";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || brazil: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }

        const int n_new    = std::max(1, acfg.n_brazil_throws);
        const int n_cells  = bank.n_cells;
        const int n_cl     = (int)acfg.cl_targets.size();
        const int min_pes  = std::max(10, acfg.n_pe_min);

        // (1) Load existing archive if compatible. Otherwise initialise fresh.
        BrazilArchive arc;
        arc.finest_nx = bank.finest_nx;
        arc.finest_ny = bank.finest_ny;
        arc.n_cells   = n_cells;
        {
            std::ifstream test(brazil_bin, std::ios::binary);
            if (test.is_open()) {
                test.close();
                BrazilArchive existing;
                if (load_brazil_archive(existing, brazil_bin)
                    && existing.n_cells   == n_cells
                    && existing.finest_nx == bank.finest_nx
                    && existing.finest_ny == bank.finest_ny) {
                    arc.per_throw_global_chi2 = std::move(existing.per_throw_global_chi2);
                    arc.per_throw_dchi2       = std::move(existing.per_throw_dchi2);
                    log<LOG_INFO>(L"%1% || brazil: loaded existing archive %2% (%3% prior throws).")
                        % __func__ % brazil_bin.c_str() % (int)arc.per_throw_dchi2.size();
                } else {
                    log<LOG_WARNING>(L"%1% || brazil: existing archive %2% has mismatched footprint; ignoring and starting fresh.")
                        % __func__ % brazil_bin.c_str();
                }
            }
        }
        const int n_existing = (int)arc.per_throw_dchi2.size();
        const int n_total    = n_existing + n_new;

        log<LOG_INFO>(L"%1% || brazil: bank=%2% (%3% cells, %4% CLs, min_pes=%5%); existing=%6% throws, new=%7%, total after=%8%.")
            % __func__ % bank_in.c_str() % n_cells % n_cl % min_pes
            % n_existing % n_new % n_total;

        // Reserve room for new throws so we can index by absolute throw_idx.
        arc.per_throw_global_chi2.resize((size_t)n_total, 0.0f);
        arc.per_throw_dchi2.resize((size_t)n_total);
        for (int t = n_existing; t < n_total; ++t) {
            arc.per_throw_dchi2[(size_t)t].assign((size_t)n_cells, 0.0f);
        }

        // (2) Run new throws. Outer bar wasn't displayed for brazil mode
        // (see PROfit.cxx), so finish_all(false) — no spurious refresh.
        progress.finish_all(false);
        std::vector<std::pair<int, std::string>> bar_cfg;
        bar_cfg.push_back({n_new, "Brazil throws (new)"});
        MultiPROgressBar brazil_progress(bar_cfg);
        brazil_progress.initialize_display();
        brazil_progress.start_display_thread();

        std::vector<std::pair<int, std::string>> silent_cfg;
        silent_cfg.push_back({n_cells, "_silent"});
        MultiPROgressBar silent_progress(silent_cfg);

        // Throw-invariant: CV spectrum + covariance decomposition (an SVD)
        // hoisted out of the per-throw loop.
        PROspec brazil_cv = FillSpectra(config, prop, systs, *model, fakeDataParams,
                                        acfg.binned, config.i_prime);
        Eigen::MatrixXf brazil_L = systs.DecomposeFractionalCovariance(config, brazil_cv.Spec());

        for (int t_new = 0; t_new < n_new; ++t_new) {
            const int t_abs = n_existing + t_new;
            PROdata throw_data = generate_pseudo_experiment_data(
                config, prop, systs, *model, fakeDataParams, acfg.binned, brazil_L, proseed);

            AsimovObs obs = compute_asimov_obs(
                config, prop, systs, *model, fitconfig, throw_data,
                acfg.chi2, acfg.binned, acfg.shape_only, xaxis_idx, yaxis_idx,
                bank.cell_center_x, bank.cell_center_y,
                proseed, nthreads, silent_progress, 0);

            arc.per_throw_global_chi2[(size_t)t_abs] = obs.chi2_osc_global;
            for (int c = 0; c < n_cells; ++c) {
                arc.per_throw_dchi2[(size_t)t_abs][(size_t)c] = obs.dchi2_obs[(size_t)c];
            }

            log<LOG_INFO>(L"%1% || brazil throw %2%/%3% (abs=%4%): chi2_osc_global=%5%.")
                % __func__ % (t_new + 1) % n_new % t_abs % obs.chi2_osc_global;
            brazil_progress.increment_bar(0);
        }
        brazil_progress.finish_all(true);

        // (3) Persist the merged archive.
        save_brazil_archive(arc, brazil_bin);

        // (4)+(5) Re-classify ALL throws (existing + new) against the current
        //     bank, apply the closed-contour filter, and aggregate per-cell
        //     inclusion fractions — shared with --mode brazil-cleanup, see
        //     aggregate_brazil_throws above for the details.
        BrazilAggregation agg =
            aggregate_brazil_throws(bank, arc, acfg.cl_targets, min_pes, nthreads);

        // Outputs.
        const std::string brazil_pdf  = acfg.output_tag + "_brazil_band.pdf";
        const std::string brazil_root = acfg.output_tag + "_brazil.root";
        emit_brazil_outputs(bank, bank_in, arc, agg, acfg, *model,
                            xaxis_idx, yaxis_idx, fakeDataParams);

        // Populate result.
        res.bank_path    = bank_in;
        res.n_meta_cells = n_cells;
        int64_t total_pes = 0;
        for (const auto &v : bank.cell_pes) total_pes += (int64_t)v.size();
        res.total_pes_generated = total_pes;
        res.mean_pes_per_cell   = n_cells > 0 ? (float)total_pes / (float)n_cells : 0.0f;

        log<LOG_INFO>(L"%1% || brazil done: %2% new + %3% existing = %4% total throws across %5% cells; outputs %6%, %7%, %8%.")
            % __func__ % n_new % n_existing % n_total % n_cells
            % brazil_bin.c_str() % brazil_pdf.c_str() % brazil_root.c_str();
        return res;
    }

    // ---- Mode: brazil-cleanup ----------------------------------------------
    // Post-brazil mesh refinement targeting the Brazil ±2σ band edges, which
    // often fall in coarse baseline cells (the Wilks prepass only refined
    // around the Asimov contour, not the throw spread). Reads THIS tag's
    // <tag>_bank.bin (grid geometry only) + <tag>_brazil.root, takes the
    // SAVED band-edge contour TGraphs (--cleanup-quantiles, default
    // 0.025 / 0.975 — the exact curves the band PDF drew; nothing is
    // recomputed, so mesh and plot cannot disagree), and writes
    // <tag>_cleanup_mesh.bin: finest cells along the curves (± --cleanup-halo
    // bins), coarsest tiling elsewhere. Same finest grid + bounds as the
    // bank, so it union-merges with the mesh that made the band:
    //
    //   ... -o orig --mode brazil-cleanup
    //   ... -o v2   --mode merge-mesh --merge-input orig_mesh.bin orig_cleanup_mesh.bin
    //   ... -o v2   --mode merge-bank --merge-input orig_bank.bin
    //   ... -o v2   --mode init-bank  ...   (top up the new fine cells)
    //   ... -o v2   --mode brazil     ...   (re-throw; archives are per-mesh)
    if (acfg.mode == AdaptiveFCMode::BrazilCleanup) {
        const std::string bank_in     = acfg.output_tag + "_bank.bin";
        const std::string brazil_root = acfg.output_tag + "_brazil.root";
        PEBank bank;
        if (!load_bank(bank, bank_in)) {
            log<LOG_ERROR>(L"%1% || brazil-cleanup: failed to load %2%.") % __func__ % bank_in.c_str();
            return res;
        }
        if (acfg.cleanup_quantiles.empty()) {
            log<LOG_ERROR>(L"%1% || brazil-cleanup: empty --cleanup-quantiles.") % __func__;
            return res;
        }
        log<LOG_INFO>(L"%1% || brazil-cleanup: bank=%2% (%3% cells, grid geometry), contours from %4%, "
                      L"%5% CLs requested, %6% quantile levels, halo=%7%.")
            % __func__ % bank_in.c_str() % bank.n_cells % brazil_root.c_str()
            % (int)acfg.cl_targets.size() % (int)acfg.cleanup_quantiles.size()
            % acfg.cleanup_halo;

        const int W = bank.finest_nx, H = bank.finest_ny;
        const bool xlog_axis = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog_axis = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;
        int n_curves_used = 0;
        std::vector<uint8_t> flags = flag_bins_from_saved_brazil_contours(
            bank, brazil_root, acfg.cl_targets, acfg.cleanup_quantiles,
            xlog_axis, ylog_axis, acfg.cleanup_halo, n_curves_used);
        if (n_curves_used == 0) {
            log<LOG_ERROR>(L"%1% || brazil-cleanup: no matching contour TGraphs in %2% "
                           L"(need brazil_cl_<CL>_<q>_seg* for the requested --cl / --cleanup-quantiles). "
                           L"Run --mode brazil (same -o) first, and check the CLs match.")
                % __func__ % brazil_root.c_str();
            return res;
        }
        log<LOG_INFO>(L"%1% || brazil-cleanup: %2% saved contour curve(s) used.")
            % __func__ % n_curves_used;
        const int n_flagged = (int)std::count(flags.begin(), flags.end(), (uint8_t)1);
        if (n_flagged == 0) {
            log<LOG_WARNING>(L"%1% || brazil-cleanup: no quantile crossings found "
                             L"(band edges undecidable everywhere, or outside the grid); no mesh written.")
                % __func__;
            return res;
        }
        log<LOG_INFO>(L"%1% || brazil-cleanup: %2% / %3% finest bins flagged around the quantile contours.")
            % __func__ % n_flagged % (W * H);

        MetaMesh cleanup = build_mesh_from_flags(
            W, H, bank.max_levels,
            bank.x_lo, bank.x_hi, bank.y_lo, bank.y_hi,
            flags, acfg.baseline_level);
        const std::string mesh_out = acfg.output_tag + "_cleanup_mesh.bin";
        if (save_mesh(cleanup, mesh_out)) {
            log<LOG_INFO>(L"%1% || brazil-cleanup: wrote %2% (%3% cells). Next: "
                          L"--mode merge-mesh -o <new> --merge-input %4%_mesh.bin %2%; "
                          L"--mode merge-bank -o <new> --merge-input %5%; "
                          L"--mode init-bank -o <new>; --mode brazil -o <new>.")
                % __func__ % mesh_out.c_str() % (int)cleanup.cells.size()
                % acfg.output_tag.c_str() % bank_in.c_str();
        }
        res.n_meta_cells     = (int)cleanup.cells.size();
        res.n_baseline_cells = cleanup.n_baseline_cells;
        res.n_refined_cells  = cleanup.n_refined_cells;
        return res;
    }

    const std::string mesh_path = acfg.output_tag + "_mesh.bin";
    const std::string bank_path = acfg.output_tag + "_bank.bin";

    // ---- Mode: build-mesh ---------------------------------------------------
    // Wilks prepass + meta-mesh + diagnostics. Saves <tag>_mesh.bin and the
    // slice-1 PDFs. Does NOT generate PEs. Always rebuilds — running this mode
    // is the explicit request to (re)compute the mesh.
    if (acfg.mode == AdaptiveFCMode::BuildMesh) {
        std::vector<PROmesh::AMRResult> per_throw_meshes =
            generate_throws(config, prop, systs, *model, fitconfig, proseed,
                            fakeDataParams, acfg, nthreads, xaxis_idx, yaxis_idx, progress);
        res.n_throws_done = (int)per_throw_meshes.size();
        res.leaves_per_throw.reserve(per_throw_meshes.size());
        for (const auto &amr : per_throw_meshes) res.leaves_per_throw.push_back((int)amr.leaves.size());

        MetaMesh mm = build_meta_mesh(per_throw_meshes, acfg.p_thresh, acfg.baseline_level);
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;

        write_slice1_diagnostics(per_throw_meshes, mm, *model, systs, acfg,
                                 xaxis_idx, yaxis_idx, res);
        save_mesh(mm, mesh_path);

        progress.finish_all(true);
        log<LOG_INFO>(L"%1% || build-mesh done: cells=%2% (baseline=%3%, refined=%4%), saved %5%. "
                      L"Next step: --mode init-bank.")
            % __func__ % (int)mm.cells.size() % mm.n_baseline_cells % mm.n_refined_cells
            % mesh_path.c_str();
        return res;
    }

    // ---- Mode: init-bank ----------------------------------------------------
    // Loads <tag>_mesh.bin and generates the PE bank. Errors out if the mesh
    // doesn't exist — the user is expected to run --mode build-mesh first.
    if (acfg.mode == AdaptiveFCMode::InitBank) {
        MetaMesh mm;
        if (!load_mesh(mm, mesh_path)) {
            log<LOG_ERROR>(L"%1% || init-bank: required mesh artifact %2% not found. "
                           L"Run --mode build-mesh first.") % __func__ % mesh_path.c_str();
            progress.finish_all(false);
            return res;
        }
        log<LOG_INFO>(L"%1% || init-bank: loaded meta-mesh from %2% (cells=%3%).")
            % __func__ % mesh_path.c_str() % (int)mm.cells.size();
        res.n_meta_cells     = (int)mm.cells.size();
        res.n_baseline_cells = mm.n_baseline_cells;
        res.n_refined_cells  = mm.n_refined_cells;

        if (mm.cells.empty()) {
            log<LOG_WARNING>(L"%1% || empty meta-mesh; nothing to bank.") % __func__;
            progress.finish_all(false);
            return res;
        }

        const bool xlog = (xaxis_idx < model->is_log10.size()) ? model->is_log10[xaxis_idx] : acfg.logx;
        const bool ylog = (yaxis_idx < model->is_log10.size()) ? model->is_log10[yaxis_idx] : acfg.logy;

        std::vector<float> cell_x_model, cell_y_model;
        compute_cell_centers(mm, xlog, ylog, cell_x_model, cell_y_model);

        // Cholesky factor of the total covariance — built once, reused across all
        // cells and all PEs. Same construction as the brazil-band path
        // (bin/PROfit.cxx:1586) and the existing fc block (bin/PROfit.cxx:2511).
        PROspec cv = FillSpectra(config, prop, systs, *model, fakeDataParams,
                                 acfg.binned, config.i_prime);
        Eigen::MatrixXf L = systs.DecomposeFractionalCovariance(config, cv.Spec());

        log<LOG_INFO>(L"%1% || init-bank: starting PE generation for %2% cells "
                      L"(n_pe_min=%3%, n_pe_max=%4%, wilson_eps=%5%).")
            % __func__ % (int)mm.cells.size() % acfg.n_pe_min % acfg.n_pe_max % acfg.wilson_eps;

        PEBank bank;
        bank.finest_nx = mm.finest_nx;
        bank.finest_ny = mm.finest_ny;
        bank.max_levels = mm.max_levels;
        bank.x_lo = mm.x_lo; bank.x_hi = mm.x_hi;
        bank.y_lo = mm.y_lo; bank.y_hi = mm.y_hi;
        bank.n_cells = (int)mm.cells.size();
        bank.cell_center_x = cell_x_model;
        bank.cell_center_y = cell_y_model;
        bank.cell_i_bl.reserve(mm.cells.size());
        bank.cell_j_bl.reserve(mm.cells.size());
        bank.cell_step.reserve(mm.cells.size());
        bank.cell_level.reserve(mm.cells.size());
        for (const auto &c : mm.cells) {
            bank.cell_i_bl.push_back(c.i_bl);
            bank.cell_j_bl.push_back(c.j_bl);
            bank.cell_step.push_back(c.step);
            bank.cell_level.push_back(c.level);
        }

        // Top-up: try to carry forward an existing <tag>_bank.bin if its mesh
        // footprint matches. Each cell's PE list is preserved; schedule_pes
        // only generates additional PEs for cells not yet at the criterion.
        {
            std::ifstream test(bank_path, std::ios::binary);
            if (test.is_open()) {
                test.close();
                PEBank existing;
                if (load_bank(existing, bank_path)
                    && existing.n_cells   == bank.n_cells
                    && existing.finest_nx == bank.finest_nx
                    && existing.finest_ny == bank.finest_ny
                    && existing.max_levels == bank.max_levels) {
                    bank.cell_pes = std::move(existing.cell_pes);
                    int64_t existing_total = 0;
                    for (const auto &v : bank.cell_pes) existing_total += (int64_t)v.size();
                    log<LOG_INFO>(L"%1% || init-bank top-up: loaded existing bank %2% (%3% PEs / %4% cells). Generating additional PEs to meet criterion.")
                        % __func__ % bank_path.c_str() % (long long)existing_total % bank.n_cells;
                } else {
                    log<LOG_WARNING>(L"%1% || init-bank: existing bank at %2% has mismatched footprint; ignoring and regenerating.")
                        % __func__ % bank_path.c_str();
                }
            }
        }

        // Stop the throws progress bar (never displayed in init-bank). Launch a
        // PE-counted bar: bar size = total PEs to be added across all cells.
        // Updating per-PE (rather than per-cell) makes the bar smooth on
        // large meshes where each cell takes minutes.
        progress.finish_all(false);
        int total_pes_to_add = 0;
        for (int c = 0; c < bank.n_cells; ++c) {
            const int level = bank.cell_level[(size_t)c];
            // Mirror compute_to_add's level eligibility: only_layer >= 0 wins.
            if (acfg.only_layer >= 0) {
                if (level != acfg.only_layer) continue;
            } else {
                if (level < acfg.update_layer) continue;
            }
            const int n_existing = (c < (int)bank.cell_pes.size())
                ? (int)bank.cell_pes[(size_t)c].size() : 0;
            if (n_existing >= acfg.n_pe_max) continue;
            int to_add_raw;
            if (acfg.only_layer >= 0) {
                to_add_raw = acfg.n_pe_min;  // no doubling in only-layer mode
            } else {
                const int delta = std::max(0, level - acfg.update_layer);
                to_add_raw = (delta > 20)
                    ? acfg.n_pe_max
                    : (int)((int64_t)acfg.n_pe_min << delta);
            }
            total_pes_to_add += std::min(to_add_raw, acfg.n_pe_max - n_existing);
        }
        log<LOG_INFO>(L"%1% || init-bank: will add %2% PEs total across %3% cells.")
            % __func__ % total_pes_to_add % bank.n_cells;
        std::vector<std::pair<int, std::string>> cells_bar_cfg;
        cells_bar_cfg.push_back({std::max(1, total_pes_to_add), "AFC PEs added"});
        MultiPROgressBar cells_progress(cells_bar_cfg);
        cells_progress.initialize_display();
        cells_progress.start_display_thread();

        schedule_pes(acfg, config, prop, systs, *model, fitconfig, proseed, L,
                     xaxis_idx, yaxis_idx, cell_x_model, cell_y_model,
                     bank, nthreads, cells_progress, 0, res);

        cells_progress.finish_all(true);

        if (save_bank(bank, bank_path)) {
            res.bank_path = bank_path;
        }

        // Auto-print bank summary + per-cell PDFs.
        {
            const std::string summary_pdf = acfg.output_tag + "_bank_summary.pdf";
            const bool xlog_axis = xlog;
            const bool ylog_axis = ylog;
            const std::string xlabel = xaxis_idx < model->nparams
                ? model->pretty_param_names.at(xaxis_idx) : std::string("x");
            const std::string ylabel = yaxis_idx < model->nparams
                ? model->pretty_param_names.at(yaxis_idx) : std::string("y");
            plot_pebank_summary_pdf(bank, summary_pdf, bank_path, xlabel, ylabel,
                                    acfg.logx, acfg.logy, xlog_axis, ylog_axis);

            const std::string per_cell_pdf = acfg.output_tag + "_bank_per_cell.pdf";
            plot_pebank_pes_multipage_pdf(bank, per_cell_pdf, xlabel, ylabel,
                                          xlog_axis, ylog_axis);
        }

        return res;
    }

    // Classify / unknown — not yet implemented.
    log<LOG_ERROR>(L"%1% || mode '%2%' is not yet implemented.")
        % __func__ % (int)acfg.mode;
    progress.finish_all(false);
    return res;
}

} // namespace PROfit
