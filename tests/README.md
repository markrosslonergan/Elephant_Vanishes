# PROfit short test suite

Two scripts for fast, deterministic, end-to-end regression testing of PROfit —
not CI, just something you run by hand before/after a change.

## Quick start

```bash
# Run the suite once per code state you want to compare:
tests/run_short_tests.sh ref          # e.g. on the base branch build
tests/run_short_tests.sh mybranch     # after rebuilding with your changes

# Compare the two:
tests/compare_tags.sh ref mybranch
# -> tests/runs/compare_ref_vs_mybranch.md
```

## What it runs

`run_short_tests.sh <TAG> [XML]` processes the fake ND+FD SBN MC
(`working_dir/Neutrino2026/fake_sbn_v2.xml` by default; the hardcoded
`/exp/...` MC paths are rewritten to sit next to the XML) and then runs one
short, seeded (`--seed 405 -n 1 --preset fast fast`) instance of each workflow:

| Tests | Coverage |
|---|---|
| t00 | `process` (MC → `_prop.bin`/`_syst.bin`) |
| t01–t06 | `global` with neyman (PROchi) / CNP (t02, legacy spelling `-c PROCNP` — covers the alias path) / poisson (t03, legacy `-c Poisson`) / pearson (t03b) / `--statonly` / `--inject` / `--pseudo-experiment` |
| t07 | `profile` (all parameters) |
| t08–t09 | `surface` dense grid and `--surface-amr` |
| t10–t12 | `plot` with `--with-splines`, `--scale-by-width`, `--bkg-subtract` |
| t12s* | `--shapeonly`: global (neyman/CNP/poisson), profile, fc, and `--scale-by-width plot` |
| t13 | `fc` (2 universes) |
| t14 (×5) | full `fc-adaptive` chain: build-mesh → init-bank → print-bank → asimov → brazil |
| t15–t16 | `mcmc` (1 chain), `scale-test` benchmark smoke |
| t17–t19 | PROjector: ND pre-fit → projected `global` → projected `fc` |
| t20–t21 | PROjector negative tests (partial-channel and match-everything patterns must be refused) |

Outputs land in `tests/runs/<TAG>/` with per-test logs in `logs/` and a
PASS/FAIL `summary.txt`. Exit code = number of failures.

Everything is single-threaded and fixed-seed, so **two runs of identical code
produce byte-identical `.bin`/`.txt` artifacts** and semantically identical
ROOT files (ROOT file bytes always differ via embedded timestamps).

## What the comparison does

`compare_tags.sh <TAG1> <TAG2>` walks the two run directories and writes a
markdown report (`tests/runs/compare_<TAG1>_vs_<TAG2>.md`):

- `.bin` / `.txt` / `.xml` — byte-level `cmp` (expected IDENTICAL for
  identical physics),
- `.root` — dumped to text with `tests/dump_root.C` (histogram cells, graph
  points, per-leaf tree statistics) and diffed (MATCH/DIFFER),
- `.pdf` — size note only (never comparable, never fails the compare),
- `summary.txt` — PASS/FAIL verdicts compared, timings ignored,
- missing/extra files flagged.

Exit code = number of DIFFER + MISSING findings, so it can gate a script.

## Knobs

Environment variables (see script headers for the full list):
`PROFIT_BIN` (default `build/bin/PROfit`), `PROFIT_TEST_MCDIR` (where the
`fake_sbn_mc_{ND,FD}.root` files live), `PROFIT_TEST_OUTDIR`,
`PROFIT_TEST_TIMEOUT` (per test, default 1800 s), `ROOTEXE` (for the dumper;
auto-detected from PATH or `/usr/local/root/root/bin/root`).

Caveats: comparisons are only meaningful between runs on the **same machine
with the same MC directory** (the localized XML path enters the config hash),
and `-n 1` must not be raised (thread scheduling breaks AMR determinism).
