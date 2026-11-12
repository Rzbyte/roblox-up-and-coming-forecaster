# V4 study: three candidates, two holdouts, one decision

Reference entry: V3 (repo `main.py` at commit beebba2). All tuning happened on the 23
walk-forward folds inside the kit history; the two holdouts were scored once, at the end.

## Scorer audit first

- **Residual / unnamed mass.** A place's named rows need not sum to 1. Whatever is left is
  spread by the scorer over the games on that read that the forecast did not name there. So
  leaving a game unnamed pays `spare / unnamed_count`, not zero. With ~0.15 spare and ~90
  unnamed games that is about 0.0017, which is why naming a game at p < 0.002 is roughly
  break-even and below ~0.0008 it is a loss. The entry keeps `MIN_P = 0.002`; tests at 0.0005,
  0.005 and 0.012 moved folds by -0.26, +0.25 and +0.08 respectively, i.e. noise-level.
- **House baseline check.** Rebuilding the practice window from the public archive gives
  -42.963 against the published -43.512039, with top-10 hits matching exactly (6.4). The 0.55
  gap is in the residual handling, not in the reads: the read timestamps are right, so the
  scorer is faithful to about one place per read.
- **28-day age-out.** All 34 kit reads carry `details: null`, so `born` is empty and the
  expiry path never fires on this data. It is dead code here; it only matters if the final
  history ships `details`. V3 keeps it, and V4 did not change it.
- **Fit of the unknown share.** `fit_unnamed` measures the share of top places held by games
  outside the candidate pool, shrunk toward a 0.30/0.19 prior. On 34 reads it lands near 0.14
  on average, matching the archive's own 0-15% by horizon.

## Candidates

| model | idea |
|---|---|
| V4a | identity Markov: P(game at place k) from observed rank transitions, plus an occupancy prior |
| V4a2 | place Markov: P(dst place \| src place), horizon-averaged, with occupancy prior and continuity weight |
| V4b | Gaussian/tail rank diffusion with empirically calibrated sd, spike and unnamed share, no persistence anchor |
| V4c | adaptive blend of V3 with V4a2 (weights 0.1-0.3), renormalised per place |

## Results (log score, higher is better)

| model | folds mean | folds median | folds worst | Δ folds | holdout A (10 reads, 08-10 Nov) | holdout B (6 reads, 11-12 Nov) |
|---|---|---|---|---|---|---|
| **V3 (entry)** | **-38.53** | -39.05 | -45.70 | baseline | **-38.28** | -34.84 |
| V4a identity Markov | -45.90 | -45.24 | -57.60 | -7.37 | -43.39 | -47.57 |
| V4a2 place Markov | -43.40 | -45.37 | -52.00 | -4.87 | -36.79 | -44.41 |
| V4b calibrated diffusion | -41.16 | -41.16 | -45.85 | -2.63 | -39.05 | -38.30 |
| V4c ensemble (w=0.1) | -44.41 | | | -5.88 | -43.38 | -34.89 |
| V6 tuned V3 (+stay spike) | -38.15 | -38.94 | -45.15 | +0.38 | -38.50 | -33.94 |

V6 is V3 with a measured same-place persistence spike and retuned constants
(SD_BASE 0.16, VOL_MULT 1.20, MIN_P 0.005, PERSIST_W 0.10, UNNAMED_SCALE 0.55, STAY_MULT 1.30).
Paired per-fold delta is +0.377 with sd 0.411 (t = 4.39, 17 of 23 folds better), and it is better
on holdout B by 0.90 - but worse on holdout A (the practice window) by 0.22 when scored with the
harness that reproduces the published baseline. Two of three sets favour it, one contradicts, and
the margin is a quarter of a log point.

## Ablations (folds, V6 constants)

| change | folds | Δ |
|---|---|---|
| no stay spike | -38.19 | -0.04 |
| no persistence anchor | -38.11 | +0.04 |
| pure Gaussian, no heavy tail | -38.14 | +0.02 |
| no unknown mass | -38.12 | +0.04 |
| name every game (MIN_P 1e-4) | -39.86 | -1.70 |
| unknown unscaled (1.0) | -38.25 | -0.09 |

## Decision

V3 stays the entry. V4a, V4a2, V4b and the ensemble all lose on the folds and, except V4a2 on
holdout A, lose on the holdouts too; V4a2's holdout A win (-36.79) flips to -44.41 on holdout B,
which is the signature of a model that fits one chart regime. V6 is a hair better on two of
three sets and worse on the graded practice window, so it is not proven better out of sample;
per the rule "keep V3 unless V4 wins on unused data", it stays a documented alternative
(`tools/model_v6_alternative.py`) and is not submitted.

The honest ceiling this data supports is around -38. Every component that predicts *which game*
beyond ~12 hours is worth less than 0.5 log points; the rest is chart noise plus new entrants.
Beating -13.2 requires the fitted practice table, which cannot exist for the final window.
