# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun's `roblox-up-and-coming-forecast-01`. `main.py` reads `/input/task.json`
and `/input/history.ndjson`, writes `/output/forecast.json`: one probability row per
(game, place) plus a guessed top 10. Python 3.13 standard library only, no network, no model
calls, deterministic (two runs match byte for byte), about one second per run.

## Model

Each place is a mixture of latent log-rank states, one per candidate game, plus an explicit
pool for games that are not in the history yet.

- **Drift.** Time-weighted slope of `-ln(rank)` over the last 40 h, half-life 14 h, damped by
  sample count, clamped to +/-0.075/h.
- **Diffusion.** Spread `0.17 + vol*sqrt(hour)*1.08`, `vol` measured from the history's own
  hourly rank churn, with a Student-type tail (weight 0.28) so a 12-place jump is unlikely
  rather than impossible.
- **Occupancy.** Each game's realised place share over the last 96 h, blended at 20%, which
  anchors the far horizon against overreacting to the newest snapshot.
- **Survival and age-out.** Per-hour exit rate `exp(-t*(0.0018 + 0.000028*rank))` plus a hard
  stop at the chart's 28-day eligibility: an expired game holds no probability, and the mass
  it would have held stays unnamed instead of being redistributed.
- **CCU momentum.** `decay = ln(current CCU / weekly peak)` and `heat = z(current CCU vs its
  own weekly mean)`, both computed from the `playing` values already in the history, tilt each
  candidate by `exp(0.40*z_decay + 0.20*z_heat)`. Measured on 2720-3000 (game, cutoff) pairs of
  external hourly CCU history, the rank correlation with the 24 h change in place is -0.234 for
  `decay` and -0.235 for `heat` (-0.135 and -0.160 at 72 h): a fading game loses places, a hot
  one gains them.
- **Unseen entrants.** The share of each place held by games outside the candidate pool is
  fitted on up to ten older cuts with a full 72 h follow-up and shrunk toward a prior (0.30 to
  0.19 by place). The complement stays unassigned on purpose.
- **Filtering.** A row below `p = 0.002` is dropped: the unnamed pool pays about 0.0017 per
  game, so a token guess is worth less than silence.

## Scores (log score, higher is better)

| model | practice window (10 reads) | 23 walk-forward folds | holdout A | holdout B |
|---|---|---|---|---|
| house baseline, rebuilt locally | -42.963 | -47.297 | | |
| V3 (previous entry) | -38.276 | -38.530 | -38.067 | -34.839 |
| **this entry** | **-37.608** | **-38.026** | **-37.415** | **-34.004** |

Folds of this entry: median -38.503, best -29.690, worst -44.710; top-10 hits 7.0 per read
(V3: 6.4). The published house baseline on the practice window is -43.512039; the local rebuild
lands within 0.55 with identical hit counts, which is the noise floor of the scorer
reconstruction.

Rejected on evidence: rank-transition Markov (-45.90 folds), place-Markov (-43.40), calibrated
Gaussian diffusion (-41.16), adaptive ensemble (-44.41), retuned V3 with a stay-spike (-38.15
folds but -38.50 practice), and a 24 h-sliced unseen share (-38.145 folds, -37.812 practice).
Every candidate and ablation is in `evidence/experiments.md`.

## Running it

```sh
KIT=../roblox-up-and-coming-forecast-01-kit
MAKEFUN_INPUT_DIR=$KIT/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` the entry.
- `evidence/experiments.md` candidates, ablations, fold and holdout numbers.
- `evidence/why_not_13.md` why the top of the board sits at -13.2 and this entry does not chase it.
- `.makefun/owner.json` wallet anchor for the repository owner check.

MIT licensed.
