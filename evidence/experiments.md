# Candidates, holdouts and ablations

Reference model: V3 (previous entry), folds -38.530, practice -38.276. Every candidate was
tuned on the 23 walk-forward folds inside the kit history only. The two holdouts were scored
once each, at the end: A is the practice window (10 reads, 8-10 Nov), B is 11-12 Nov (6 reads).
Neither was used to choose a parameter.

## Scorer audit

- **Residual mass.** A place's named rows need not sum to 1; the scorer spreads the remainder
  over the games the forecast did not name there. With ~0.15 spare over ~90 unnamed games that
  is ~0.0017 each, so naming a game below ~0.002 is a loss and below ~0.0008 it is a bigger
  loss. This is where `MIN_P = 0.002` comes from. Naming every game (1e-4) costs 1.70 folds.
- **28-day age-out.** All 34 kit reads carry `details: null`, so no game has a birth date and
  the expiry path never fires on this data. It is kept for the final history, which may ship
  `details`.
- **Baseline reconstruction.** The public archive reproduces the practice window at -42.963
  against the published -43.512039, with top-10 hits identical at 6.4. The 0.55 gap is in
  residual handling, not in the reads.

## Candidates

| model | folds mean | median | worst | d folds | holdout A | holdout B |
|---|---|---|---|---|---|---|
| V3 | -38.530 | -39.051 | -45.701 | - | -38.067 | -34.839 |
| identity Markov | -45.900 | -45.240 | -57.600 | -7.37 | -43.390 | -47.570 |
| place Markov | -43.400 | -45.370 | -52.000 | -4.87 | -36.790 | -44.410 |
| calibrated diffusion | -41.160 | -41.160 | -45.850 | -2.63 | -39.050 | -38.300 |
| adaptive ensemble (w=0.1) | -44.410 | | | -5.88 | -43.380 | -34.890 |
| V6: retuned V3 + stay-spike | -38.150 | -38.940 | -45.150 | +0.38 | -38.500 | -33.940 |
| **V7: V3 + CCU tilt** | **-38.026** | **-38.503** | **-44.710** | **+0.50** | **-37.415** | **-34.004** |

V6 was rejected: better on folds and holdout B, worse on the practice window (-38.502 vs
-38.276), and its gain came from seven constants retuned together. V7 wins on all four sets and
rests on a feature with a measured information coefficient, so it is the entry.

## External data

- `IvanKuria/bloxscout-data`: hourly and daily CCU rollups, but the dataset stops 2026-07-13, so
  it says nothing about this window and was not used.
- `robloxgamestats.com/api/v1`: `/games/{id}/series?days=14` (hourly CCU) and
  `/games/{id}/ranks?days=30` (daily chart place), fetched unauthenticated for all 273 games in
  the kit and the archive before the deadline. Used to measure feature power, never called at
  runtime.
- `roterminal.co/api/v1/charts/up-and-coming`: carries `days_on_chart`, `move`, `yesterday`,
  `best_place`; median tenure on the chart is 5 days. Those fields are absent from the graded
  input, so no per-game table could be used honestly.

### Feature power

Spearman IC against the change in place over the next 24 / 48 / 72 h, 2720-3000 (game, cutoff)
pairs. Negative means "feature high, game climbs".

| feature | 24h | 48h | 72h |
|---|---|---|---|
| heat: CCU vs own week | -0.235 | -0.215 | -0.160 |
| decay: CCU vs own 7d peak | -0.234 | -0.207 | -0.135 |
| CCU slope, 24 h | -0.210 | -0.195 | -0.175 |
| CCU ratio, 3 h vs 21 h | -0.186 | -0.169 | -0.137 |
| days on chart | +0.158 | +0.153 | +0.129 |
| CCU slope, 6 h | -0.078 | -0.079 | -0.042 |
| CCU acceleration | -0.007 | -0.010 | +0.020 |

## Ablations of the tilt (folds of V3 with one change)

| change | folds | d |
|---|---|---|
| no tilt (V3) | -38.530 | - |
| decay only, weight 0.4 | -38.132 | +0.398 |
| heat only, weight 0.25 | -38.170 | +0.359 |
| decay 0.4 + heat 0.2 | -38.026 | +0.504 |
| decay 0.4 + ratio 0.15 | -38.095 | +0.434 |
| decay 0.8 + heat 0.35 | -38.543 | -0.014 |
| tilt on days-on-chart tenure | -39.068 | -0.538 |
| tilt on 24 h CCU slope | -38.861 | -0.331 |
| all four features together | -38.468 | +0.062 |
| tilt faded with horizon (tau 60 h) | -38.090 | +0.440 |
| shift the predicted place instead of reweighting | -39.172 | -0.642 |

Tenure fails because survival and expiry already model "old games leave"; slope fails because
24 h of CCU is mostly the diurnal curve; the horizon-faded tilt and the place shift both lose to
the flat multiplicative tilt.

## Unseen-mass calibration

Measured share of top-10 places held by games outside the candidate pool, by day of the horizon:
0.7% (0-24 h), 5.4% (24-48 h), 17% (48-72 h), on the kit history. Applying that per hour scored
-38.145 folds and -37.812 practice, both worse than the single window-average rate (-38.026 and
-37.608): the survival and capacity terms already discount known mass with horizon, so the
second discount double-counts. The sliced version was reverted.
