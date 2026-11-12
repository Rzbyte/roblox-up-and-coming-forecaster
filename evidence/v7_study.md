# External data: what actually predicts 72h rank moves

V3 folds: -38.530. Candidate V7 folds: -38.026 (+0.504). Practice window -38.276 -> -37.608.
Measured on the same 23 folds and on two holdouts that were not used for tuning.

## Sources

| source | verdict |
|---|---|
| `IvanKuria/bloxscout-data` | hourly + daily CCU rollups, but the dataset stops at 2026-07-13. Nothing from Oct-Nov 2026, so it can only supply generic priors; not used. |
| `robloxgamestats.com/api/v1` | works unauthenticated: `/games/{id}/series?days=14` hourly CCU and `/games/{id}/ranks?days=30` daily up-and-coming place. Fetched for all 273 games that appear in the kit or the public archive, cached in `/tmp/rbxext`. |
| `roterminal.co/api/v1/charts/up-and-coming` | works and carries `days_on_chart`, `move`, `yesterday`, `best_place`; median tenure on the chart is 5 days. Those fields are not in the official input, so only a learned curve could ship, and that curve did not help (see below). |

## Feature information coefficients

Spearman IC against the change in place over the next 24/48/72h, over 2720-3000 (game, cutoff)
pairs. Negative means "feature high -> game climbs".

| feature | 24h | 48h | 72h |
|---|---|---|---|
| ccu_z (vs own week) | **-0.235** | -0.215 | -0.160 |
| decay = ln(cur/7d peak) | **-0.234** | -0.207 | -0.135 |
| ccu_slope24 | -0.210 | -0.195 | **-0.175** |
| ccu_ratio (3h vs 21h) | -0.186 | -0.169 | -0.137 |
| tenure_d (days on chart) | +0.158 | +0.153 | +0.129 |
| ccu_slope6 | -0.078 | -0.079 | -0.042 |
| ccu_accel | -0.007 | -0.010 | +0.020 |
| best place in last 7d | +0.039 | +0.062 | +0.024 |

Games trading below their own weekly peak and above their weekly mean keep climbing; older
games on the chart drift down. Short-window slope and acceleration carry almost nothing, so
momentum is a level effect over days, not an hour-to-hour effect.

## V7: tilt V3 by the two robust features

`p_i <- p_i * exp(0.40 * z(decay_i) + 0.20 * z(ccu_z_i))`, capped at 6x either way, computed
only from the `playing` values in the history. No network, no external table at runtime.

| weights | folds | median | worst | d folds | holdout A | holdout B |
|---|---|---|---|---|---|---|
| V3 | -38.530 | -39.051 | -45.701 | - | -38.067 | -34.839 |
| decay 0.4 + z_ccu 0.2 | **-38.026** | -38.503 | -44.710 | **+0.504** | **-37.415** | **-34.004** |
| decay 0.4 + ratio 0.15 | -38.095 | | | +0.434 | -37.174 | -33.997 |
| decay 0.6 | -38.122 | | | +0.407 | -37.421 | -34.174 |
| decay 0.4 | -38.132 | | | +0.398 | | |
| z_ccu 0.25 | -38.170 | | | +0.359 | | |
| decay 0.8 + z_ccu 0.35 | -38.543 | | | -0.014 | | |

Official-style harness on the adopted weights: practice window **-37.608** (V3 -38.276),
top-10 hits **7.0** (V3 6.4), 23 folds **-38.026**, deterministic across two runs, 0.9 s.

## Failures, measured

- **tenure_d tilt**: best single-feature IC at 24h (+0.158) but tilting on it costs 0.538 folds
  (-39.068). The IC is doing the work of "old games are about to age out", which V3's survival
  and expiry terms already model; adding it double-counts.
- **slope24 tilt**: -0.331 folds. Slope over 24h is mostly diurnal shape, not momentum.
- **all four features together**: +0.062 folds, i.e. the helpful two are cancelled by the other two.
- **weight 0.8**: gains vanish and turn negative, so the tilt has a narrow useful band, a sign it
  is a real but weak effect.
- **bloxscout-data**: unusable for this window (ends 2026-07-13).
- **roterminal fields**: not present in the graded input, so any per-game table would be stale by
  the final window and was not used.

## Status

Candidate lives at `tools/candidate_v7.py`. The entry (`main.py`) is still V3 and unchanged;
swapping in V7 needs an explicit go-ahead.
