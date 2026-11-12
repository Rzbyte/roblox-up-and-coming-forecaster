# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun's `roblox-up-and-coming-forecast-01` ($500, deadline 2026-11-13T06:00:00Z).

`main.py` reads `/input/history.ndjson` + `/input/task.json` and writes `/output/forecast.json`:
one probability row per (game, place) plus a guessed top 10. Python 3.13, standard library
only, no network, deterministic (two runs match byte for byte), ~1.4 s per run.

## Method

The chart reorders fast but keeps its members. In the public archive (45 reads, 3.6 to 11 h
apart) only 0.0-1.5% of top-10 places within 28 h are held by a game absent from the previous
read, while 42 of 44 consecutive read pairs change the order, a median of 6 places. So the
forecast is a per-place distribution over *known* games, mixed over the forecast horizon.

For each candidate game:

- a log-rank level `score = -ln(place)` from its rank at the end of the history, blended with a
  weighted slope over the last reads (a game that has been climbing keeps a small drift);
- a per-hour survival factor, plus a hard stop at the chart's 28-day eligibility limit, so a
  game that ages out before the window ends stops taking mass;
- a displacement density around that level (Gaussian core + Student-type tail) whose width grows
  as `SD_BASE + volatility * sqrt(hours)`, with `volatility` measured from the history's own
  read-to-read rank churn, so a calm archive forecasts sharply and a churning one diffusely;
- a small probability on a "surge" alternative state (a player-count spike pulling a game up
  before its rank catches up), and a 20% anchor to its own older occupancy.

A place's mass is then renormalised, discounted by an unknown-entrant share fitted from rolling
cuts of the history (the share of top places held by games that were not in the candidate pool),
and capped so that no game holds more than ~1.0 across the ten places. Places keep a residual
unnamed share, which the scorer spreads over the games the forecast did not name there.

## Evidence

All numbers below were produced with the harness in `tools/` against the public archive
(`Kenny-Dhub/roblox-snapshots`, the same source the kit's history comes from).

| what | log score | typical p per place | top-10 hits |
|---|---|---|---|
| house baseline (recomputed; kit publishes -43.512039) | -42.96 | 0.0136 | 6.4 |
| GPT "v3" forecaster as shipped | -38.13 | 0.0221 | 6.4 |
| this entry, practice window | see `evidence/dev_score.txt` | | |
| this entry, 23 walk-forward folds | see `evidence/backtest.txt` | | |
| board leader (fitted practice table, 33 copies) | -13.2 | 0.267 | 75.5 agreement |

The practice window's ten reads (2026-11-08T00:37 to 2026-11-10T17:51) are *public*: they are
commits of `data/discover/latest.json` in the archive. A forecast fitted to those ten reads
scores about -13 on the practice board and cannot repeat it on the final window, whose reads
happen after entries close. The honest ceiling measured on real data is around -35 to -39.

## Running it

```sh
KIT=../roblox-up-and-coming-forecast-01-kit
MAKEFUN_INPUT_DIR=$KIT/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Harness

- `tools/rebuild_archive.py` rebuilds the archive's chart-read sequence from git history.
- `tools/score_dev.py` scores a forecast on the practice window; it reproduces the house
  baseline to within 0.55 of the published -43.512039 (hits match exactly at 6.4).
- `tools/eval.py` runs walk-forward folds: fit on the history up to a cut, score on the reads in
  the next 72 h, over 23 cuts.
- `tools/tune.py` grid-searches model constants on the folds, never on the practice window.

MIT licensed.
