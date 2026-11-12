# Where -13.2 comes from, and why it cannot repeat

The practice phase's ten reads are not secret. They are commits of
`data/discover/latest.json` in `Kenny-Dhub/roblox-snapshots`, the public archive the kit's
`history.ndjson` is built from. Rebuilt from git history, the read sequence is:

```
2026-11-08T00:37:14Z  2026-11-08T07:22:10Z  2026-11-08T15:13:05Z  2026-11-08T19:42:36Z
2026-11-08T23:43:21Z  2026-11-09T06:31:34Z  2026-11-09T17:30:12Z  2026-11-10T01:32:21Z
2026-11-10T09:51:34Z  2026-11-10T17:51:16Z
```

Ten reads inside `2026-11-07T23:00Z .. 2026-11-10T23:00Z`, exactly the practice window RULES.txt
describes. The archive is public and hourly-ish, so anyone can read those ten charts and fit a
forecast to them. The board's leader is titled "Fitted practice table + self-calibrating stack
forecaster", and 33 entries hold the same -13.2 with the same 75.5 agreement: one fitted table,
copied.

The final window (`2026-11-13T06:00Z .. 2026-11-16T06:00Z`) has reads that do not exist yet at
submission time, so a fitted table has nothing to fit. Every entry, the fitted ones included,
falls back to whatever generalises.

## What an honest forecast scores

Harness in `tools/`. `score_dev.py` recomputes the house baseline at -42.96 against the published
-43.512039 (top-10 hits match exactly at 6.4, so the reconstruction of the ten reads is right to
about one place per read).

| model | practice window | 23 walk-forward folds |
|---|---|---|
| house baseline (persistence, kit) | -42.96 | -47.30 |
| GPT "v3" diffusion forecaster | -38.13 | -38.57 |
| this entry | -38.28 | -38.53 |
| empirical rank-transition kernel (tested, rejected) | -45.28 | -43.65 |
| distance-kernel family, best of 108 configs | -39.26 | -42.37 |

Every honest model lands in -35..-45. The reason is measurable in the archive: of the games in
the top 10 at one read, only 41.5% are at the same place at the next read (median gap 6.4 h), and
the destination is close to uniform over the top ten at gaps over 4 h. A place therefore cannot
honestly take much more than ~0.05-0.15 for the most likely game, which is a log score near -2.3
to -3.0 per place, or -23 to -30 per read at best, and the read-to-read churn plus 15% new
entrants at 48-72 h push the realised figure to about -38.
