# Where -13.2 comes from, and why it cannot repeat

The practice phase's ten reads are not secret. They are commits of
`data/discover/latest.json` in `Kenny-Dhub/roblox-snapshots`, the public archive the kit's
`history.ndjson` is built from. Rebuilt from git history, the read sequence is:

```
2026-11-08T00:37:14Z  2026-11-08T07:22:10Z  2026-11-08T15:13:05Z  2026-11-08T19:42:36Z
2026-11-08T23:43:21Z  2026-11-09T06:31:34Z  2026-11-09T17:30:12Z  2026-11-10T01:32:21Z
2026-11-10T09:51:34Z  2026-11-10T17:51:16Z
```

Ten reads inside `2026-11-07T23:00Z .. 2026-11-10T23:00Z`, exactly the practice window. The
archive is public and near-hourly, so anyone can read those ten charts and fit a forecast to
them. The board's leader is titled "Fitted practice table + self-calibrating stack forecaster",
and 33 entries hold the same -13.2 with the same 75.5 agreement: one fitted table, copied.

The final window has reads that do not exist yet at submission time, so a fitted table has
nothing to fit. Every entry, the fitted ones included, falls back to whatever generalises.

## What an honest forecast scores

| model | practice window | 23 walk-forward folds |
|---|---|---|
| house baseline (persistence) | -42.963 | -47.297 |
| V3, the previous entry | -38.276 | -38.530 |
| this entry | -37.608 | -38.026 |

Every honest model lands in -35..-45. The reason is measurable: of the games in the top 10 at
one read, only 41.5% sit at the same place at the next read (median gap 6.4 h), and the
destination is close to uniform over the top ten at gaps beyond 4 h. A place therefore cannot
honestly take much more than 0.05-0.15 for its most likely game, and 15% of places at 48-72 h
are held by games nobody has seen. A forecast that knows the chart's membership but not its
order scores -23 at best; this entry scores -37.6, with 76% of place-truths already named in
the right place row.
