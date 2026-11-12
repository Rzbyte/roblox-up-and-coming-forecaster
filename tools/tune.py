#!/usr/bin/env python3
"""Grid-search a small family of place-conditional forecast shapes on real folds."""
import json, math, itertools, statistics
from datetime import datetime
from pathlib import Path

KIT = Path.home() / "makefun/rbx/roblox-up-and-coming-forecast-01-kit/input/history.ndjson"
PLACES = 10


def epoch(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def load():
    rows = []
    for line in open(KIT, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        g = {int(x['universe_id']): int(x['rank']) for x in r.get('up_and_coming') or []}
        if g:
            rows.append((epoch(r['at']), g))
    rows.sort()
    return rows


def build(rows, i, tau, boost, spare, tau_lag=1.0, dist_pow=1.0):
    """Forecast at cutoff i: place k mass over candidates by rank distance."""
    _, rank = rows[i]
    cand = {u: r for u, r in rank.items() if r <= 25}
    out = {}
    for k in range(1, PLACES + 1):
        w = {}
        for u, r in cand.items():
            d = abs(r - k)
            val = math.exp(-d / tau)
            if r == k:
                val *= boost
            if d > 0:
                val *= math.exp(-0.5 * ((math.log(max(1, r)) - math.log(k)) ** 2) / (tau_lag ** 2)) if tau_lag else 1.0
            if val > 1e-6:
                w[u] = val ** dist_pow
        s = sum(w.values())
        if s <= 0:
            continue
        keep = 1.0 - spare
        for u, v in w.items():
            p = keep * v / s
            if p >= 0.002:
                out[(u, k)] = p
    return out


def score(fc, read_rank):
    order = [u for u, _ in sorted(read_rank.items(), key=lambda kv: kv[1])]
    total = 0.0
    for k in range(1, PLACES + 1):
        p = fc.get((order[k - 1], k))
        if p is None:
            named = [u for (u, kk) in fc if kk == k]
            spare = max(0.0, 1.0 - sum(fc[(u, k)] for u in named))
            unnamed = sum(1 for u in read_rank if u not in named)
            p = spare / unnamed if unnamed else 0.0
        total += math.log(max(p, 0.0001))
    return total


def folds(rows, min_hist=8, hours=72):
    out = []
    for i in range(min_hist, len(rows)):
        start = rows[i][0]
        win = [r for t, r in rows[i + 1:] if t <= start + hours * 3600]
        if len(win) >= 3:
            out.append((i, win))
    return out


def evaluate(rows, fl, **kw):
    return sum(sum(score(build(rows, i, **kw), r) for r in win) / len(win) for i, win in fl) / len(fl)


if __name__ == '__main__':
    rows = load()
    fl = folds(rows)
    print("folds", len(fl))
    grid = []
    for tau, boost, spare, tl in itertools.product([0.7, 1.0, 1.4, 2.0, 3.0], [1.0, 1.5, 2.5, 4.0], [0.02, 0.05, 0.1], [0.5, 1.0, 2.0]):
        kw = dict(tau=tau, boost=boost, spare=spare, tau_lag=tl)
        grid.append((evaluate(rows, fl, **kw), kw))
    grid.sort(key=lambda x: -x[0])
    for s, kw in grid[:12]:
        print(f"{s:8.3f}  {kw}")
