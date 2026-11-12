#!/usr/bin/env python3
"""External-feature study: does CCU/momentum/tenure predict 72h rank moves?

Data: hourly CCU series + 30-day chart ranks from robloxgamestats.com, fetched before the
deadline and cached. Nothing here runs at grading time; only learned constants would ship.
"""
from __future__ import annotations

import bisect
import json
import math
import statistics
import sys
from pathlib import Path

EXT = Path('/tmp/rbxext')
sys.path.insert(0, '/tmp/rbxv3/v4')
import harness as H  # noqa: E402
import model_v6 as M  # noqa: E402


def series(uid):
    p = EXT / f's_{uid}.json'
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return [(h['t'] / 1000.0, h['v']) for h in d.get('hours') or [] if isinstance(h.get('v'), (int, float))]


def ranks(uid):
    p = EXT / f'r_{uid}.json'
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    for s in d.get('sorts') or []:
        if s.get('id') == 'up-and-coming':
            return [(x['t'] / 1000.0, x['rank']) for x in s.get('points') or []]
    return []


def close(points, now, hours):
    lo = now - hours * 3600
    return [(t, v) for t, v in points if lo <= t <= now]


def slope(points):
    if len(points) < 3:
        return None
    xs = [(t - points[-1][0]) / 3600 for t, _ in points]
    ys = [math.log(max(1.0, v)) for _, v in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def features(uid, now):
    s = series(uid)
    if not s or len(s) < 6:
        return None
    out = {}
    last3 = close(s, now, 3)
    prev = [(t, v) for t, v in close(s, now, 24) if t <= now - 3 * 3600]
    if last3 and prev:
        a = statistics.mean(v for _, v in last3)
        b = statistics.mean(v for _, v in prev) or 1.0
        out['ccu_ratio'] = math.log(max(1.0, a) / max(1.0, b))
    else:
        out['ccu_ratio'] = 0.0
    out['ccu_slope6'] = slope(close(s, now, 6)) or 0.0
    out['ccu_slope24'] = slope(close(s, now, 24)) or 0.0
    out['ccu_accel'] = out['ccu_slope6'] - out['ccu_slope24']
    week = close(s, now, 168)
    if week:
        vals = [v for _, v in week]
        peak = max(vals)
        cur = vals[-1]
        out['decay'] = math.log(max(1.0, cur) / max(1.0, peak))
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        out['ccu_z'] = (cur - mu) / sd
    else:
        out['decay'] = 0.0
        out['ccu_z'] = 0.0
    r = ranks(uid) or []
    past = [(t, v) for t, v in r if t <= now]
    out['tenure_d'] = min(28.0, (now - past[0][0]) / 86400) if past else 14.0
    out['rank_best_7d'] = min((v for t, v in past if t >= now - 7 * 86400), default=40)
    out['rank_age'] = out['tenure_d']
    return out


def spearman(xs, ys):
    n = len(xs)
    if n < 6:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    reads = M.parse_reads()
    plain = [{'t': r.time, 'rank': {u: o.rank for u, o in r.games.items()},
              'play': {u: o.players for u, o in r.games.items()}} for r in reads]
    arch = H.load_archive()
    timeline = sorted([(r['t'], r['rank']) for r in plain] + [(a['t'], a['rank']) for a in arch])
    times = [t for t, _ in timeline]
    print(f"reads {len(timeline)} from {timeline[0][0]} to {timeline[-1][0]}", flush=True)

    names = ['ccu_ratio', 'ccu_slope6', 'ccu_slope24', 'ccu_accel', 'decay', 'ccu_z',
             'tenure_d', 'rank_best_7d']
    obs = {h: {n: ([], []) for n in names} for h in (24, 48, 72)}
    used = 0
    for idx, (now, rank) in enumerate(timeline[:-1]):
        # only cutoffs that have a read at least 72h later
        if times[-1] - now < 72 * 3600:
            continue
        rows = []
        for uid, place in rank.items():
            if place > 40:
                continue
            f = features(uid, now)
            if not f:
                continue
            rows.append((uid, place, f))
        if len(rows) < 10:
            continue
        used += 1
        for h in (24, 48, 72):
            j = bisect.bisect_right(times, now + h * 3600) - 1
            if j <= idx or times[j] - now < h * 3600 * 0.75:
                continue
            fut = timeline[j][1]
            for uid, place, f in rows:
                fp = fut.get(uid)
                d = (fp - place) if fp else 12.0        # leaving the top counts as falling
                for n in names:
                    obs[h][n][0].append(f[n])
                    obs[h][n][1].append(d)
    print(f"cutoffs used {used}", flush=True)
    for h in (24, 48, 72):
        line = []
        for n in names:
            ic = spearman(obs[h][n][0], obs[h][n][1])
            line.append(f"{n} {ic:+.3f}" if ic is not None else f"{n} n/a")
        print(f"h={h:2d} n={len(obs[h]['ccu_ratio'][0]):5d}  " + '  '.join(line), flush=True)
    json.dump({str(h): {n: [len(obs[h][n][0]), spearman(obs[h][n][0], obs[h][n][1])] for n in names} for h in obs},
              open('/tmp/rbxv3/v4/feature_ic.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
