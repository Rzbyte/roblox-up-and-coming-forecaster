#!/usr/bin/env python3
"""V7: V3 plus a CCU-momentum tilt computed from the official input only.

The external study (features.py) measured, on 2720-3000 game-cutoff pairs,
IC(ccu_z, dplace) = -0.235 at 24h, -0.160 at 72h, and IC(tenure, dplace) = +0.16/+0.13:
a game whose CCU is above its own week and still climbing tends to gain places, and older
games on the chart tend to lose them. All of those inputs are computable from the
`playing` values the history already carries, so nothing here needs the network at grading.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/rbxv3/v4')
import harness as H  # noqa: E402

SRC = Path.home() / 'makefun/entry/main.py'
spec = importlib.util.spec_from_file_location('v3mod', SRC)
V = importlib.util.module_from_spec(spec)
sys.modules['v3mod'] = V
spec.loader.exec_module(V)

CONST = dict(SD_BASE=0.17, VOL_MULT=1.08, TAIL_W=0.28, MIN_P=0.002,
             PERSIST_W=0.20, UNNAMED_SCALE=1.0)


def log_slope(points, now, hours):
    pts = [(t, v) for t, v in points if now - hours * 3600 <= t <= now and v > 0]
    if len(pts) < 2:
        return 0.0
    xs = [(t - now) / 3600.0 for t, _ in pts]
    ys = [math.log(v) for _, v in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den > 1e-9 else 0.0


def input_features(reads, cands):
    """CCU features per candidate from the history's own playing values."""
    now = reads[-1].time
    out = {}
    for c in cands:
        pts = []
        first = None
        for snap in reads:
            g = snap.games.get(c.uid)
            if g is not None:
                pts.append((snap.time, max(1.0, g.players)))
                if first is None:
                    first = snap.time
        if len(pts) < 3:
            out[c.uid] = {'z_ccu': 0.0, 'ratio': 0.0, 'decay': 0.0, 'tenure': 0.0, 'slope24': 0.0}
            continue
        week = [(t, v) for t, v in pts if now - t <= 168 * 3600]
        vals = [v for _, v in week]
        mu = statistics.mean(vals)
        sd = statistics.pstdev(vals) or 1.0
        z = (vals[-1] - mu) / sd
        last3 = [v for t, v in pts if now - t <= 3 * 3600]
        prev = [v for t, v in pts if 3 * 3600 < now - t <= 24 * 3600]
        ratio = math.log((statistics.mean(last3) if last3 else vals[-1]) /
                         (statistics.mean(prev) if prev else vals[-1]) or 1.0)
        decay = math.log(vals[-1] / max(vals))
        tenure = min(28.0, (now - first) / 86400) if first else 0.0
        out[c.uid] = {'z_ccu': z, 'ratio': ratio, 'decay': decay,
                      'tenure': tenure, 'slope24': log_slope(pts, now, 24)}
    return out


def zscore(d):
    vals = list(d.values())
    if len(vals) < 3:
        return {k: 0.0 for k in d}
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) or 1.0
    return {k: (v - mu) / sd for k, v in d.items()}


def forecast(reads, weights=None, const=None):
    """V3's forecast with a multiplicative exp(w . z) tilt on each candidate."""
    w = weights or {}
    const = const or CONST
    for k, v in const.items():
        setattr(V, k, v)
    places, hours = 10, 72
    start = reads[-1].time
    candidates = V.candidates_from(reads)
    n = len(candidates)
    feats = input_features(reads, candidates)
    z = {name: zscore({u: feats[u][name] for u in feats}) for name in feats[candidates[0].uid]}
    tilt = [math.exp(sum(w.get(name, 0.0) * z[name][c.uid] for name in z)) for c in candidates]
    for i, c in enumerate(candidates):          # cap so one feature cannot dominate
        tilt[i] = min(6.0, max(1 / 6.0, tilt[i]))

    row_weights = [[0.0] * n for _ in range(places)]
    volatility = V.interval_variability(reads)
    unnamed = V.fit_unnamed(reads, places)
    initial_capacity = sum(math.exp(-c.rank / 35.0) * (1.0 if c.present else 0.05)
                           for c in candidates if c.expiry > start) or 1.0
    for h in range(hours):
        t = h + 0.5
        sd = const['SD_BASE'] + volatility * math.sqrt(t) * const['VOL_MULT']
        trend_time = 24 * (1 - math.exp(-t / 24))
        rank_level = [-math.log(p) for p in range(1, places + 1)]
        eligible_capacity = sum(math.exp(-c.rank / 35.0) * (1.0 if c.present else 0.05)
                                for c in candidates if c.expiry > start + h * 3600)
        coverage = min(1.0, math.sqrt(eligible_capacity / initial_capacity))
        for p, place_score in enumerate(rank_level):
            vals = [0.0] * n
            for i, c in enumerate(candidates):
                if start + h * 3600 >= c.expiry:
                    continue
                survival = math.exp(-t * (0.0018 + 0.000028 * c.rank))
                if not c.present:
                    survival *= 0.08 * math.exp(-c.lag_h / 10)
                centre = c.score + c.slope * 0.76 * trend_time
                surge_prob = min(0.28, c.player_surge * 0.12)
                effect = 1.15 * math.exp(-max(0.0, t - 30) / 27)
                state = ((1 - surge_prob) * V.pdf_tailed(place_score - centre, sd)
                         + surge_prob * V.pdf_tailed(place_score - centre - effect, sd * 1.1))
                if c.persist_scores:
                    occ = sum(wt * V.pdf_tailed(place_score - old, sd * 1.15)
                              for old, wt in c.persist_scores)
                    occ /= sum(wt for _, wt in c.persist_scores)
                    state = (1.0 - const['PERSIST_W']) * state + const['PERSIST_W'] * occ
                vals[i] = survival * state * tilt[i]
            total = sum(vals)
            if total <= 1e-200:
                continue
            for i, value in enumerate(vals):
                row_weights[p][i] += coverage * value / total / hours

    targets = [(1.0 - unnamed[i]) * sum(row) for i, row in enumerate(row_weights)]
    for i, row in enumerate(row_weights):
        row_weights[i] = [(1.0 - unnamed[i]) * v for v in row]
    for _ in range(18):
        sums = [sum(row_weights[i][j] for i in range(places)) for j in range(n)]
        for j, summed in enumerate(sums):
            if summed > 0.995:
                scale = 0.995 / summed
                for i in range(places):
                    row_weights[i][j] *= scale
        for i in range(places):
            s = sum(row_weights[i])
            if s > 0:
                row_weights[i] = [v * targets[i] / s for v in row_weights[i]]
    probabilities = []
    top_scores = {}
    for place, row in enumerate(row_weights, 1):
        for j, pr in enumerate(row):
            if pr < const['MIN_P']:
                continue
            pr = round(pr, 8)
            uid = candidates[j].uid
            probabilities.append({'universe_id': uid, 'place': place, 'p': pr})
            top_scores[uid] = top_scores.get(uid, 0.0) + pr * (places + 1 - place)
    probabilities.sort(key=lambda r: (r['place'], r['universe_id']))
    top10 = sorted(top_scores, key=lambda u: (-top_scores[u], u))[:places]
    if len(top10) < places:
        for c in candidates:
            if c.uid not in top10:
                top10.append(c.uid)
                if len(top10) >= places:
                    break
    return {'probabilities': probabilities, 'top10': top10}


# ---------------------------------------------------------------- evaluation
def main():
    reads = V.parse_input(Path.home() / 'makefun/rbx/roblox-up-and-coming-forecast-01-kit/input')[1]
    plain = [{'t': r.time, 'rank': {u: o.rank for u, o in r.games.items()},
              'play': {u: o.players for u, o in r.games.items()}} for r in reads]
    fl = H.folds(plain)
    arch = H.load_archive()

    def folds(weights, const=None):
        out = []
        for i, win in fl:
            fc = forecast(reads[:i + 1], weights, const)
            out.append(sum(H.score_fc(fc, r['rank']) for r in win) / len(win))
        return out

    base = folds({})
    print(f"V3 folds mean {statistics.mean(base):.3f} med {statistics.median(base):.3f} worst {min(base):.3f}", flush=True)
    rows = {'V3': {'folds': statistics.mean(base)}}
    for name, weights in (('z_ccu', {'z_ccu': 0.25}), ('z_ccu x2', {'z_ccu': 0.5}),
                          ('slope24', {'slope24': 2.0}), ('ratio', {'ratio': 0.15}),
                          ('tenure', {'tenure': 0.35}), ('decay', {'decay': 0.6}),
                          ('ccu+slope', {'z_ccu': 0.2, 'slope24': 1.0}),
                          ('all', {'z_ccu': 0.2, 'slope24': 1.0, 'tenure': 0.3, 'decay': 0.4})):
        sc = folds(weights)
        rows[name] = {'folds': statistics.mean(sc), 'median': statistics.median(sc), 'worst': min(sc)}
        print(f"{name:12s} mean {statistics.mean(sc):7.3f} med {statistics.median(sc):7.2f} "
              f"worst {min(sc):7.2f} d {statistics.mean(sc)-statistics.mean(base):+.3f}", flush=True)

    # holdouts
    ok = [k for k, v in rows.items()]
    def snaps(t):
        out = [{'t': r.time, 'rank': {u: o.rank for u, o in r.games.items()},
                'play': {u: o.players for u, o in r.games.items()}} for r in reads if r.time <= t]
        return out
    A_cut = reads[-1].time
    A_win = [a for a in arch if A_cut < a['t'] <= A_cut + 72 * 3600]
    B_cut = next(a['t'] for a in arch if a['at'].startswith('2026-11-10T23'))
    B_win = [a for a in arch if B_cut < a['t'] <= B_cut + 72 * 3600]

    def vreads(t):
        pl = [d for d in snaps(t)]
        seen = {d['t'] for d in pl}
        for a in arch:
            if a['t'] <= t and a['t'] not in seen:
                pl.append({'t': a['t'], 'rank': dict(a['rank']), 'play': {}})
        pl.sort(key=lambda d: d['t'])
        return [V.Snapshot(d['t'], {u: V.Observation(r, d['play'].get(u, 0.0)) for u, r in d['rank'].items()}, {}) for d in pl]

    print("holdouts", flush=True)
    for name, weights in (('V3', {}), ('ccu+slope', {'z_ccu': 0.2, 'slope24': 1.0}),
                          ('all', {'z_ccu': 0.2, 'slope24': 1.0, 'tenure': 0.3, 'decay': 0.4})):
        for tag, cut, win in (('A', A_cut, A_win), ('B', B_cut, B_win)):
            fc = forecast(vreads(cut), weights)
            sc = sum(H.score_fc(fc, r['rank']) for r in win) / len(win)
            rows.setdefault(name, {})[tag] = round(sc, 3)
            print(f"  {name:10s} holdout {tag} {sc:7.3f}", flush=True)
    json.dump(rows, open('/tmp/rbxv3/v4/v7_eval.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
