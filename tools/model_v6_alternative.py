#!/usr/bin/env python3
"""V6 = entry V3 with explicable knobs, plus a measured same-place persistence spike.

Tuning protocol: 23 walk-forward folds over the kit history only. Holdouts are never used.
"""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '/tmp/rbxv3/v4')
import harness as H

SRC = Path.home() / 'makefun/entry/main.py'
spec = importlib.util.spec_from_file_location('v6', SRC)
V = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V
spec.loader.exec_module(V)


# ------------------------------------------------------------ measured stay curve
def stay_curve(reads, places=10, max_lag=96):
    buckets = {}
    for i, a in enumerate(reads[:-1]):
        for b in reads[i + 1:]:
            lag = int(round((b.time - a.time) / 3600))
            if lag < 1 or lag > max_lag:
                continue
            tot, same = buckets.get(lag, (0, 0))
            for uid, obs in a.games.items():
                if obs.rank <= places:
                    tot += 1
                    if uid in b.games and b.games[uid].rank == obs.rank:
                        same += 1
            buckets[lag] = (tot, same)
    return {k: (s / t if t else 0.35) for k, (t, s) in buckets.items()}


def stay_at(curve, h, bw=6.0):
    num = den = 0.0
    for lag, p in curve.items():
        w = math.exp(-0.5 * ((lag - h) / bw) ** 2)
        num += w * p
        den += w
    return num / den if den else 0.35


PARAMS = dict(SD_BASE=0.17, VOL_MULT=1.08, TAIL_W=0.28, MIN_P=0.002,
              PERSIST_W=0.20, UNNAMED_SCALE=1.0, STAY_MULT=0.0, STAY_BW=6.0)


def set_params(**kw):
    for k, v in kw.items():
        if k not in PARAMS:
            raise KeyError(k)
        PARAMS[k] = v
        if hasattr(V, k):
            setattr(V, k, v)


def forecast(reads, params=None, stay_on=True):
    p = dict(PARAMS)
    if params:
        for k, v in params.items():
            if k not in p:
                raise KeyError(k)
            p[k] = v
    for k in ('SD_BASE', 'VOL_MULT', 'TAIL_W', 'MIN_P', 'PERSIST_W', 'UNNAMED_SCALE'):
        setattr(V, k, p[k])
    task = {'places': 10, 'window': {'slots': 72}}
    # patch inner loop only when the spike is on
    if stay_on and p['STAY_MULT'] > 0:
        return _spike_forecast(reads, p)
    return V.forecast(task, reads)


def _spike_forecast(reads, p):
    """Copy of V.forecast with a same-place spike mixed into each hour's state."""
    places, hours = 10, 72
    start = reads[-1].time
    offset = 0.0
    candidates = V.candidates_from(reads)
    n = len(candidates)
    row_weights = [[0.0] * n for _ in range(places)]
    volatility = V.interval_variability(reads)
    unnamed = V.fit_unnamed(reads, places)
    curve = stay_curve(reads)
    initial_capacity = sum(math.exp(-c.rank / 35.0) * (1.0 if c.present else 0.05)
                           for c in candidates if c.expiry > start) or 1.0
    for h in range(hours):
        t = offset + h + 0.5
        sd = p['SD_BASE'] + volatility * math.sqrt(t) * p['VOL_MULT']
        trend_time = 24 * (1 - math.exp(-t / 24))
        rank_level = [-math.log(x) for x in range(1, places + 1)]
        eligible_capacity = sum(math.exp(-c.rank / 35.0) * (1.0 if c.present else 0.05)
                                for c in candidates if c.expiry > start + h * 3600)
        coverage = min(1.0, math.sqrt(eligible_capacity / initial_capacity))
        spike_w = min(0.75, max(0.0, p['STAY_MULT'] * stay_at(curve, t, p['STAY_BW'])))
        for pi, place_score in enumerate(rank_level):
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
                    occ = sum(w * V.pdf_tailed(place_score - old, sd * 1.15)
                              for old, w in c.persist_scores)
                    occ /= sum(w for _, w in c.persist_scores)
                    state = (1.0 - p['PERSIST_W']) * state + p['PERSIST_W'] * occ
                if spike_w > 0 and c.present and pi + 1 == int(round(c.rank)):
                    state = (1 - spike_w) * state + spike_w * 1.0
                vals[i] = survival * state
            total = sum(vals)
            if total <= 1e-200:
                continue
            for i, value in enumerate(vals):
                row_weights[pi][i] += coverage * value / total / hours

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
            if pr < p['MIN_P']:
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


# ------------------------------------------------------------ evaluation
def parse_reads():
    task, reads = V.parse_input(H.KIT / 'input')
    return reads


def run(reads, fl, model):
    out = []
    for i, win in fl:
        fc = model(reads[:i + 1])
        out.append(sum(H.score_fc(fc, r['rank']) for r in win) / len(win))
    return out


def report(name, sc, base):
    m = statistics.mean(sc)
    print(f"{name:44s} mean {m:7.2f} med {statistics.median(sc):7.2f} worst {min(sc):7.2f} d {m-base:+6.2f}", flush=True)
    return m
