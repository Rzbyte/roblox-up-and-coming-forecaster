#!/usr/bin/env python3
"""V4 candidates: Markov occupancy, calibrated diffusion, ensemble.

Protocol
  tune        : 23 walk-forward folds over the kit history (34 reads, to 2026-11-07T20:53Z)
  holdout A   : practice window 2026-11-07T23:00Z .. 2026-11-10T23:00Z (10 real reads, never tuned on)
  holdout B   : 2026-11-10T23:23Z .. 2026-11-12T18:57Z (6 real reads, never tuned on)
Scoring follows RULES.txt exactly (see score_fc).
"""
from __future__ import annotations

import json, math, os, statistics, subprocess, sys, tempfile, shutil
from pathlib import Path

PLACES = 10
KIT = Path.home() / "makefun/rbx/roblox-up-and-coming-forecast-01-kit"
ARCH = Path('/tmp/rbxv3/archive_reads.ndjson')
CACHE = Path('/tmp/rbxv3/v4cache')
V3 = Path.home() / "makefun/entry/main.py"


def ep(stamp: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()


# ---------------------------------------------------------------- data
def load_kit():
    reads = []
    details_seen = 0
    for line in (KIT / 'input/history.ndjson').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        rank, play, born = {}, {}, {}
        for g in r.get('up_and_coming') or []:
            uid = int(g['universe_id'])
            rank[uid] = int(g['rank'])
            if isinstance(g.get('playing'), (int, float)):
                play[uid] = float(g['playing'])
        det = r.get('details')
        if det:
            details_seen += 1
            for k, v in det.items():
                if isinstance(v, dict) and v.get('created'):
                    born[int(k)] = ep(v['created'])
        reads.append({'at': r['at'], 't': ep(r['at']), 'rank': rank, 'play': play, 'born': born})
    reads.sort(key=lambda x: x['t'])
    return reads, details_seen


def load_archive():
    out = []
    for line in ARCH.read_text().splitlines():
        r = json.loads(line)
        out.append({'at': r['at'], 't': ep(r['at']), 'rank': {int(k): v for k, v in r['rank'].items()}, 'play': {}, 'born': {}})
    out.sort(key=lambda x: x['t'])
    return out


# ---------------------------------------------------------------- scoring
def score_fc(fc: dict, rank: dict) -> float:
    named = {}
    for row in fc['probabilities']:
        named.setdefault(int(row['place']), {})[int(row['universe_id'])] = float(row['p'])
    order = [u for u, _ in sorted(rank.items(), key=lambda kv: kv[1])]
    total = 0.0
    for k in range(1, PLACES + 1):
        truth = order[k - 1]
        row = named.get(k, {})
        if truth in row:
            p = row[truth]
        else:
            spare = max(0.0, 1.0 - sum(row.values()))
            unnamed = sum(1 for u in rank if u not in row)
            p = spare / unnamed if unnamed else 0.0
        total += math.log(max(p, 0.0001))
    return total


def score_window(fc: dict, reads, start_t: float, hours: float = 72):
    win = [r for r in reads if start_t < r['t'] <= start_t + hours * 3600]
    if not win:
        return None
    return sum(score_fc(fc, r['rank']) for r in win) / len(win)


# ---------------------------------------------------------------- V3 (subprocess, cached)
def v3_forecast(reads, tag: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    tmp = Path(tempfile.mkdtemp(prefix='v3-', dir='/tmp'))
    try:
        (tmp / 'input').mkdir(); (tmp / 'output').mkdir()
        stamp = lambda t: __import__('datetime').datetime.fromtimestamp(t, __import__('datetime').timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        task = {'places': PLACES, 'output': 'forecast.json', 'history_until': stamp(reads[-1]['t']),
                'window': {'from': stamp(reads[-1]['t']), 'until': stamp(reads[-1]['t'] + 72 * 3600), 'slots': 72, 'slot_minutes': 60},
                'snapshots': len(reads)}
        (tmp / 'input/task.json').write_text(json.dumps(task))
        (tmp / 'input/history.ndjson').write_text('\n'.join(json.dumps({'at': r['at'], 'up_and_coming': [{'rank': v, 'universe_id': u} for u, v in sorted(r['rank'].items(), key=lambda kv: kv[1])]}) for r in reads) + '\n')
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(tmp / 'input'), MAKEFUN_OUTPUT_DIR=str(tmp / 'output'))
        env.pop('PYTHONPATH', None)
        p = subprocess.run([sys.executable, str(V3)], env=env, cwd=str(tmp), capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"v3 failed: {p.stderr[-300:]}")
        fc = json.loads((tmp / 'output/forecast.json').read_text())
        path.write_text(json.dumps(fc))
        return fc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- shared helpers
def kernel_lag_table(bw: float, hours: int = 72, max_lag: float = 120.0, step: float = 0.5):
    """weight(lag) = mean over h=1..hours of exp(-0.5*((lag-h)/bw)^2)."""
    n = int(max_lag / step) + 1
    out = [0.0] * n
    hs = [(h, math.exp(-0.5 * ((h) / bw) ** 2)) for h in range(1, hours + 1)]
    total = float(hours)
    for i in range(n):
        lag = i * step
        s = 0.0
        for h in range(1, hours + 1):
            d = (lag - h) / bw
            if d > 6 or d < -6:
                continue
            s += math.exp(-0.5 * d * d)
        out[i] = s / total
    return out, step


def lag_weight(table, step, lag):
    if lag <= 0:
        return table[0]
    i = int(lag / step)
    if i + 1 >= len(table):
        return table[-1]
    frac = lag / step - i
    return table[i] * (1 - frac) + table[i + 1] * frac


def transitions(reads, cand_max: int, table, step):
    """For every game on the last read: mass of future places, weighted by the horizon mixture.

    Returns {uid: (rank_now, [mass per place 1..10], mass_out)}
    """
    last = reads[-1]['rank']
    cands = {u: r for u, r in last.items() if r <= cand_max}
    out = {u: [0.0] * (PLACES + 1) for u in cands}   # index 0 = left the places
    for i, a in enumerate(reads[:-1]):
        for b in reads[i + 1:]:
            lag = (b['t'] - a['t']) / 3600
            w = lag_weight(table, step, lag)
            if w < 1e-6:
                continue
            for u in cands:
                src = a['rank'].get(u)
                if src is None:
                    continue
                dst = b['rank'].get(u)
                idx = dst if (dst is not None and dst <= PLACES) else 0
                out[u][idx] += w
    return {u: (cands[u], v) for u, v in out.items()}


# ---------------------------------------------------------------- V4a Markov + occupancy
def v4a(reads, cand_max=25, bw=5.0, occ=0.25, tau_occ=2.0, unknown=0.10, min_p=0.0002, sharp=1.0):
    """Markov rank-transition with an occupancy prior, horizon-mixture weighted."""
    table, step = kernel_lag_table(bw)
    obs = transitions(reads, cand_max, table, step)
    conf, dist = {}, {}
    for u, (r_now, mass) in obs.items():
        places = mass[1:]
        tot = sum(places)
        if tot <= 0:
            continue
        prior = [math.exp(-abs(r_now - k) / tau_occ) for k in range(1, PLACES + 1)]
        ps = sum(prior)
        dist[u] = [(1 - occ) * (places[k - 1] / tot) + occ * (prior[k - 1] / ps) for k in range(1, PLACES + 1)]
        conf[u] = tot
    if not conf:
        return {'probabilities': [], 'top10': []}
    cmax = max(conf.values())
    fc = {}
    for k in range(1, PLACES + 1):
        w = {u: conf[u] * dist[u][k - 1] for u in dist}
        s = sum(w.values())
        if s <= 0:
            continue
        if sharp != 1.0:
            w = {u: (v / s) ** sharp for u, v in w.items()}
            s = sum(w.values())
        for u, v in w.items():
            p = (1 - unknown) * v / s
            if p >= min_p:
                fc[(u, k)] = p
    rows = [{'universe_id': u, 'place': k, 'p': round(p, 8)} for (u, k), p in fc.items()]
    rows.sort(key=lambda x: (x['place'], x['universe_id']))
    top = []
    for k in range(1, PLACES + 1):
        best = max(((p, u) for (u, kk), p in fc.items() if kk == k), default=None)
        if best and best[1] not in top:
            top.append(best[1])
    return {'probabilities': rows, 'top10': top[:PLACES]}


# ---------------------------------------------------------------- V4b calibrated diffusion
def stay_curve(reads, places=PLACES):
    """Measured probability that a top-10 game is still at its place, by lag bucket."""
    buckets = {}
    for i, a in enumerate(reads[:-1]):
        for b in reads[i + 1:]:
            lag = round((b['t'] - a['t']) / 3600)
            if lag > 96:
                continue
            key = max(1, lag)
            tot, same = buckets.get(key, (0, 0))
            for u, r in a['rank'].items():
                if r <= places:
                    tot += 1
                    if b['rank'].get(u) == r:
                        same += 1
            buckets[key] = (tot, same)
    return {k: (same / tot if tot else 0.5) for k, (tot, same) in buckets.items()}


def stay_at(curve, h, bw=6.0):
    num = den = 0.0
    for lag, p in curve.items():
        w = math.exp(-0.5 * ((lag - h) / bw) ** 2)
        num += w * p
        den += w
    return num / den if den else 0.4


def v4b(reads, cand_max=30, a_sd=0.15, b_sd=0.9, spike_scale=1.0, tail=0.28, unknown=0.10,
        min_p=0.0002, slope_w=0.5, calibrate=True):
    last = reads[-1]
    curve = stay_curve(reads)
    cands = {u: r for u, r in last['rank'].items() if r <= cand_max}
    # rank slope over the last few reads (log-rank per hour), limited
    hist = reads[-4:]
    slope = {}
    for u in cands:
        pts = [(r['t'], math.log(r['rank'][u])) for r in hist if u in r['rank'] and r['rank'][u] <= 60]
        if len(pts) >= 2:
            t0 = pts[-1][0]
            num = sum((t - t0) / 3600 * (y - pts[-1][1]) for t, y in pts)
            den = sum(((t - t0) / 3600) ** 2 for t, y in pts)
            slope[u] = (num / den) if den else 0.0
        else:
            slope[u] = 0.0
    rank_level = [-math.log(k) for k in range(1, PLACES + 1)]
    w = {k: {} for k in range(1, PLACES + 1)}
    for u, r in cands.items():
        centre0 = -math.log(max(1, r))
        for h in range(1, 73):
            t = h + 0.5
            trend = 24 * (1 - math.exp(-t / 24))
            centre = centre0 + slope[u] * slope_w * trend
            sd = a_sd + b_sd * math.sqrt(t) if not calibrate else a_sd + b_sd * math.sqrt(t)
            hill = stay_at(curve, h) * spike_scale if calibrate else 0.0
            hill = min(0.85, max(0.0, hill))
            for k in range(1, PLACES + 1):
                d = rank_level[k - 1] - centre
                z = d / max(0.08, sd)
                pdf = (1 - tail) * math.exp(-0.5 * z * z) + tail * (1 + z * z / 4) ** -2.5
                val = (1 - hill) * pdf
                if k == r:
                    val += hill
                w[k][u] = w[k].get(u, 0.0) + val / 72.0
    rows, fc = [], {}
    for k in range(1, PLACES + 1):
        s = sum(w[k].values())
        if s <= 0:
            continue
        keep = 1 - unknown
        for u, v in w[k].items():
            p = keep * v / s
            if p >= min_p:
                fc[(u, k)] = p
                rows.append({'universe_id': u, 'place': k, 'p': round(p, 8)})
    rows.sort(key=lambda x: (x['place'], x['universe_id']))
    top = []
    for k in range(1, PLACES + 1):
        best = max(((p, u) for (u, kk), p in fc.items() if kk == k), default=None)
        if best and best[1] not in top:
            top.append(best[1])
    return {'probabilities': rows, 'top10': top[:PLACES]}


# ---------------------------------------------------------------- ensemble
def blend(fcs, weights):
    per_place = {}
    for fc, wgt in zip(fcs, weights):
        if wgt <= 0:
            continue
        for row in fc['probabilities']:
            k, u = int(row['place']), int(row['universe_id'])
            per_place.setdefault(k, {})
            per_place[k][u] = per_place[k].get(u, 0.0) + wgt * float(row['p'])
    rows, flat = [], {}
    for k, games in per_place.items():
        s = sum(games.values())
        if s <= 0:
            continue
        for u, v in games.items():
            p = v / s
            if p >= 0.0002:
                flat[(u, k)] = p
                rows.append({'universe_id': u, 'place': k, 'p': round(p, 8)})
    rows.sort(key=lambda x: (x['place'], x['universe_id']))
    top = []
    for k in range(1, PLACES + 1):
        best = max(((p, u) for (u, kk), p in flat.items() if kk == k), default=None)
        if best and best[1] not in top:
            top.append(best[1])
    return {'probabilities': rows, 'top10': top[:PLACES]}


# ---------------------------------------------------------------- fold machinery
def folds(reads, min_hist=8, hours=72):
    out = []
    for i in range(min_hist, len(reads)):
        start = reads[i]['t']
        win = [r for r in reads[i + 1:] if r['t'] <= start + hours * 3600]
        if len(win) >= 3:
            out.append((i, win))
    return out


def eval_folds(model, reads, fl, tag_base=''):
    scores = []
    for i, win in fl:
        hist = reads[:i + 1]
        fc = model(hist)
        scores.append(sum(score_fc(fc, r['rank']) for r in win) / len(win))
    return scores


# ---------------------------------------------------------------- V4a2 empirical place Markov
def v4a2(reads, src_max=14, bw=5.0, smooth=0.7, occ=0.2, unknown=0.10, min_p=0.0002, sharp=1.0):
    """Markov chain on the place itself: P(dst place | src place), horizon-averaged.

    Games are then weighted by an occupancy prior (how often that seat is won from this src)
    and by how continuously the game has been on the chart.
    """
    table, step = kernel_lag_table(bw)
    last = reads[-1]['rank']
    cands = {u: r for u, r in last.items() if r <= src_max}
    counts = {s: [0.0] * (PLACES + 1) for s in range(1, PLACES + 1)}   # index 0 = out
    conf = {u: 0.0 for u in cands}
    for i, a in enumerate(reads[:-1]):
        for b in reads[i + 1:]:
            lag = (b['t'] - a['t']) / 3600
            w = lag_weight(table, step, lag)
            if w < 1e-6:
                continue
            for s in range(1, PLACES + 1):
                # any game that was at place s in read a, where did place s go in b?
                who = [u for u, r in a['rank'].items() if r == s]
                if not who:
                    continue
                u = who[0]
                dst = b['rank'].get(u)
                counts[s][dst if (dst is not None and dst <= PLACES) else 0] += w
            for u in cands:
                if u in a['rank']:
                    conf[u] += w
    share = {}
    for u, r in cands.items():
        row = counts.get(r)
        if row is None:
            row = [0.0] * (PLACES + 1)
        tot = sum(row[1:])
        if tot > 0:
            base = [row[k] / tot for k in range(1, PLACES + 1)]
        else:
            base = [1.0 / PLACES] * PLACES
        prior = [math.exp(-abs(r - k) / 1.5) for k in range(1, PLACES + 1)]
        ps = sum(prior)
        share[u] = [(1 - occ) * base[k] + occ * prior[k] / ps for k in range(PLACES)]
    if not share:
        return {'probabilities': [], 'top10': []}
    cmax = max(conf.values()) or 1.0
    fc = {}
    for k in range(1, PLACES + 1):
        w = {u: (0.25 + 0.75 * conf[u] / cmax) * share[u][k - 1] for u in share}
        s = sum(w.values())
        if s <= 0:
            continue
        if sharp != 1.0:
            w = {u: (v / s) ** sharp for u, v in w.items()}
            s = sum(w.values())
        for u, v in w.items():
            p = (1 - unknown) * v / s
            if p >= min_p:
                fc[(u, k)] = p
    rows = [{'universe_id': u, 'place': k, 'p': round(p, 8)} for (u, k), p in fc.items()]
    rows.sort(key=lambda x: (x['place'], x['universe_id']))
    top = []
    for k in range(1, PLACES + 1):
        best = max(((p, u) for (u, kk), p in fc.items() if kk == k), default=None)
        if best and best[1] not in top:
            top.append(best[1])
    return {'probabilities': rows, 'top10': top[:PLACES]}
