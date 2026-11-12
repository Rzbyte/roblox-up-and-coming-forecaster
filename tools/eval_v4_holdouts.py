#!/usr/bin/env python3
"""Holdouts (never used for tuning) + component ablations for the report."""
import json, statistics, sys
sys.path.insert(0, '/tmp/rbxv3/v4')
import harness as H
import model_v6 as M

reads = M.parse_reads()
plain = [{'t': r.time, 'rank': {u: o.rank for u, o in r.games.items()},
          'play': {u: o.players for u, o in r.games.items()}} for r in reads]
fl = H.folds(plain)
arch = H.load_archive()

BEST = {'STAY_MULT': 1.3, 'SD_BASE': 0.16, 'VOL_MULT': 1.2, 'TAIL_W': 0.28,
        'MIN_P': 0.005, 'PERSIST_W': 0.1, 'UNNAMED_SCALE': 0.55}
V3P = {**BEST, 'STAY_MULT': 0.0}


def score_folds(params):
    out = []
    for i, win in fl:
        fc = M.forecast(reads[:i + 1], params)
        out.append(sum(H.score_fc(fc, r['rank']) for r in win) / len(win))
    return out


def stat(sc):
    return round(statistics.mean(sc), 3), round(statistics.median(sc), 3), round(min(sc), 3)


v3_sc = score_folds(V3P)
v6_sc = score_folds(BEST)
print("folds V3 ", stat(v3_sc), flush=True)
print("folds V6 ", stat(v6_sc), flush=True)


def snap_from_plain(d):
    return M.V.Snapshot(d['t'], {u: M.V.Observation(r, d['play'].get(u, 0.0)) for u, r in d['rank'].items()}, {})


def plains_until(t, use_arch=False):
    out = [{'t': r.time, 'rank': {u: o.rank for u, o in r.games.items()},
            'play': {u: o.players for u, o in r.games.items()}} for r in reads if r.time <= t]
    if use_arch:
        seen = {r.time for r in reads}
        for a in arch:
            if a['t'] <= t and a['t'] not in seen:
                out.append({'t': a['t'], 'rank': dict(a['rank']), 'play': {}})
    out.sort(key=lambda x: x['t'])
    return out


A_cut = reads[-1].time
A_win = [a for a in arch if A_cut < a['t'] <= A_cut + 72 * 3600]
B_cut = next(a['t'] for a in arch if a['at'].startswith('2026-11-10T23'))
B_win = [a for a in arch if B_cut < a['t'] <= B_cut + 72 * 3600]
print(f"A {len(A_win)} reads {A_win[0]['at']}..{A_win[-1]['at']}", flush=True)
print(f"B {len(B_win)} reads {B_win[0]['at']}..{B_win[-1]['at']}", flush=True)

GRID = {'V4a2 place-Markov': ('v4a2', {'src_max': 12, 'occ': 0.45, 'unknown': 0.12}),
        'V4b calibrated diffusion': ('v4b', {'a_sd': 0.12, 'b_sd': 0.7, 'spike_scale': 1.0, 'unknown': 0.12})}
CAND = {'V3': ('v3', V3P), 'V6': ('v3', BEST), **GRID}


def run_cand(kind, params, plains):
    if kind == 'v3':
        return M.forecast([snap_from_plain(d) for d in plains], params)
    return getattr(H, kind)(plains, **params)


out = {}
for name, (kind, params) in CAND.items():
    row = {'A': None, 'B': None}
    for tag, cut, win, use_arch in (('A', A_cut, A_win, False), ('B', B_cut, B_win, True)):
        if not win:
            continue
        fc = run_cand(kind, params, plains_until(cut, use_arch))
        row[tag] = round(sum(H.score_fc(fc, w['rank']) for w in win) / len(win), 3)
    out[name] = row
    print(f"{name:26s} A {row['A']:>8} B {row['B']:>8}", flush=True)

# ensemble V6 + place Markov, weights on folds, then holdout check
def ens(plains, w):
    a = M.forecast([snap_from_plain(d) for d in plains], BEST)
    b = H.v4a2(plains, src_max=12, occ=0.45, unknown=0.12)
    return H.blend([a, b], [1 - w, w])


ens_rows = {}
for w in (0.1, 0.2, 0.3):
    sc = [sum(H.score_fc(ens(plains_until(reads[i].time), w), r['rank']) for r in win) / len(win)
          for i, win in fl]
    ens_rows[w] = {'folds': round(statistics.mean(sc), 3)}
    print(f"ensemble w={w} folds {statistics.mean(sc):.3f}", flush=True)
    for tag, cut, win, ua in (('A', A_cut, A_win, False), ('B', B_cut, B_win, True)):
        fc = ens(plains_until(cut, ua), w)
        ens_rows[w][tag] = round(sum(H.score_fc(fc, r['rank']) for r in win) / len(win), 3)
    print(f"   A {ens_rows[w]['A']} B {ens_rows[w]['B']}", flush=True)

abl = {}
for name, params in (('no stay spike', {**BEST, 'STAY_MULT': 0.0}),
                     ('no persistence anchor', {**BEST, 'PERSIST_W': 0.0}),
                     ('no heavy tail (pure Gaussian)', {**BEST, 'TAIL_W': 0.0}),
                     ('no unknown mass', {**BEST, 'UNNAMED_SCALE': 0.0}),
                     ('name every game (MIN_P 1e-4)', {**BEST, 'MIN_P': 0.0001}),
                     ('unknown unscaled (1.0)', {**BEST, 'UNNAMED_SCALE': 1.0})):
    abl[name] = round(statistics.mean(score_folds(params)), 3)
    print(f"ablation {name:32s} {abl[name]:8.3f}  d {abl[name]-statistics.mean(v6_sc):+.3f}", flush=True)

json.dump({'folds': {'V3': v3_sc, 'V6': v6_sc}, 'holdouts': out, 'ensemble': ens_rows,
           'ablations': abl, 'params': BEST}, open('/tmp/rbxv3/v4/holdout_eval.json', 'w'), indent=1)
print("done", flush=True)
