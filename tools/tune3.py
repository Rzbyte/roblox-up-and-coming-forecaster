#!/usr/bin/env python3
"""Tune v5's constants against the walk-forward folds (never against the practice window)."""
import importlib.util, itertools, json, sys, types
from pathlib import Path
sys.path.insert(0, '/tmp/rbxv3')
import score_dev

KIT = Path.home() / "makefun/rbx/roblox-up-and-coming-forecast-01-kit"


def load_v5():
    spec = importlib.util.spec_from_file_location('m5', '/tmp/rbxv3/model_v5.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['m5'] = mod
    spec.loader.exec_module(mod)
    return mod


def hist_lines():
    return (KIT / 'input/history.ndjson').read_text(encoding='utf-8').splitlines()


def parse_prefix(mod, lines, i):
    """Feed the model the first i+1 history lines through its own parser."""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix='pfx-', dir='/tmp'))
    try:
        (tmp / 'input').mkdir()
        (tmp / 'input' / 'history.ndjson').write_text('\n'.join(lines[:i + 1]) + '\n', encoding='utf-8')
        task = json.loads((KIT / 'input/task.json').read_text())
        task['history_until'] = json.loads(lines[i])['at']
        (tmp / 'input/task.json').write_text(json.dumps(task))
        return mod.parse_input(tmp / 'input')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pairs(mod, lines, hours=72, min_hist=8):
    reads = []
    for line in lines:
        r = json.loads(line)
        reads.append((score_dev.to_epoch(r['at']), {int(g['universe_id']): int(g['rank']) for g in r['up_and_coming']}))
    out = []
    for i in range(min_hist, len(reads)):
        start = reads[i][0]
        win = [r for t, r in reads[i + 1:] if t <= start + hours * 3600]
        if len(win) >= 3:
            out.append({'i': i, 'win': win, 'at': json.loads(lines[i])['at']})
    return out


def score_fc(fc, rank):
    named = {}
    for row in fc['probabilities']:
        named.setdefault(int(row['place']), {})[int(row['universe_id'])] = float(row['p'])
    order = [u for u, _ in sorted(rank.items(), key=lambda kv: kv[1])]
    total = 0.0
    for k in range(1, 11):
        truth = order[k - 1]
        row = named.get(k, {})
        if truth in row:
            p = row[truth]
        else:
            spare = max(0.0, 1.0 - sum(row.values()))
            unnamed = sum(1 for u in rank if u not in row)
            p = spare / unnamed if unnamed else 0.0
        total += __import__('math').log(max(p, 0.0001))
    return total


def prepare(mod, lines, folds):
    out = []
    for f in folds:
        task, snap = parse_prefix(mod, lines, f['i'])
        task = dict(task)
        task['window'] = {'from': f['at'], 'slots': 72}
        out.append((task, snap, f['win']))
    return out


def fold_score(mod, prepared):
    total = 0.0
    for task, snap, win in prepared:
        fc = mod.forecast(task, snap)
        total += sum(score_fc(fc, r) for r in win) / len(win)
    return total / len(prepared)


if __name__ == '__main__':
    lines = hist_lines()
    mod = load_v5()
    folds = pairs(mod, lines)
    prepared = prepare(mod, lines, folds)
    print("folds", len(prepared), flush=True)
    print("base", round(fold_score(mod, prepared), 3), flush=True)
    grid = []
    for sd, vol, tail, minp, pers in itertools.product([0.17, 0.28], [0.9, 1.08, 1.3], [0.28, 0.45], [0.001, 0.003], [0.2, 0.35]):
        mod.SD_BASE, mod.VOL_MULT, mod.TAIL_W, mod.MIN_P, mod.PERSIST_W = sd, vol, tail, minp, pers
        s = fold_score(mod, prepared)
        grid.append((s, dict(SD_BASE=sd, VOL_MULT=vol, TAIL_W=tail, MIN_P=minp, PERSIST_W=pers)))
        print(f"{s:8.3f} {grid[-1][1]}", flush=True)
    grid.sort(key=lambda x: -x[0])
    print("\nBEST:", json.dumps(grid[:5], indent=1))
