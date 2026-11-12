#!/usr/bin/env python3
"""Real-data walk-forward evaluation, feeding each model its own kit-style input.

Scores exactly as RULES.txt's Scoring section: per read, per place 1..10,
p = your row for the game actually there, else the place's unnamed share over
the count of games on that read not named there; ln floored at 0.0001; a read's
score sums 10 places; the log score is the mean over the window's reads.
"""
import json, math, os, sys, tempfile, shutil, subprocess
from pathlib import Path
from datetime import datetime, timezone

KIT = Path.home() / "makefun/rbx/roblox-up-and-coming-forecast-01-kit"
PLACES = 10
HOURS = 72


def epoch(stamp):
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()


def load_raw(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        if line.strip():
            row = json.loads(line)
            games = {int(g['universe_id']): int(g['rank']) for g in row.get('up_and_coming') or []}
            if games:
                rows.append({'raw': line, 'at': row['at'], 'time': epoch(row['at']),
                             'order': [u for u, _ in sorted(games.items(), key=lambda kv: kv[1])],
                             'games': games})
    rows.sort(key=lambda r: r['time'])
    return rows


def score(forecast, read):
    named = [dict() for _ in range(PLACES + 1)]
    for row in forecast['probabilities']:
        p, uid = int(row['place']), int(row['universe_id'])
        if 1 <= p <= PLACES:
            named[p][uid] = float(row['p'])
    total = 0.0
    for place in range(1, PLACES + 1):
        truth = read['order'][place - 1]
        if truth in named[place]:
            p = named[place][truth]
        else:
            spare = max(0.0, 1.0 - sum(named[place].values()))
            unnamed = sum(1 for u in read['games'] if u not in named[place])
            p = spare / unnamed if unnamed else 0.0
        total += math.log(max(p, 0.0001))
    return total


def window_score(forecast, rows, start_time, hours=HOURS):
    window = [r for r in rows if start_time < r['time'] <= start_time + hours * 3600]
    if not window:
        return None
    return sum(score(forecast, r) for r in window) / len(window), len(window)


def baseline_forecast(order, places=PLACES):
    probs = []
    chart = order
    for place in range(1, min(places, len(chart)) + 1):
        probs.append({'universe_id': chart[place - 1], 'place': place, 'p': 0.5})
        if place == 1:
            if len(chart) > 1:
                probs.append({'universe_id': chart[1], 'place': 1, 'p': 0.3})
            continue
        probs.append({'universe_id': chart[place - 2], 'place': place, 'p': 0.15})
        if len(chart) > place:
            probs.append({'universe_id': chart[place], 'place': place, 'p': 0.15})
    return {'probabilities': probs, 'top10': chart[:places]}


def load_model(path):
    spec = importlib.util.spec_from_file_location('model_under_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_forecast(mod_path, history_rows, start_time):
    """Run the model exactly as the grader does: a subprocess over a kit-style input dir."""
    tmp = Path(tempfile.mkdtemp(prefix='fold-'))
    try:
        (tmp / 'input').mkdir()
        (tmp / 'output').mkdir()
        stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        task = {'history_until': stamp(start_time),
                'window': {'from': stamp(start_time), 'until': stamp(start_time + HOURS * 3600),
                           'slot_minutes': 60, 'slots': HOURS},
                'places': PLACES, 'output': 'forecast.json', 'snapshots': len(history_rows)}
        (tmp / 'input/task.json').write_text(json.dumps(task), encoding='utf-8')
        (tmp / 'input/history.ndjson').write_text(
            ''.join(r['raw'] if r['raw'].endswith('\n') else r['raw'] + '\n' for r in history_rows),
            encoding='utf-8')
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(tmp / 'input'), MAKEFUN_OUTPUT_DIR=str(tmp / 'output'))
        env.pop('PYTHONPATH', None)
        proc = subprocess.run([sys.executable, str(mod_path)], env=env, cwd=str(tmp),
                              capture_output=True, text=True, timeout=300)
        path = tmp / 'output/forecast.json'
        if proc.returncode != 0 or not path.exists():
            raise RuntimeError(f"model exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        return json.loads(path.read_text(encoding='utf-8')), path.read_text(encoding='utf-8')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(folds_min_history=8, hours=HOURS):
    rows = load_raw(KIT / 'input/history.ndjson')
    print(f"reads={len(rows)}  {rows[0]['at']} -> {rows[-1]['at']}")
    model = Path(os.getenv('MODEL', '/tmp/rbxv3/roblox_forecaster_v3/main.py'))
    results = []
    for i in range(folds_min_history, len(rows)):
        hist = rows[:i + 1]
        start = hist[-1]['time']
        if len([r for r in rows[i + 1:] if r['time'] <= start + hours * 3600]) < 3:
            continue
        try:
            fc, _raw = model_forecast(model, hist, start)
        except Exception as exc:
            print(f"fold {i:2d} {hist[-1]['at']} ERROR {type(exc).__name__}: {exc}")
            continue
        got, nreads = window_score(fc, rows, start, hours)
        base, _ = window_score(baseline_forecast(hist[-1]['order']), rows, start, hours)
        results.append({'i': i, 'at': hist[-1]['at'], 'model': got, 'reads': nreads, 'baseline': base})
        print(f"fold {i:2d} {hist[-1]['at']}  model {got:8.3f} (reads {nreads})  baseline {base:8.3f}  delta {got-base:+7.3f}")
    if results:
        m = sum(r['model'] for r in results) / len(results)
        b = sum(r['baseline'] for r in results) / len(results)
        srt = sorted(r['model'] for r in results)
        print(f"\nMEAN over {len(results)} folds: model {m:.3f}  baseline {b:.3f}  delta {m-b:+.3f}")
        print(f"median model {srt[len(srt)//2]:.3f}  best {max(srt):.3f}  worst {min(srt):.3f}")
    return results


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
