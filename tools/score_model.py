#!/usr/bin/env python3
"""Run a model on the real kit input, then score its forecast on the practice window."""
import json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, '/tmp/rbxv3')
import score_dev

KIT = Path.home() / "makefun/rbx/roblox-up-and-coming-forecast-01-kit"


def run_model(model_path, history_lines=None):
    tmp = Path(tempfile.mkdtemp(prefix='dev-'))
    try:
        (tmp / 'input').mkdir()
        (tmp / 'output').mkdir()
        shutil.copy(KIT / 'input/task.json', tmp / 'input/task.json')
        lines = (KIT / 'input/history.ndjson').read_text(encoding='utf-8').splitlines()
        if history_lines:
            lines = lines[:history_lines]
        (tmp / 'input/history.ndjson').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        env = dict(os.environ, MAKEFUN_INPUT_DIR=str(tmp / 'input'), MAKEFUN_OUTPUT_DIR=str(tmp / 'output'))
        env.pop('PYTHONPATH', None)
        proc = subprocess.run([sys.executable, str(model_path)], env=env, cwd=str(tmp),
                              capture_output=True, text=True, timeout=600)
        path = tmp / 'output/forecast.json'
        if proc.returncode != 0 or not path.exists():
            print(f"MODEL FAILED exit {proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}")
            return None
        return json.loads(path.read_text(encoding='utf-8'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_forecast(fc, start=score_dev.START, end=score_dev.END):
    reads = score_dev.window_reads(start, end)
    scores, hits, agrs = [], [], []
    for t, rank, at in reads:
        s, detail, h, a = score_dev.score(fc, rank)
        scores.append(s)
        hits.append(h)
        agrs.append(a)
    return sum(scores) / len(scores), sum(hits) / len(hits), scores


if __name__ == '__main__':
    fc = run_model(Path(sys.argv[1]))
    if fc is None:
        sys.exit(1)
    mean, hits, scores = score_forecast(fc)
    print(f"{Path(sys.argv[1]).name}: practice log score {mean:.6f}  typical {math.exp(mean/10):.6f}  hits {hits:.1f}")
    print("per read:", [round(s, 2) for s in scores])
    print("rows:", len(fc['probabilities']), "top10:", (fc.get('top10') or [])[:3])
