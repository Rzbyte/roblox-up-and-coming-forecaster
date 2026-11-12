#!/usr/bin/env python3
"""Score a forecast.json against the practice window's real reads (public archive).

Also reproduces the house baseline's published -43.512039 as a check on the scorer.
"""
import json, math, sys
from pathlib import Path

PLACES = 10
READS = Path('/tmp/rbxv3/archive_reads.ndjson')
START = '2026-11-07T23:00:00Z'
END = '2026-11-10T23:00:00Z'


def to_epoch(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def window_reads(start=START, end=END):
    lo, hi = to_epoch(start), to_epoch(end)
    out = []
    for line in READS.read_text().splitlines():
        r = json.loads(line)
        t = to_epoch(r['at'])
        if lo < t <= hi:
            out.append((t, {int(k): v for k, v in r['rank'].items()}, r['at']))
    out.sort(key=lambda x: x[0])
    return out


def score(fc, rank):
    """RULES.txt scoring at one read: sum of ln p over places 1..10."""
    named = [dict() for _ in range(PLACES + 1)]
    for row in fc['probabilities']:
        place = int(row['place'])
        if 1 <= place <= PLACES:
            named[place][int(row['universe_id'])] = float(row['p'])
    order = [u for u, _ in sorted(rank.items(), key=lambda kv: kv[1])]
    total = 0.0
    detail = []
    for place in range(1, PLACES + 1):
        truth = order[place - 1]
        if truth in named[place]:
            p = named[place][truth]
        else:
            spare = max(0.0, 1.0 - sum(named[place].values()))
            unnamed = sum(1 for u in rank if u not in named[place])
            p = spare / unnamed if unnamed else 0.0
        p = max(p, 0.0001)
        total += math.log(p)
        detail.append(p)
    hits = len(set(fc.get('top10') or []) & set(order[:PLACES]))
    agreement = sum(1 for i in range(PLACES) if (fc.get('top10') or [None] * 10)[i] == order[i]) if fc.get('top10') else 0
    return total, detail, hits, agreement


def baseline(rank):
    order = [u for u, _ in sorted(rank.items(), key=lambda kv: kv[1])]
    probs = []
    for place in range(1, PLACES + 1):
        probs.append({'universe_id': order[place - 1], 'place': place, 'p': 0.5})
        if place == 1:
            probs.append({'universe_id': order[1], 'place': 1, 'p': 0.3})
        elif True:
            probs.append({'universe_id': order[place - 2], 'place': place, 'p': 0.15})
            if place < PLACES:
                probs.append({'universe_id': order[place], 'place': place, 'p': 0.15})
    return {'probabilities': probs, 'top10': order[:PLACES]}


def cutoff_rank(start=START):
    lo = to_epoch(start)
    best = None
    for line in READS.read_text().splitlines():
        r = json.loads(line)
        if to_epoch(r['at']) <= lo:
            best = {int(k): v for k, v in r['rank'].items()}
    return best


def report(name, fc_fn, start=START, end=END):
    reads = window_reads(start, end)
    history = cutoff_rank(start)
    fc_fixed = fc_fn(history)
    scores, hits, agrs = [], [], []
    for t, rank, at in reads:
        fc = fc_fixed
        s, _, h, a = score(fc, rank)
        scores.append(s)
        hits.append(h)
        agrs.append(a)
    mean = sum(scores) / len(scores)
    print(f"{name}: log score {mean:.6f}  typical {math.exp(mean / PLACES):.6f}  "
          f"hits {sum(hits)/len(hits):.1f}  agreement {sum(agrs)/len(agrs):.1f}  reads {len(scores)}")
    return mean


if __name__ == '__main__':
    report('house baseline (recompute)', baseline)
    if len(sys.argv) > 1:
        fc = json.loads(Path(sys.argv[1]).read_text())
        report(Path(sys.argv[1]).name, lambda rank: fc)
