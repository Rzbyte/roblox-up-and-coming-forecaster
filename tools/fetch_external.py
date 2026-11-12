#!/usr/bin/env python3
"""Fetch hourly CCU series and chart-rank history for every game in the kit/archive reads."""
import json, os, time, urllib.request
from pathlib import Path

OUT = Path('/tmp/rbxext')
OUT.mkdir(exist_ok=True)
KIT = Path.home() / 'makefun/rbx/roblox-up-and-coming-forecast-01-kit/input/history.ndjson'
ARCH = Path('/tmp/rbxv3/archive_reads.ndjson')

ids = set()
for line in KIT.read_text().splitlines():
    for g in json.loads(line).get('up_and_coming') or []:
        ids.add(int(g['universe_id']))
for line in ARCH.read_text().splitlines():
    for u in json.loads(line)['rank']:
        ids.add(int(u))
ids = sorted(ids)
(OUT / 'ids.json').write_text(json.dumps(ids))
print('games', len(ids), flush=True)

UA = 'Mozilla/5.0 (compatible; research-script/1.0)'


def get(url, dest: Path, tries=3):
    if dest.exists() and dest.stat().st_size > 20:
        return True
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=25) as r:
                body = r.read()
            dest.write_bytes(body)
            return True
        except Exception as e:
            if k == tries - 1:
                print('fail', url, e, flush=True)
                return False
            time.sleep(3)


n = 0
for i, uid in enumerate(ids):
    ok1 = get(f'https://robloxgamestats.com/api/v1/games/{uid}/series?days=14', OUT / f's_{uid}.json')
    time.sleep(1.15)
    ok2 = get(f'https://robloxgamestats.com/api/v1/games/{uid}/ranks?days=30', OUT / f'r_{uid}.json')
    time.sleep(1.15)
    n += 1
    if n % 20 == 0:
        print(f'{n}/{len(ids)}', flush=True)
print('done', n, flush=True)
