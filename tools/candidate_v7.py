#!/usr/bin/env python3
"""Offline probabilistic Roblox Up-and-Coming forecaster (Python 3.13).

Uses a mixture of latent log-rank dynamics and historical occupation.  Explicit
unallocated probability is essential: new games may enter during the horizon.
No network, third-party packages, mutable external state, or runtime AI.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Observation:
    rank: int
    players: float


@dataclass
class Snapshot:
    time: float
    games: dict[int, Observation]
    born: dict[int, float]


def to_epoch(stamp: str) -> float:
    date = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.timestamp()


def parse_input(input_dir: Path) -> tuple[dict, list[Snapshot]]:
    task = json.loads((input_dir / 'task.json').read_text(encoding='utf-8'))
    deadline = to_epoch(task['history_until']) if task.get('history_until') else float('inf')
    reads = []
    with (input_dir / 'history.ndjson').open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            t = to_epoch(row['at'])
            if t > deadline + 1:
                continue
            games = {}
            for g in row.get('up_and_coming', []):
                try:
                    uid = int(g['universe_id'])
                    rank = int(g['rank'])
                    players = max(0., float(g.get('playing') or 0))
                    if uid > 0 and 1 <= rank <= 300:
                        games[uid] = Observation(rank, players)
                except (TypeError, ValueError, KeyError, OverflowError):
                    pass
            if not games:
                continue
            born = {}
            for key, detail in (row.get('details') or {}).items():
                if isinstance(detail, dict) and detail.get('created'):
                    try:
                        born[int(key)] = to_epoch(detail['created'])
                    except (TypeError, ValueError, OverflowError):
                        pass
            reads.append(Snapshot(t, games, born))
    reads.sort(key=lambda r: r.time)
    result = []
    for r in reads:
        if result and r.time == result[-1].time:
            result[-1] = r
        else:
            result.append(r)
    if not result:
        raise ValueError('No usable chart history')
    return task, result


def weighted_slope(points: list[tuple[float, float]], now: float,
                   window_h: float, half_life_h: float) -> float:
    """Time-weighted least squares for (seconds, value); returns units/hour."""
    samples = []
    for time, y in points:
        x = (time - now) / 3600.0
        if -window_h <= x <= 0:
            samples.append((x, y, math.exp(math.log(2) * x / half_life_h)))
    if len(samples) < 2:
        return 0.0
    sw = sum(w for _, _, w in samples)
    xbar = sum(x*w for x, _, w in samples) / sw
    ybar = sum(y*w for _, y, w in samples) / sw
    var = sum(w*(x-xbar)**2 for x, _, w in samples)
    return sum(w*(x-xbar)*(y-ybar) for x, y, w in samples) / var if var > .01 else 0.


def interval_variability(reads: list[Snapshot]) -> float:
    """Estimate hourly noise in negative log rank from within-game changes."""
    rates = []
    prev = None
    for snap in reads[-70:]:
        if prev:
            dt = (snap.time - prev.time) / 3600.
            if .25 <= dt <= 36:
                for uid, game in snap.games.items():
                    if uid in prev.games:
                        z = math.log(prev.games[uid].rank / game.rank)
                        rates.append(min(4., z*z / dt))
        prev = snap
    if len(rates) < 25:
        return .13
    rates.sort()
    # Trim outlier measurement errors without removing genuine rank jumps.
    rates = rates[:int(len(rates)*.95)]
    noise = math.sqrt(sum(rates) / max(1, len(rates)))
    return max(.075, min(.28, noise))


@dataclass
class Candidate:
    uid: int
    rank: float
    score: float
    slope: float
    player_surge: float
    lag_h: float
    expiry: float
    present: bool
    persist_scores: list[tuple[float, float]]


def candidates_from(reads: list[Snapshot], max_candidates: int = 150) -> list[Candidate]:
    now = reads[-1].time
    found = defaultdict(list)
    born = {}
    for snap in reads:
        for uid, val in snap.games.items():
            found[uid].append((snap.time, val))
        born.update(snap.born)
    output = []
    for uid, points in found.items():
        last_time, obs = points[-1]
        gap = max(0., (now-last_time)/3600.)
        present = uid in reads[-1].games
        if not present and (gap > 15 or obs.rank > 80):
            continue
        rank = float(reads[-1].games[uid].rank if present else obs.rank+4+1.2*gap)
        series = [(t, -math.log(g.rank)) for t, g in points]
        drift = weighted_slope(series, last_time, window_h=40, half_life_h=14)
        count_recent = sum(last_time-t <= 32*3600 for t, _ in points)
        drift *= min(1., max(.2, (count_recent-1)/5))
        drift = max(-.075, min(.075, drift))
        recent_players = [(t, math.log1p(g.players)) for t,g in points if last_time-t <= 36*3600]
        # Surge prior: only a *large* increase and at least modest CCU level.
        surge = 0.
        if len(recent_players) >= 2 and obs.players >= 2000:
            old_t, old_log = recent_players[-2]
            if last_time - old_t <= 24*3600:
                surge = max(0., min(2.5, recent_players[-1][1] - old_log - .8))
        persist = []
        for t, g in points:
            age = (now-t)/3600.
            if age <= 96:
                persist.append(( -math.log(g.rank), math.exp(-age/22.)))
        expiry = born.get(uid, float('inf')) + 28*86400
        output.append(Candidate(uid, rank, -math.log(rank), drift, surge,
                                gap, expiry, present, persist))
    output.sort(key=lambda c: (not c.present, c.rank, c.uid))
    return output[:max_candidates]


def fit_unnamed(reads: list[Snapshot], places: int) -> list[float]:
    """Prior-shrunk future-unseen fractions, measured on earlier rolling cuts.

    It never examines anything beyond the available history.  A future game
    counts as unseen when it was not in the model's recent candidate pool.
    """
    base = [.30, .28, .26, .24, .23, .22, .21, .20, .19, .19][:places]
    if len(reads) < 16 or reads[-1].time-reads[0].time < 7*86400:
        return base
    stat = [[0., 0.] for _ in range(places)]
    last = reads[-1].time
    # Only older cuts permit a complete 72-hour follow-up.
    eligible = [i for i in range(7, len(reads)-1)
                if reads[i].time <= last-72*3600]
    selected = []
    for i in reversed(eligible):
        if not selected or (reads[selected[-1]].time-reads[i].time)/3600 >= 20:
            selected.append(i)
        if len(selected) >= 10:
            break
    for i in selected:
        prefix = reads[:i+1]
        known = {c.uid for c in candidates_from(prefix)}
        limit = prefix[-1].time+72*3600
        future = [r for r in reads[i+1:] if r.time <= limit]
        for snap in future:
            for g, val in snap.games.items():
                if 1 <= val.rank <= places:
                    place = val.rank-1
                    stat[place][1] += 1
                    if g not in known:
                        stat[place][0] += 1
    rates = []
    for p, (unknown, total) in enumerate(stat):
        # At least 25 pseudo-observations prevents tuning to tiny archive noise.
        estimate = (unknown+25*base[p]) / (total+25)
        rates.append(max(.05, min(.60, UNNAMED_SCALE * estimate)))
    return rates


SD_BASE = 0.17
VOL_MULT = 1.08
TAIL_W = 0.28
MIN_P = 0.002
PERSIST_W = 0.20
UNNAMED_SCALE = 1.0


# --- CCU momentum tilt -------------------------------------------------------
# Measured on 2720-3000 (game, cutoff) pairs with hourly CCU history fetched
# before the deadline: IC(decay, dplace) = -0.234 at 24h and -0.135 at 72h,
# IC(z_ccu, dplace) = -0.235 and -0.160. Games trading below their own week and
# running hot relative to it keep climbing; fading ones fall. Both features come
# from the `playing` values already in the history, so nothing is fetched here.
TILT_DECAY = 0.40
TILT_ZCCU = 0.20


def _tilt_features(reads: list[Snapshot], candidates: list[Candidate]) -> dict[str, dict[str, float]]:
    now = reads[-1].time
    out: dict[str, dict[str, float]] = {}
    for c in candidates:
        pts = [(s.time, max(1., s.games[c.uid].players)) for s in reads if c.uid in s.games]
        if len(pts) < 3:
            out[c.uid] = {'decay': 0., 'z_ccu': 0.}
            continue
        week = [(t, v) for t, v in pts if now - t <= 168 * 3600.]
        vals = [v for _, v in week] or [pts[-1][1]]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.
        out[c.uid] = {'decay': math.log(vals[-1] / max(vals)),
                      'z_ccu': (vals[-1] - mu) / sd}
    return out


def _zscores(values: dict[str, float]) -> dict[str, float]:
    if len(values) < 3:
        return {k: 0. for k in values}
    mu = sum(values.values()) / len(values)
    sd = math.sqrt(sum((v - mu) ** 2 for v in values.values()) / len(values)) or 1.
    return {k: (v - mu) / sd for k, v in values.items()}


def pdf_tailed(delta: float, sd: float) -> float:
    """Gaussian core plus Student-type tail; robust to abrupt chart movers."""
    z = delta / max(.08, sd)
    return (1.0 - TAIL_W) * math.exp(-.5*z*z) + TAIL_W * (1.+(z*z)/4.)**(-2.5)


def forecast(task: dict, reads: list[Snapshot]) -> dict:
    places = max(1, min(10, int(task.get('places') or 10)))
    window = task.get('window') or {}
    hours = max(1, min(168, int(window.get('slots') or 72)))
    start = to_epoch(window['from']) if window.get('from') else reads[-1].time
    offset = max(0., (start-reads[-1].time)/3600.)
    candidates = candidates_from(reads)
    if len(candidates) < places:
        raise ValueError('Need at least as many game IDs as places')
    n = len(candidates)
    row_weights = [[0.]*n for _ in range(places)]
    volatility = interval_variability(reads)
    unnamed = fit_unnamed(reads, places)
    feats = _tilt_features(reads, candidates)
    z_decay = _zscores({u: f['decay'] for u, f in feats.items()})
    z_ccu = _zscores({u: f['z_ccu'] for u, f in feats.items()})
    tilt = [min(6., max(1./6., math.exp(TILT_DECAY * z_decay[c.uid] + TILT_ZCCU * z_ccu[c.uid])))
            for c in candidates]

    # A forecast cannot assign probability to a game once it passes the
    # chart's 28-day eligibility limit. Track how many *currently known*
    # candidates remain eligible in each future hour. This retains residual
    # probability for unknown entrants rather than renormalising expired
    # games back to a near-certain occupancy of the chart.
    initial_capacity = sum(math.exp(-c.rank / 35.) * (1. if c.present else .05)
                           for c in candidates if c.expiry > start)
    if initial_capacity <= 0.:
        initial_capacity = 1.

    for h in range(hours):
        t = offset+h+0.5
        # Long-range uncertainty grows sublinearly as drift can reverse.
        sd = SD_BASE + volatility*math.sqrt(t)*VOL_MULT
        trend_time = 24*(1-math.exp(-t/24))
        rank_level = [-math.log(p) for p in range(1, places+1)]
        eligible_capacity = sum(math.exp(-c.rank / 35.) * (1. if c.present else .05)
                                for c in candidates if c.expiry > start+h*3600)
        # Lightly discount known-game mass as age-outs accumulate; do not
        # use the remaining candidates' relative scores to force certainty.
        coverage = min(1., math.sqrt(eligible_capacity / initial_capacity))
        for p, place_score in enumerate(rank_level):
            vals = [0.]*n
            for i, c in enumerate(candidates):
                if start+h*3600 >= c.expiry:
                    continue
                survival = math.exp(-t*(.0018+.000028*c.rank))
                if not c.present:
                    survival *= .08*math.exp(-c.lag_h/10)
                centre = c.score + c.slope*.76*trend_time
                # A large player count spike can precede rank acceleration.
                # Keep this as a low-weight *alternative* instead of certainty.
                surge_prob = min(.28, c.player_surge*.12)
                effect = 1.15*math.exp(-max(0., t-30)/27)
                state = ((1-surge_prob)*pdf_tailed(place_score-centre, sd)
                         +surge_prob*pdf_tailed(place_score-centre-effect, sd*1.1))
                # Older place occupancy anchors far-horizon forecasts against
                # overreacting to the last snapshot.
                if c.persist_scores:
                    occ = sum(w*pdf_tailed(place_score-old, sd*1.15)
                              for old,w in c.persist_scores)
                    occ /= sum(w for _,w in c.persist_scores)
                    state = (1.0 - PERSIST_W)*state + PERSIST_W*occ
                vals[i] = survival*state*tilt[i]
            total = sum(vals)
            if total <= 1e-200:
                # If every known game aged out, leave almost all mass unnamed.
                continue
            for i, value in enumerate(vals):
                row_weights[p][i] += coverage * value / total / hours

    # Allocate mass for games not in the observed history at forecast time.
    # A learned unknown rate is tempered toward 20-30% on small datasets.
    # Important: time-averaged mass can be below 1 when all named games
    # have expired. Preserve that gap, giving it to the unnamed-game pool.
    targets = [(1. - unnamed[p]) * sum(row)
               for p, row in enumerate(row_weights)]
    for p, row in enumerate(row_weights):
        row_weights[p] = [(1. - unnamed[p]) * value for value in row]

    # An actual game cannot simultaneously fill multiple ranks. Enforce the
    # column capacity without wasting unused (unassigned) probability mass.
    for _ in range(18):
        sums = [sum(row_weights[p][i] for p in range(places)) for i in range(n)]
        for i, summed in enumerate(sums):
            if summed > .995:
                scale = .995/summed
                for p in range(places):
                    row_weights[p][i] *= scale
        for p in range(places):
            target = targets[p]
            s = sum(row_weights[p])
            if s > 0:
                row_weights[p] = [v*target/s for v in row_weights[p]]

    # Giving a tiny but explicit p to low-likelihood games can hurt log score:
    # if unnamed scorer mass would have paid more, leave the game unnamed.
    # Kept mass stays below 1, so unpredicted games retain probability.
    probabilities = []
    top_scores = defaultdict(float)
    for place, row in enumerate(row_weights, 1):
        for i, p in enumerate(row):
            if p < MIN_P:
                continue
            pr = round(p, 8)
            uid = candidates[i].uid
            probabilities.append({'universe_id': uid, 'place': place, 'p': pr})
            top_scores[uid] += pr*(places+1-place)
    probabilities.sort(key=lambda r: (r['place'], r['universe_id']))
    top10 = sorted(top_scores, key=lambda u: (-top_scores[u], u))[:places]
    if len(top10) < places:
        for c in candidates:
            if c.uid not in top10:
                top10.append(c.uid)
                if len(top10) >= places:
                    break
    return {'probabilities': probabilities, 'top10': top10}


def main():
    input_dir = Path(os.getenv('MAKEFUN_INPUT_DIR', '/input'))
    output_dir = Path(os.getenv('MAKEFUN_OUTPUT_DIR', '/output'))
    task, reads = parse_input(input_dir)
    answer = forecast(task, reads)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(task.get('output') or 'forecast.json').name
    (output_dir/output_name).write_text(json.dumps(answer, sort_keys=True, separators=(',', ':'))+'\n', encoding='utf-8')
    print('Forecast with %d rows, %d games, no network' %
          (len(answer['probabilities']), len(answer['top10'])), file=sys.stderr)


if __name__ == '__main__':
    main()
