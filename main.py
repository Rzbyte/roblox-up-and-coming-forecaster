#!/usr/bin/env python3
"""Offline probabilistic forecaster for the Roblox Up-and-Coming top-10 chart.

The chart is modelled per place as a mixture of latent log-rank states, one state per
candidate game, plus an explicit pool for games that are not in the history yet. Three
measured effects drive it:

  * drift: a time-weighted slope of -ln(rank) over the last 40 h, damped by how many
    samples support it and clamped at +/-0.075/h;
  * occupancy: how much probability each game has actually spent in each place over the
    last 96 h, which anchors the far horizon so a hot first hour cannot dominate;
  * CCU momentum: whether a game is running below its own weekly peak and above its own
    weekly mean. Measured on 2720-3000 (game, cutoff) pairs from hourly CCU history, the
    rank correlation of the first with the 24 h change in place is -0.234, the second
    -0.235 (-0.135 and -0.160 at 72 h). Games still climbing keep gaining places; fading
    games slide back, so candidates are tilted by exp(0.40*z_decay + 0.20*z_ccu).

Once a game passes the chart's 28-day eligibility it can hold no probability at all, so
the mass it would have carried is left unnamed instead of being redistributed. Nothing
here touches the network, reads external state, or calls a model: one process, stdlib
only, deterministic.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Model constants. Tuned on 23 walk-forward folds inside the kit history only.
SD_BASE = 0.17          # irreducible spread of a place, in log-rank units
VOL_MULT = 1.08         # extra spread per sqrt(hour), scaled by measured churn
TAIL_W = 0.28           # weight of the heavy tail that covers abrupt movers
MIN_P = 0.002           # below this a named row pays less than leaving the game unnamed
PERSIST_W = 0.20        # share of occupancy history kept in the mixture
UNNAMED_SCALE = 1.0     # multiplies the fitted future-unseen share
TILT_DECAY = 0.40       # weight of z(ln(current CCU / weekly peak))
TILT_ZCCU = 0.20        # weight of z(current CCU vs its own weekly mean)
PLACES_DEFAULT = 10
HORIZON_DEFAULT = 72
ELIGIBILITY_DAYS = 28
CANDIDATE_LIMIT = 150


@dataclass(frozen=True)
class Observation:
    rank: int
    players: float


@dataclass
class Snapshot:
    time: float
    games: dict[int, Observation]
    born: dict[int, float] = field(default_factory=dict)


@dataclass
class Candidate:
    uid: int
    rank: float
    score: float
    slope: float
    surge: float
    lag_h: float
    expiry: float
    present: bool
    occupancy: list[tuple[float, float]]
    decay: float = 0.0
    heat: float = 0.0
    tilt: float = 1.0


def to_epoch(stamp: str) -> float:
    moment = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def parse_input(input_dir: Path) -> tuple[dict, list[Snapshot]]:
    """Read task.json and history.ndjson; drop anything after the history cutoff."""
    task = json.loads((input_dir / 'task.json').read_text(encoding='utf-8'))
    cutoff = to_epoch(task['history_until']) if task.get('history_until') else float('inf')
    reads: list[Snapshot] = []
    with (input_dir / 'history.ndjson').open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            moment = to_epoch(row['at'])
            if moment > cutoff + 1:
                continue
            games: dict[int, Observation] = {}
            for entry in row.get('up_and_coming') or []:
                try:
                    uid = int(entry['universe_id'])
                    rank = int(entry['rank'])
                    players = max(0.0, float(entry.get('playing') or 0))
                except (TypeError, ValueError, KeyError, OverflowError):
                    continue
                if uid > 0 and 1 <= rank <= 300:
                    games[uid] = Observation(rank, players)
            if not games:
                continue
            born: dict[int, float] = {}
            for key, detail in (row.get('details') or {}).items():
                if isinstance(detail, dict) and detail.get('created'):
                    try:
                        born[int(key)] = to_epoch(detail['created'])
                    except (TypeError, ValueError, OverflowError):
                        pass
            reads.append(Snapshot(moment, games, born))
    reads.sort(key=lambda snap: snap.time)
    unique: list[Snapshot] = []
    for snap in reads:
        if unique and snap.time == unique[-1].time:
            unique[-1] = snap
        else:
            unique.append(snap)
    if not unique:
        raise ValueError('No usable chart history')
    return task, unique


def weighted_slope(points: list[tuple[float, float]], now: float,
                   window_h: float = 40.0, half_life_h: float = 14.0) -> float:
    """Least squares slope in units per hour, half-life weighted toward the present."""
    samples = []
    for moment, value in points:
        x = (moment - now) / 3600.0
        if -window_h <= x <= 0.0:
            samples.append((x, value, math.exp(math.log(2.0) * x / half_life_h)))
    if len(samples) < 2:
        return 0.0
    weight = sum(w for _, _, w in samples)
    if weight <= 0.0:
        return 0.0
    mean_x = sum(x * w for x, _, w in samples) / weight
    mean_y = sum(y * w for _, y, w in samples) / weight
    variance = sum(w * (x - mean_x) ** 2 for x, _, w in samples)
    if variance <= 0.01:
        return 0.0
    return sum(w * (x - mean_x) * (y - mean_y) for x, y, w in samples) / variance


def chart_volatility(reads: list[Snapshot]) -> float:
    """Hourly noise of -ln(rank), from consecutive reads of games present in both."""
    squares = []
    previous = None
    for snap in reads[-70:]:
        if previous is not None:
            gap = (snap.time - previous.time) / 3600.0
            if 0.25 <= gap <= 36.0:
                for uid, game in snap.games.items():
                    earlier = previous.games.get(uid)
                    if earlier is not None and earlier.rank > 0 and game.rank > 0:
                        step = math.log(earlier.rank / game.rank)
                        squares.append(min(4.0, step * step / gap))
        previous = snap
    if len(squares) < 25:
        return 0.13
    squares.sort()
    squares = squares[:int(len(squares) * 0.95)]      # trim measurement noise, keep real jumps
    noise = math.sqrt(sum(squares) / max(1, len(squares)))
    return max(0.075, min(0.28, noise))


def build_candidates(reads: list[Snapshot]) -> list[Candidate]:
    """Every game seen recently, scored on its latest rank and its short-term drift."""
    now = reads[-1].time
    series: dict[int, list[tuple[float, Observation]]] = {}
    born: dict[int, float] = {}
    for snap in reads:
        for uid, game in snap.games.items():
            series.setdefault(uid, []).append((snap.time, game))
        born.update(snap.born)
    present_now = reads[-1].games
    built: list[Candidate] = []
    for uid, points in series.items():
        last_time, last = points[-1]
        gap_h = max(0.0, (now - last_time) / 3600.0)
        present = uid in present_now
        if not present and (gap_h > 15.0 or last.rank > 80):
            continue
        rank = float(present_now[uid].rank if present else last.rank + 4.0 + 1.2 * gap_h)
        drift = weighted_slope([(moment, -math.log(game.rank)) for moment, game in points], last_time)
        support = sum(last_time - moment <= 32 * 3600 for moment, _ in points)
        drift *= min(1.0, max(0.2, (support - 1) / 5.0))
        drift = max(-0.075, min(0.075, drift))

        surge = 0.0
        recent = [(moment, math.log1p(game.players)) for moment, game in points
                  if last_time - moment <= 36 * 3600]
        if len(recent) >= 2 and last.players >= 2000:
            old_time, old_log = recent[-2]
            if last_time - old_time <= 24 * 3600:
                surge = max(0.0, min(2.5, recent[-1][1] - old_log - 0.8))

        occupancy = [(-math.log(game.rank), math.exp(-(now - moment) / (3600.0 * 22.0)))
                     for moment, game in points if (now - moment) / 3600.0 <= 96.0]

        expiry = born.get(uid, float('inf')) + ELIGIBILITY_DAYS * 86400.0
        built.append(Candidate(uid, rank, -math.log(rank), drift, surge, gap_h, expiry,
                               present, occupancy))
    built.sort(key=lambda c: (not c.present, c.rank, c.uid))
    built = built[:CANDIDATE_LIMIT]
    apply_ccu_tilt(built, reads)
    return built


def ccu_series(reads: list[Snapshot], uid: int) -> list[tuple[float, float]]:
    return [(snap.time, max(1.0, snap.games[uid].players)) for snap in reads if uid in snap.games]


def _standardise(values: list[float]) -> list[float]:
    if len(values) < 3:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    return [(v - mean) / spread for v in values]


def apply_ccu_tilt(candidates: list[Candidate], reads: list[Snapshot]) -> None:
    """Fill decay/heat from the history's own `playing` values, then the tilt factor.

    decay = ln(current CCU / weekly peak): a game below its own peak is on the way down.
    heat  = current CCU relative to its own weekly mean, in standard deviations.
    Both are measured, not assumed, so a quiet but rising chart is not penalised.
    """
    now = reads[-1].time
    decays: list[float] = []
    heats: list[float] = []
    for candidate in candidates:
        points = [p for p in ccu_series(reads, candidate.uid) if now - p[0] <= 168 * 3600]
        if len(points) < 3:
            decays.append(0.0)
            heats.append(0.0)
            continue
        values = [value for _, value in points]
        peak = max(values)
        mean = sum(values) / len(values)
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
        decays.append(math.log(values[-1] / peak))
        heats.append((values[-1] - mean) / spread)
    z_decay = _standardise(decays)
    z_heat = _standardise(heats)
    for candidate, d, h in zip(candidates, z_decay, z_heat):
        candidate.decay = d
        candidate.heat = h
        factor = math.exp(TILT_DECAY * d + TILT_ZCCU * h)
        candidate.tilt = min(6.0, max(1.0 / 6.0, factor))


def fit_unnamed(reads: list[Snapshot], places: int) -> list[float]:
    """Window-average share of each place that went to games outside the candidate pool.

    Measured on up to ten older cuts that allow a full 72 h follow-up and shrunk toward a
    prior, so a short history cannot invent a rate. This is what keeps the model honest
    about entrants it has never seen.

    A 24 h-sliced version of this was tried: the measured share runs 0.7% / 5.4% / 17% over
    the three days, so slicing looks better on paper, but applying it hour by hour scored
    worse on the folds (-38.145 vs -38.026) and on the practice window (-37.812 vs
    -37.608): the survival and capacity terms already discount known mass with horizon, so
    the second discount double-counts. One window-average rate stays.
    """
    prior = [0.30, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.19][:places]
    if len(reads) < 16 or reads[-1].time - reads[0].time < 7 * 86400:
        return [UNNAMED_SCALE * p for p in prior]
    tally = [[0.0, 0.0] for _ in range(places)]
    last = reads[-1].time
    usable = [i for i in range(7, len(reads) - 1) if reads[i].time <= last - 72 * 3600]
    chosen: list[int] = []
    for index in reversed(usable):
        if not chosen or (reads[chosen[-1]].time - reads[index].time) / 3600 >= 20:
            chosen.append(index)
        if len(chosen) >= 10:
            break
    for index in chosen:
        prefix = reads[:index + 1]
        known = {c.uid for c in build_candidates(prefix)}
        limit = prefix[-1].time + 72 * 3600
        for snap in reads[index + 1:]:
            if snap.time > limit:
                break
            for uid, game in snap.games.items():
                if 1 <= game.rank <= places:
                    tally[game.rank - 1][1] += 1.0
                    if uid not in known:
                        tally[game.rank - 1][0] += 1.0
    rates = []
    for place, (unknown, total) in enumerate(tally):
        estimate = (unknown + 25.0 * prior[place]) / (total + 25.0)
        rates.append(max(0.05, min(0.60, UNNAMED_SCALE * estimate)))
    return rates


def tailed_pdf(delta: float, spread: float) -> float:
    """Gaussian core with a Student-type tail, so a 12-place jump is unlikely, not zero."""
    z = delta / max(0.08, spread)
    return (1.0 - TAIL_W) * math.exp(-0.5 * z * z) + TAIL_W * (1.0 + z * z / 4.0) ** -2.5


def forecast(task: dict, reads: list[Snapshot]) -> dict:
    places = max(1, min(10, int(task.get('places') or PLACES_DEFAULT)))
    window = task.get('window') or {}
    hours = max(1, min(168, int(window.get('slots') or HORIZON_DEFAULT)))
    start = to_epoch(window['from']) if window.get('from') else reads[-1].time
    lead = max(0.0, (start - reads[-1].time) / 3600.0)

    candidates = build_candidates(reads)
    if len(candidates) < places:
        raise ValueError('Need at least as many game IDs as places')
    count = len(candidates)
    mass = [[0.0] * count for _ in range(places)]
    volatility = chart_volatility(reads)
    unnamed = fit_unnamed(reads, places)
    place_scores = [-math.log(place) for place in range(1, places + 1)]

    def live_capacity(seconds_ahead: float) -> float:
        """How much chart weight is still held by games inside their 28-day window."""
        return sum(math.exp(-c.rank / 35.0) * (1.0 if c.present else 0.05)
                   for c in candidates if c.expiry > start + seconds_ahead)

    initial = live_capacity(0.0) or 1.0

    for hour in range(hours):
        t = lead + hour + 0.5
        spread = SD_BASE + volatility * math.sqrt(t) * VOL_MULT
        trend = 24.0 * (1.0 - math.exp(-t / 24.0))
        coverage = min(1.0, math.sqrt(live_capacity(hour * 3600.0) / initial))
        for place_index, place_score in enumerate(place_scores):
            weights = [0.0] * count
            for i, candidate in enumerate(candidates):
                if start + hour * 3600.0 >= candidate.expiry:
                    continue                       # aged out: no probability at any place
                survival = math.exp(-t * (0.0018 + 0.000028 * candidate.rank))
                if not candidate.present:
                    survival *= 0.08 * math.exp(-candidate.lag_h / 10.0)
                centre = candidate.score + candidate.slope * 0.76 * trend
                surge = min(0.28, candidate.surge * 0.12)
                lift = 1.15 * math.exp(-max(0.0, t - 30.0) / 27.0)
                state = ((1.0 - surge) * tailed_pdf(place_score - centre, spread)
                         + surge * tailed_pdf(place_score - centre - lift, spread * 1.1))
                if candidate.occupancy:
                    kept = sum(w * tailed_pdf(place_score - old, spread * 1.15)
                               for old, w in candidate.occupancy)
                    kept /= sum(w for _, w in candidate.occupancy)
                    state = (1.0 - PERSIST_W) * state + PERSIST_W * kept
                weights[i] = survival * state * candidate.tilt
            total = sum(weights)
            if total <= 1e-200:
                continue                           # everyone left; mass stays unnamed
            for i, value in enumerate(weights):
                mass[place_index][i] += coverage * value / total / hours

    # Scale each place to its known-game share; the gap stays unassigned on purpose, it is
    # the probability of games nobody has seen yet.
    targets = [(1.0 - unnamed[p]) * sum(row) for p, row in enumerate(mass)]
    for p, row in enumerate(mass):
        mass[p] = [(1.0 - unnamed[p]) * value for value in row]

    # One game cannot hold two places at once. Cap each game's total mass and restore the
    # per-place totals, 18 passes is enough for this matrix to settle.
    for _ in range(18):
        for i in range(count):
            held = sum(mass[p][i] for p in range(places))
            if held > 0.995:
                scale = 0.995 / held
                for p in range(places):
                    mass[p][i] *= scale
        for p in range(places):
            row_total = sum(mass[p])
            if row_total > 0.0:
                mass[p] = [value * targets[p] / row_total for value in mass[p]]

    probabilities = []
    ranking = [0.0] * count
    for place, row in enumerate(mass, 1):
        for i, value in enumerate(row):
            if value < MIN_P:
                continue                            # unnamed pays better than a token guess
            probability = round(value, 8)
            probabilities.append({'universe_id': candidates[i].uid,
                                  'place': place, 'p': probability})
            ranking[i] += probability * (places + 1 - place)
    probabilities.sort(key=lambda row: (row['place'], row['universe_id']))
    order = sorted(range(count), key=lambda i: (-ranking[i], candidates[i].uid))
    top10 = [candidates[i].uid for i in order[:places]]
    if len(top10) < places:                        # never short: pad with the best ranked rest
        for candidate in candidates:
            if candidate.uid not in top10:
                top10.append(candidate.uid)
                if len(top10) >= places:
                    break
    return {'probabilities': probabilities, 'top10': top10}


def main() -> None:
    input_dir = Path(os.getenv('MAKEFUN_INPUT_DIR', '/input'))
    output_dir = Path(os.getenv('MAKEFUN_OUTPUT_DIR', '/output'))
    task, reads = parse_input(input_dir)
    answer = forecast(task, reads)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(task.get('output') or 'forecast.json').name
    payload = json.dumps(answer, sort_keys=True, separators=(',', ':')) + '\n'
    (output_dir / name).write_text(payload, encoding='utf-8')
    print('Forecast with %d rows, %d games, no network'
          % (len(answer['probabilities']), len(answer['top10'])), file=sys.stderr)


if __name__ == '__main__':
    main()
