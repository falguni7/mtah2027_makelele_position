"""DFL event data, placed on the tracking frames: which team made a controlled action, and when.

    from mtah.external import dfl, dfl_events
    m = dfl.load("DFL-MAT-J03WMX")
    events = dfl_events.load(m)                # DataFrame, one row per event, with its tracking row
    events[events.controlled].head()           # passes, shots, claimed balls, won tackles ...

The event file (events_raw, see mtah/external/dfl.py) is the official DFL event data: about
1,400-1,700 events per match, each with a wall-clock EventTime, written by human data loggers
during the match. We use it for one thing only: to decide which short possessions in the
tracking are real ("controlled", CHALLENGE.md) and which are just touches (the OR rule in
mtah/external/controlled.py). It is never a model input, because there is no such data for
BAR-ATH.

A controlled action is a moment where a team clearly has the ball under control:

  Play           a pass or cross by a player (also when it starts a throw-in, free kick, corner,
                 goal kick or kick-off), successful or not: to play the ball on deliberately,
                 the player had it under control
  ShotAtGoal     a shot (also a penalty)
  BallClaiming   the ball won without a duel: BallClaimed, InterceptedBall, or BallHeld (the
                 goalkeeper catches it); not Block, which only stops the ball
  TacklingGame   a duel, but only when its winner ends up with the ball: WinnerResult
                 ballClaimed (won it), ballControlRetained (kept it), dribbledAround (beat the
                 opponent with it); the team is the WINNER's. Duels that end in a mere touch
                 (ballcontactSucceeded, layoff) or a foul (fouled) do not count

Not controlled: OtherBallAction (touches, clearances), BallDeflection (off the post, the bar
or the referee), Foul, Offside, cards, substitutions and the other administrative events, and
Delete (a placeholder left where an event was removed after a video review).

Restarts (KickOff, ThrowIn, FreeKick, CornerKick, GoalKick, Penalty) are flagged separately:
their EventTime is the moment the ball is put back into play, so they show where the event
clock sits against the tracking (dead -> alive in BallStatus), which sync_offsets() measures.

Syncing. The tracking frames carry a UTC timestamp (attribute T); clock() reads the first and
last ball frame of each half, and the frame number of an event is the half's first frame plus
its time since that frame times 25. Event times carry a local time-zone offset (+01:00 / +02:00),
which the parsing handles. The loggers' clock is still off by a constant per match (up to
~1.5 s either way); best_shift() finds it as the shift at which the controlled passes best agree
with the tracking's owner (92-96 % after the shift), and load() applies it. sync_offsets()
measures the same error independently at the restarts and agrees with it.

In this file: seconds() (time stamps to one clock), clock() (per-half frame clock from the
positions file, cached), parse() (the event file into a table), rows() (tracking row of each
event by time), best_shift() (the match's constant clock error), load() (all of it for one match)
and sync_offsets() (event clock against BallStatus at restarts, a second check of the shift).
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

from mtah.external import dfl
from mtah.utils import runs_of

# Params
FPS = 25
CLAIMS = ("BallClaimed", "InterceptedBall", "BallHeld")
TACKLE_WINS = ("ballClaimed", "ballControlRetained", "dribbledAround")
RESTARTS = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
EDGE = 2 * FPS              # events this close outside a half go on its first / last frame
MAX_SHIFT = 5 * FPS         # clock shifts tried by best_shift(), either way
BALL_FRAME = re.compile(r'<Frame N="(\d+)" T="([^"]+)"')


def seconds(stamp):
    """Seconds since 1970 (UTC) of an ISO time stamp with a time-zone offset.

    Event times are local ("+02:00" in summer, "+01:00" in winter) and tracking times UTC
    ("+00:00"); turning both into seconds since 1970 puts them on one clock.

        seconds("2023-05-27T15:30:12.230+02:00") - seconds("2023-05-27T13:30:12.600+00:00")
        # -> -0.37    the kick-off event is logged 0.37 s before the first tracking frame
    """
    return datetime.fromisoformat(stamp).timestamp()


def clock(match_id):
    """{half: (first frame number, its time in seconds, last frame number)} of the ball frames.

    Reads the ball's <FrameSet> of each half in the positions file (a full pass over the file,
    ~20 s; the result is kept in cache/external/dfl/<match>_clock.json). Checks that the frames
    are evenly spaced at 25 Hz: the time of the last frame must equal the first plus
    (last - first) / 25 to within 0.1 s, otherwise frame numbers could not be turned into times
    this way and a ValueError is raised.

        clock("DFL-MAT-J03WMX")
        # -> {1: (10000, 1685194212.6, 80000-ish), 2: (100000, ..., ...)}
    """
    path = os.path.join(dfl.CACHE, match_id + "_clock.json")
    if os.path.exists(path):
        with open(path) as handle:
            return {int(k): tuple(v) for k, v in json.load(handle).items()}
    out = {}
    half, first, last = None, None, None
    with open(dfl.paths(match_id)[2], encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                if half is not None:
                    out[half] = (first[0], first[1], last[0])
                    _check(first, last)
                match = dfl.FRAMESET.search(line)
                half = SECTIONS[match.group(1)] if match.group(2) == "BALL" else None
                first = None
            elif half is not None and "<Frame " in line:
                frame = BALL_FRAME.search(line)
                last = (int(frame.group(1)), seconds(frame.group(2)))
                first = first or last
    if half is not None:
        out[half] = (first[0], first[1], last[0])
        _check(first, last)
    os.makedirs(dfl.CACHE, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out


def _check(first, last):
    """Raise ValueError unless two (frame number, seconds) points are 1/25 s per frame apart.

    clock() turns an event time into a frame number assuming the tracking runs at exactly 25 Hz
    with no missing frame numbers. This checks that assumption on the first and last frame of a
    half: their time difference must be (last - first) / 25 to within 0.1 s.

        _check((10000, 0.0), (10250, 10.0))      # passes: 250 frames in 10 s
        _check((10000, 0.0), (10250, 12.0))      # ValueError: 2 s off
    """
    expected = first[1] + (last[0] - first[0]) / FPS
    if abs(last[1] - expected) > 0.1:
        raise ValueError("tracking clock not 25 Hz: {} vs {}".format(last, expected))


def parse(match_id, home, guest):
    """DataFrame of the match's events: time, kind, team, controlled, restart.

    One row per <Event> of the event file, in file order, with

      time        seconds since 1970 (UTC), from EventTime
      kind        the event's type, the tag of its first child (Play, TacklingGame, ThrowIn ...)
      team        0 home, 1 guest, -1 none: the team that acts (for a duel, its winner; for a
                  set piece, the team taking it)
      controlled  True for a controlled action (module docstring): a Play anywhere inside
                  the event, a ShotAtGoal, a BallClaiming of a controlling type, a duel won
                  with the ball
      restart     True for a restart (RESTARTS)

    home and guest are the team ids (DFLMatch.home / .guest), which give the team codes.

        events = parse("DFL-MAT-J03WMX", m.home, m.guest)
        events.iloc[0]
        # -> time 1685194212.23, kind "KickOff", team 1 (Bayern, the guest), controlled True
        #    (the kick-off is a pass), restart True
    """
    code = {home: 0, guest: 1}
    rows = []
    for event in ET.parse(dfl.paths(match_id)[1]).getroot():
        if len(event) == 0:
            continue
        child = event[0]
        kind = child.tag
        if kind == "TacklingGame":
            team = child.get("WinnerTeam", "")
            controlled = child.get("WinnerResult") in TACKLE_WINS
        elif kind == "BallClaiming":
            team = child.get("Team", "")
            controlled = child.get("Type") in CLAIMS
        else:
            # a Play or a shot, on its own or inside a set piece (a penalty holds its shot)
            action = [e for e in child.iter() if e.tag in ("Play", "ShotAtGoal")]
            team = child.get("Team") or (action[0].get("Team") if action else "") or ""
            controlled = bool(action)
        rows.append((seconds(event.get("EventTime")), kind, code.get(team, -1), controlled,
                     kind in RESTARTS))
    out = pd.DataFrame(rows, columns=["time", "kind", "team", "controlled", "restart"])
    out.loc[out.team < 0, "controlled"] = False
    return out


def rows(match, events, clocks):
    """int array: the tracking row of each event, -1 when it falls outside both halves.

    match is a DFLMatch, events a parse() table and clocks the clock() dict. The event's half is
    the later half whose first frame it is not more than 60 s before (events are logged from
    the kick-off on); its frame number is the half's first frame number plus its time since that
    frame times 25, rounded; and the row is where that frame number sits in match.frame_n
    within that half. Events up to EDGE frames (2 s) before the first or after the last frame of
    the half are put on that first or last frame (the kick-off is often logged a moment before
    the tracking starts); events further out get -1.

        clocks = {1: (10000, 0.0, 10099)}, events.time = [2.0, -1.0, 10.0]   # 100 frames, 4 s
        rows(match, events, clocks)        # -> [50, 0, -1]   frame 10050; 9975 is 1 s early,
                                           #    so frame 10000; 10250 is past the end
    """
    out = np.full(len(events), -1, dtype=np.int64)
    half = np.asarray(match.half)
    frame_n = np.asarray(match.frame_n)
    for number, (first, start, last) in sorted(clocks.items()):
        later = [clocks[h][1] for h in clocks if h > number]
        inside = (events.time.to_numpy() >= start - 60) & (
            events.time.to_numpy() < (later[0] - 60 if later else np.inf))
        frame = first + np.round((events.time.to_numpy() - start) * FPS).astype(np.int64)
        ok = inside & (frame >= first - EDGE) & (frame <= last + EDGE)
        frame = np.clip(frame, first, last)
        offset = np.flatnonzero(half == number)[0]
        n = frame_n[half == number]
        at = np.searchsorted(n, frame[ok])
        at = np.minimum(at, len(n) - 1)
        exact = n[at] == frame[ok]
        out[np.flatnonzero(ok)[exact]] = at[exact] + offset
    return out


def best_shift(match, events, max_shift=MAX_SHIFT):
    """(shift in frames, agreement): the clock correction that best lines events up with the owner.

    The event clock and the tracking clock of a match differ by a constant: the loggers' times
    are up to ~2 s early or late (measured: -30 to +38 frames depending on the match). For each
    shift from -max_shift to +max_shift frames, this moves every controlled Play (a pass or
    cross, ~800 per match) by that many rows and counts how often the tracking's BallPossession
    there names the event's team; the shift with the highest share wins. Adding it to an event's
    row gives the tracking row it belongs to.

        a home pass at row 100, the tracking names home only from row 104 on
        best_shift(match, events)          # -> (4, ...)   events are logged 4 frames early
    """
    plays = events[events.controlled & (events.row >= 0) & (events.kind == "Play")]
    at = plays.row.to_numpy()
    team = plays.team.to_numpy()
    owner = np.asarray(match.possession) - 1
    shifts = np.arange(-max_shift, max_shift + 1)
    agree = [(owner[np.clip(at + s, 0, match.n_frames - 1)] == team).mean() for s in shifts]
    k = int(np.argmax(agree))
    return int(shifts[k]), float(agree[k])


def load(match):
    """The parse() table of a DFLMatch with its tracking "row", clock-corrected, sorted by time.

    The one call scripts need. rows() places every event on the tracking frames by time;
    best_shift() then measures the constant clock error of this match and every row is moved
    by it (kept inside its half). Events outside both halves keep row -1. The shift and its
    agreement are stored in events.attrs["shift"] and events.attrs["agreement"].

        events = load(dfl.load("DFL-MAT-J03WMX"))
        events.attrs["shift"], events.attrs["agreement"]       # -> (4, 0.93)
        events[["kind", "team", "controlled", "row"]].iloc[0]  # -> KickOff 1 True 4
    """
    events = parse(match.match_id, match.home, match.guest)
    events["row"] = rows(match, events, clock(match.match_id))
    events = events.sort_values("time", kind="stable").reset_index(drop=True)
    shift, agreement = best_shift(match, events)
    half = np.asarray(match.half)
    placed = events.row.to_numpy() >= 0
    moved = events.row.to_numpy().copy()
    for number in (1, 2):
        inside = np.flatnonzero(half == number)
        mine = placed & (half[np.maximum(moved, 0)] == number)
        moved[mine] = np.clip(moved[mine] + shift, inside[0], inside[-1])
    events["row"] = moved
    events.attrs["shift"], events.attrs["agreement"] = shift, agreement
    return events


def sync_offsets(match, events, window=5 * FPS):
    """Frames from each restart event to the nearest dead -> alive switch of BallStatus.

    For every restart event with a row, the signed distance (event row minus switch row) to
    the closest frame where the ball becomes alive again, if one is within `window` frames
    (5 s); restarts with no switch that close are left out. It is an independent check of
    best_shift(): on rows from rows() alone (no shift) the median is about minus the shift;
    on rows from load() (shifted) it should be near 0. The spread shows how precisely restarts
    are logged (about +/- 1 s).

        alive = [F, F, F, T, T, T], one restart event at row 4
        sync_offsets(match, events)        # -> array([1])   logged 1 frame after the switch
    """
    alive = np.asarray(match.alive)
    half = np.asarray(match.half)
    switches = np.array([a for h in (1, 2) for a, _ in runs_of(alive & (half == h))
                         if a > 0 and half[a - 1] == h])
    out = []
    for row in events.loc[events.restart & (events.row >= 0), "row"]:
        if len(switches) == 0:
            break
        nearest = switches[np.abs(switches - row).argmin()]
        if abs(row - nearest) <= window:
            out.append(row - nearest)
    return np.array(out, dtype=np.int64)
