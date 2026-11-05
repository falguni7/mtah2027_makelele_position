"""Check ideas 7 and 8 on the possession changes we already find. Prints aggregates only.

    python scripts/possession/change_check.py
    sbatch scripts/slurm/run.sbatch scripts/possession/change_check.py

Both ideas are about the moment possession changes, and both are only hints:

  7. a change always comes with a sharp ball acceleration (a touch), but most touches are not
     changes;
  8. a change is usually followed by a rapid change of the teams' shape (the team that won the
     ball pushes up and spreads, the other drops back), but shapes also change for other
     reasons.

Before either goes into a method, this script checks whether the data agree, using the
step-2 control events: a CHANGE is the start of an event whose team differs from the previous
event's, a SAME-TEAM start is one that continues the previous event's team (a pass received),
and RANDOM frames are a fixed-seed sample of graded frames to compare against. Nothing here
uses labels; nothing is written to disk.

1. Idea 7, ball acceleration (mtah/possession/touches.py), on frames with the ball tracked.
   a. median acceleration (m/s^2) at offsets before and after the event start, for changes,
      same-team starts and random tracked frames: a touch shows up as a peak a few frames
      before the event start;
   b. where the strongest acceleration lies in the search window before each change (offset
      in frames from the event start, percentiles) and how strong it is, against the same window
      at random frames: this is what SNAP_BEFORE and MIN_TOUCH_ACCEL in touches.py are set from;
   c. the veto question: the share of changes with no acceleration at or above each threshold
      anywhere from the previous event's stop to 2 frames after the new event's start (ball
      tracked all the way; an acceleration the tracker cannot compute, NaN, counts as no
      touch). If idea 7 holds, this should be close to 0.
   d. what snap_changes() does: how many changes move, and by how much.
   e. a geometric alternative to the acceleration: how many frames before the event start the
      ball arrives at the new team (stays within ARRIVAL_RADIUS of one of its players from then
      on). The control rule waits until the ball also moves with the player, so this says how
      late its change point is compared with the arrival of the ball. "Share at the search
      limit" is the share whose arrival lies SEARCH (25) frames or more before the start, i.e.
      the ball was already near the new team when the 1 s search window began (windows cut
      short by the previous event are not counted there).

2. Idea 8, shape-change rate, from the ball-free team-shape table (mtah/features/team_shape.py).
   For each signal, its rate of change is |x(t + 12 frames) - x(t - 12 frames)| per second
   (+/- 0.48 s, RATE_HALF); "all diffs"
   combines every home-minus-away feature, each scaled by its spread over the graded frames.
   a. the rate at offsets from -5 s to +5 s around changes and same-team starts, in standard
      deviations above its mean on random frames (0 = no different from any other moment);
   b. the offset where it peaks around changes, and how well it separates changes from random
      frames and from same-team starts there (AUC: 0.5 = no better than a coin, 1 = perfect);
   c. the SIGNED change, oriented so that positive means "towards the team that just won the
      ball" (e.g. forward_vel_diff rising when home wins it): the direction of the change,
      in standard deviations of the same 1 s (24-frame) change at random frames (so 0.5 means half a
      typical change, consistently in one direction).

How to read it: idea 7 is supported if 1a shows a clear peak just before the event start for
changes, 1b puts the strongest acceleration a few frames early, well above the random windows,
and 1c is small. Idea 8 is supported if 2a rises clearly above 0 after changes and more than
after same-team starts, with an AUC well above 0.5 in 2b.
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.features import team_shape  # noqa: E402
from mtah.possession import touches  # noqa: E402
from mtah.possession.control import clean_ball, control_events  # noqa: E402

# Params
FPS = 25
N_RANDOM = 5000
SEARCH = 25                         # frames before the event start searched in 1b (1 s)
THRESHOLDS = (15.0, 30.0, 50.0)     # m/s^2
ARRIVAL_RADIUS = 2.0                # metres, ball to the nearest player of the new team (1e)
RATE_HALF = FPS // 2                # rate of change over +/- 12 frames (0.48 s)
SHAPE_SIGNALS = ["advance_diff", "forward_vel_diff", "approach_diff", "hull_area_diff",
                 "spread_x_diff", "marking_diff"]
SIGNED = ["advance_diff", "forward_vel_diff", "approach_diff", "hull_area_diff"]


def split_starts(events):
    """(changes, same_team): event starts split by whether the team differs from the previous one.

    Each is a list of (start, previous_stop, team) for the events from the second on; the first
    event of the match has no previous event and is in neither.

        events = [(10, 20, 0), (40, 50, 1), (60, 70, 1)]
        split_starts(events)
        # -> ([(40, 20, 1)], [(60, 50, 1)])   one change to away, one away pass received
    """
    changes, same = [], []
    for (_, previous_stop, previous_team), (start, _, team) in zip(events, events[1:]):
        (changes if team != previous_team else same).append((start, previous_stop, team))
    return changes, same


def at_offsets(values, anchors, offsets):
    """(anchors, offsets) array of values[anchor + offset], NaN where that is off the match.

    values is one value per frame (e.g. the ball's acceleration), anchors are frames of
    interest (e.g. the starts of control events after a possession change), offsets frame
    shifts around them. Row i, column j is the value offsets[j] frames after anchors[i]; frames
    before 0 or past the end are NaN. Averaging the rows gives the typical profile of a signal
    around the anchors, e.g. whether ball acceleration peaks just before a change.

        values = [0, 1, 2, 3, 4]
        at_offsets(values, [1, 4], [-1, 0, 1])     # -> [[0, 1, 2], [3, 4, nan]]
    """
    values = np.asarray(values, dtype=np.float64)
    index = np.asarray(anchors)[:, None] + np.asarray(offsets)[None, :]
    ok = (index >= 0) & (index < len(values))
    out = np.full(index.shape, np.nan)
    out[ok] = values[index[ok]]
    return out


def rate(values, half=RATE_HALF):
    """|values[t + half] - values[t - half]| per second; NaN at the ends or where either is NaN.

    values is one value per frame (e.g. a team-shape feature such as the hull area), half the
    number of frames looked ahead and back (default RATE_HALF, 12 frames = 0.48 s). The result is how fast
    the value changes around each frame, per second, regardless of direction: idea 8's "rapid
    shape change" signal. The first and last `half` frames have no window and are NaN.

        values = [0, 0, 0, 1, 2, 2, 2], half = 1 (at 25 fps the window is 2 frames, 0.08 s)
        rate(values, 1)   # -> [nan, 0, 12.5, 25, 12.5, 0, nan]
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan)
    out[half:-half] = (values[2 * half:] - values[:-2 * half]) * FPS / (2 * half)
    return np.abs(out)


def signed_change(values, anchors, sign, after, half=RATE_HALF):
    """Mean of sign * (values[a + after + half] - values[a + after - half]) over anchors a.

    The change of a signal over 1 s centred `after` frames past each anchor, multiplied by +1
    or -1 per anchor (here: +1 when home won the ball, -1 when away did), averaged. Positive
    means the signal moves towards the team that just won the ball.

        values = [0, 0, 1, 2], anchors = [1], sign = [1], after = 1, half = 1
        signed_change(values, anchors, sign, 1, 1)   # -> 2.0   (values[3] - values[1])
    """
    values = np.asarray(values, dtype=np.float64)
    a = np.asarray(anchors) + after
    ok = (a - half >= 0) & (a + half < len(values))
    return float(np.nanmean(np.asarray(sign)[ok] * (values[a[ok] + half] - values[a[ok] - half])))


if __name__ == '__main__':
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    events = control_events(match)
    changes, same = split_starts(events)
    rng = np.random.default_rng(0)
    print("control events {}, changes {}, same-team starts {}".format(
        len(events), len(changes), len(same)))

    # ---------------------------------------------------------------- 1. ball acceleration
    accel = touches.ball_acceleration(match)
    _, ball_present = clean_ball(match)
    tracked = np.flatnonzero(graded & np.isfinite(accel))
    random_tracked = rng.choice(tracked, N_RANDOM, replace=False)
    change_starts = np.array([s for s, _, _ in changes])
    same_starts = np.array([s for s, _, _ in same])

    print("\n1a. median ball acceleration (m/s^2) by offset from the event start (frames)")
    offsets = np.arange(-20, 8, 2)
    print("    offset      " + " ".join("{:>5d}".format(o) for o in offsets))
    for name, anchors in (("changes", change_starts), ("same team", same_starts),
                          ("random", random_tracked)):
        print("    {:<11} ".format(name) + " ".join(
            "{:5.1f}".format(v) for v in np.nanmedian(at_offsets(accel, anchors, offsets), axis=0)))

    print("\n1b. strongest acceleration in the {} frames before the start (+2 after), "
          "not before the previous event's end".format(SEARCH))
    peaks = {"changes": [], "random": []}
    for start, previous_stop, _ in changes:
        low, high = max(start - SEARCH, previous_stop), start + 3
        window = accel[low:high]
        if np.isfinite(window).sum() >= 0.8 * len(window) and len(window) >= 5:
            peaks["changes"].append((low + int(np.nanargmax(window)) - start, np.nanmax(window)))
    for start in random_tracked:
        low = max(start - SEARCH, 0)
        window = accel[low:start + 3]
        if len(window) and np.isfinite(window).sum() >= 0.8 * len(window):
            peaks["random"].append((low + int(np.nanargmax(window)) - start, np.nanmax(window)))
    for name, rows in peaks.items():
        rows = np.array(rows)
        print("    {:<8} n {:>5}  offset p10/p25/p50/p75/p90 {}  peak m/s^2 p10/p50/p90 {}".format(
            name, len(rows), np.percentile(rows[:, 0], [10, 25, 50, 75, 90]).round(0),
            np.percentile(rows[:, 1], [10, 50, 90]).round(1)))
        print("             share with peak >= " + "  ".join(
            "{:.0f}: {:.2f}".format(t, (rows[:, 1] >= t).mean()) for t in THRESHOLDS))

    print("\n1c. changes with NO acceleration above a threshold between the two events "
          "(ball tracked throughout)")
    whole = [accel[previous_stop:start + 3] for start, previous_stop, _ in changes
             if ball_present[previous_stop:start + 3].all() and start + 3 - previous_stop >= 5]
    print("    {} changes with the ball tracked between the events".format(len(whole)))
    for t in THRESHOLDS:
        print("    none >= {:>4.0f} m/s^2: {:.3f}".format(
            t, np.mean([not (np.nan_to_num(w) >= t).any() for w in whole])))

    print("\n1d. snap_changes (SNAP_BEFORE {}, MIN_TOUCH_ACCEL {:.0f})".format(
        touches.SNAP_BEFORE, touches.MIN_TOUCH_ACCEL))
    snapped = touches.snap_changes(events, accel)
    shift = np.array([s0 - s1 for (s0, _, _), (s1, _, _) in zip(events, snapped)])
    moved = shift[shift > 0]
    print("    changes moved {} of {} ({:.0%}); frames earlier p25/p50/p75/max {}".format(
        len(moved), len(changes), len(moved) / max(len(changes), 1),
        np.percentile(moved, [25, 50, 75, 100]).round(0) if len(moved) else "-"))

    print("\n1e. arrival: first frame from which the ball stays within {} m of a player of the "
          "new team up to the event start".format(ARRIVAL_RADIUS))
    players = np.asarray(match.players_xy, dtype=np.float64)
    ball, _ = clean_ball(match)
    arrival = []
    for start, previous_stop, team in changes:
        low = max(start - SEARCH, previous_stop)
        frames = np.arange(low, start + 1)
        mine = np.asarray(match.team_of_slot) == team
        distance = np.hypot(players[frames][:, mine, 0] - ball[frames, None, 0],
                            players[frames][:, mine, 1] - ball[frames, None, 1])
        near = np.nan_to_num(np.nanmin(np.where(np.isfinite(distance), distance, np.inf), axis=1),
                             nan=np.inf, posinf=np.inf) < ARRIVAL_RADIUS
        if not near[-1]:
            continue
        outside = np.flatnonzero(~near)
        arrival.append((outside[-1] + 1 if len(outside) else 0) + low - start)
    arrival = np.array(arrival)
    print("    {} changes; offset from the event start p10/p25/p50/p75/p90 {} frames; "
          "share at the search limit {:.2f}".format(
              len(arrival), np.percentile(arrival, [10, 25, 50, 75, 90]).round(0),
              np.mean(arrival <= -SEARCH)))

    # ---------------------------------------------------------------- 2. shape-change rate
    table = team_shape.compute(match)
    diffs = [c for c in team_shape.COLUMNS if c.endswith("_diff")]
    scaled = (table[diffs] - table[diffs][graded].mean()) / table[diffs][graded].std()
    combined = np.sqrt(np.nanmean(np.column_stack(
        [rate(scaled[c].to_numpy()) ** 2 for c in diffs]), axis=1))
    signals = {name: rate(table[name].to_numpy()) for name in SHAPE_SIGNALS}
    signals["all diffs"] = combined
    random_graded = rng.choice(np.flatnonzero(graded), N_RANDOM, replace=False)
    # every 0.5 s from -5 s to +5 s, rounded to whole frames (0 is in the grid)
    offsets = np.round(np.arange(-5.0, 5.01, 0.5) * FPS).astype(int)

    print("\n2a. shape-change rate around event starts, in SDs above random frames; "
          "offsets in seconds")
    print("    offset                  " + " ".join("{:>5.1f}".format(o / FPS) for o in offsets))
    best = {}
    for name, values in signals.items():
        base = values[random_graded]
        mean, sd = np.nanmean(base), np.nanstd(base)
        for label, anchors in (("changes", change_starts), ("same team", same_starts)):
            profile = (np.nanmean(at_offsets(values, anchors, offsets), axis=0) - mean) / sd
            print("    {:<16} {:<8}".format(name, label) + " ".join(
                "{:5.2f}".format(v) for v in profile))
            if label == "changes":
                best[name] = offsets[int(np.nanargmax(profile))]

    print("\n2b. at each signal's peak offset after changes: AUC changes vs random, "
          "vs same-team starts")
    for name, values in signals.items():
        lag = best[name]
        c = at_offsets(values, change_starts, [lag])[:, 0]
        s = at_offsets(values, same_starts, [lag])[:, 0]
        r = values[random_graded]
        aucs = []
        for other in (r, s):
            x = np.concatenate([c, other])
            y = np.concatenate([np.ones(len(c)), np.zeros(len(other))])
            ok = np.isfinite(x)
            aucs.append(roc_auc_score(y[ok], x[ok]))
        print("    {:<16} peak at {:+5.1f} s   AUC vs random {:.3f}   vs same team {:.3f}".format(
            name, lag / FPS, aucs[0], aucs[1]))

    print("\n2c. signed change over 1 s, towards the team that won the ball (+ = home "
          "won it and the value rose, or away won it and it fell)")
    sign = np.array([1.0 if team == 0 else -1.0 for _, _, team in changes])
    afters = np.round(np.arange(-2.0, 4.01, 0.5) * FPS).astype(int)
    print("    in SDs of the same 1 s change at random frames")
    print("    centred at (s)          " + " ".join("{:>5.1f}".format(a / FPS) for a in afters))
    for name in SIGNED:
        values = table[name].to_numpy()
        h = RATE_HALF
        r = random_graded[(random_graded >= h) & (random_graded + h < len(values))]
        sd = np.nanstd(values[r + h] - values[r - h])
        print("    {:<24}".format(name) + " ".join(
            "{:5.2f}".format(signed_change(values, change_starts, sign, a) / sd) for a in afters))
