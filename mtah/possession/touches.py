"""Ball touches from the tracked ball's acceleration (idea 7): when exactly does possession change?

    from mtah.possession.control import control_events
    from mtah.possession.touches import ball_acceleration, snap_changes
    accel = ball_acceleration(match)               # m/s^2 per frame, NaN where unknown
    events = snap_changes(control_events(match), accel)

The physics: a ball in flight or rolling keeps its velocity, slowly losing speed. Every time a
player touches it -- a pass, a trap, a shot, an interception, a deflection -- its velocity
changes abruptly, in speed, direction or both. So a possession change can only happen at a
frame where the ball accelerates sharply. The reverse is not true: most touches are a player
keeping the ball (dribbling, passing to a teammate), not a change.

The step-2 control rule (mtah/possession/control.py) puts a change at the start of the new
team's control event: the first frame where the ball is close to the new player AND moving with
them. That is usually a few frames after the actual touch, because the ball must first slow
down to the player's speed. snap_changes() tries moving each change back to the touch: the
strongest acceleration shortly before the new team's event.

Result of the check (scripts/possession/change_check.py, work log): on this match the ATD ball
track is too smooth for this. Touches do not stand out: the strongest acceleration before a
change is no stronger than in any random second of play (median ~10 m/s^2), and 9 in 10 changes
have no acceleration above 30 m/s^2 at all, so snap_changes() moves almost nothing. No method
uses it; the module is kept for the check and for a ball track from our own detection (step 5),
which may be sharper.

The same physics read the other way round is used (hmm_v16_touch, mtah/methods.py): not "where
is the touch" but "where is there clearly NO touch". Touches are smeared in this ball track, so
a quiet window cannot prove much on its own, but a ball tracked on every frame for a second
whose velocity hardly changes was played by nobody. A switch INTO possession (H -> A, A -> H,
D -> H, D -> A) there is physically impossible, so the temporal model is made to pay
NO_TOUCH_COST extra for it (no_touch_costs, used by mtah.temporal.hmm.viterbi as the third
array of its switch costs). Switches into D are not affected: the ball can go out or the
referee can whistle without anyone touching it. In a missed stoppage, where the ball lies still
or is carried to the restart, the cheapest path then usually becomes H -> D where play stops
and D -> A at the restart kick (a touch). The settings were fixed from the measurement before
any result was seen (work log, 2026-11-14): velocities over 0.2 s, a window of +/- 0.5 s, a
limit of 1.5 m/s (well above tracking noise once averaged over 0.2 s, well below any kick or
trap), and the cost of a switch into D (DEAD_SWITCH, 4).

In this file: ball_acceleration() measures how sharply the ball's velocity changes on every
frame; snap_changes() moves the start of every control event that changes the team back to the
touch that began it; velocity_change() and no_touch_frames() find the frames with clear
evidence of no touch, and no_touch_costs() turns them into the extra switch cost.
"""

import numpy as np

from mtah.possession.control import clean_ball, velocity

# Params
FPS = 25
ACCEL_HALF = 2              # acceleration from velocities +/- this many frames apart
SNAP_BEFORE = 12            # frames (0.48 s) before an event start searched for the touch
SNAP_AFTER = 2              # frames after it, as the event start itself can be a frame early
MIN_TOUCH_ACCEL = 30.0      # m/s^2: below this the strongest acceleration is not a touch
CHANGE_HALF = 5             # frames (0.2 s): velocity_change compares the velocities over the
                            # 5 frames before and the 5 frames after each frame
MAX_STEP = 1.6              # metres per frame (40 m/s): a longer step is a tracking jump
NO_TOUCH_WINDOW = 12        # frames (~0.5 s) on each side of a switch that must all be quiet
NO_TOUCH_LIMIT = 1.5        # m/s: "quiet" means a velocity change below this
NO_TOUCH_COST = 4.0         # extra cost of a switch into H or A on a no-touch frame


def ball_acceleration(match):
    """How sharply the ball's velocity changes, in m/s^2 per frame; NaN where it is unknown.

    The ball is first cleaned of one-frame tracking spikes (control.clean_ball). Its velocity
    (control.velocity, over +/- 2 frames) is compared 2 frames before and 2 frames after each
    frame; the size of the difference, divided by that 0.16 s, is the acceleration. A change of
    direction counts as much as a change of speed, so a pass received and redirected first
    time shows up just as clearly as a ball stopped dead. Any frame that needs a ball position
    the tracker does not have (frames i-4, i and i+4: the ends and the middle of the two
    velocity windows; frames in between are bridged) is NaN, as are the first and last 4 frames.

    Toy example, 25 fps: a ball rolling at 10 m/s along x is stopped dead on frame 10.

        velocity before frame 10     # (10, 0) m/s
        velocity after frame 10      # (0, 0) m/s
        ball_acceleration(match)[10] # -> 62.5   (10 m/s lost in 0.16 s)
        on frames far from 10        # -> 0.0    (constant velocity, no touch)
    """
    ball, _ = clean_ball(match)
    v = velocity(ball)
    k = ACCEL_HALF
    out = np.full(len(ball), np.nan)
    change = v[2 * k:] - v[:-2 * k]
    out[k:-k] = np.hypot(change[:, 0], change[:, 1]) * FPS / (2 * k)
    return out


def snap_changes(events, accel):
    """Control events with each team change moved back to the touch that started it.

    events are (start, stop, team) as from control_events(), sorted by start; accel is
    ball_acceleration(). For every event whose team differs from the previous event's (a
    possession change), the frames from SNAP_BEFORE (12) before its start to SNAP_AFTER (2) after
    are searched for the strongest ball acceleration, never going back onto a frame of the
    previous event (the ball was still the other team's then; at the earliest the new start is
    the previous event's stop, the first frame after it). If that acceleration is at
    least MIN_TOUCH_ACCEL (30 m/s^2), the event now starts there; otherwise -- no clear touch, for
    example because the ball was not tracked -- it is left alone. Events that keep the same team
    are never moved, and an event is never made later than it was.

        accel  = 0 everywhere except 45 m/s^2 on frame 37
        events = [(10, 20, 0), (40, 50, 1), (60, 70, 1)]     # home, then away from frame 40
        snap_changes(events, accel)
        # -> [(10, 20, 0), (37, 50, 1), (60, 70, 1)]
        #    the change to away now starts at the touch on frame 37; the second away event
        #    is a teammate's, not a change, so it stays
    """
    accel = np.asarray(accel)
    out = list(events[:1])
    for (_, previous_stop, previous_team), (start, stop, team) in zip(events, events[1:]):
        if team != previous_team:
            low = max(start - SNAP_BEFORE, previous_stop)
            high = min(start + SNAP_AFTER + 1, stop, len(accel))
            window = accel[low:high]
            if len(window) and np.isfinite(window).any():
                touch = low + int(np.nanargmax(window))
                if window[touch - low] >= MIN_TOUCH_ACCEL and touch < start:
                    start = touch
        out.append((start, stop, team))
    return out


def velocity_change(xy, present, half=CHANGE_HALF, fps=FPS):
    """How much the ball's velocity changes at each frame, in m/s; NaN where it cannot be measured.

    dv(t) = | v_after - v_before |, with v_before = (p[t] - p[t - half]) / (half / fps) and
    v_after = (p[t + half] - p[t]) / (half / fps): the average velocity over the 0.2 s before
    the frame and over the 0.2 s after it (half = 5 at 25 fps). Speeding up, slowing down and
    turning all count. dv is NaN unless the ball is present on every frame from t - half to
    t + half and no single step in between is longer than MAX_STEP (1.6 m, 40 m/s: a tracking
    jump, not a ball), so a touch can never hide in a missing frame. xy is (n, 2) metres,
    present (n,) bool; frames that are not present are ignored whatever xy says.

        # 1 m/s along x, then the ball stops at frame 2; half = 1, fps = 1
        xy = [[0, 0], [1, 0], [2, 0], [2, 0], [2, 0]]
        velocity_change(xy, [True] * 5, half=1, fps=1)        # -> [nan, 0, 1, 0, nan]
        velocity_change(xy, [True, True, False, True, True], half=1, fps=1)
        # -> [nan, nan, nan, nan, nan]   frame 2 is missing: nothing around it is measurable
    """
    xy = np.asarray(xy, dtype=np.float64).copy()
    xy[~np.asarray(present, dtype=bool)] = np.nan
    n = len(xy)
    step = np.full(n, np.nan)
    step[1:] = np.hypot(*np.diff(xy, axis=0).T)
    bad = ~(step <= MAX_STEP)               # step[i] is frame i-1 -> i; NaN (missing) is bad
    bad[0] = True
    bad_before = np.cumsum(bad)             # bad steps among frames 0..i
    out = np.full(n, np.nan)
    if n <= 2 * half:
        return out
    t = np.arange(half, n - half)
    t = t[bad_before[t + half] - bad_before[t - half] == 0]      # steps t-half+1 .. t+half
    v_before = (xy[t] - xy[t - half]) / (half / fps)
    v_after = (xy[t + half] - xy[t]) / (half / fps)
    out[t] = np.hypot(*(v_after - v_before).T)
    return out


def no_touch_frames(match, window=NO_TOUCH_WINDOW, limit=NO_TOUCH_LIMIT):
    """(n,) bool: frames with clear evidence that nobody touched the ball around them.

    Frame t is True when velocity_change() of the tracking ball (match.ball_xy where
    match.ball_present) exists on EVERY frame from t - window to t + window and stays below
    limit there: a ball tracked throughout, without a jump, moving at a nearly constant velocity
    (or lying still) for about a second. Missing ball means no evidence, so False. On a masked
    match (mtah/validation/masking.py) the masked frames are not present and so give no evidence.

        # a ball lying still on frames 0..99, tracked throughout; window 12, limit 1.5
        no_touch_frames(match)[[10, 17, 50, 82, 90]]      # -> [F, T, T, T, F]
        # frame 17 is the first whose window (5..29) has a dv on every frame: dv needs the
        # ball from 5 frames before to 5 frames after, so dv exists on frames 5..94
        # a kick on frame 50 (dv > 0 on frames 46..54, up to 10 m/s) makes frames 34..66 False
    """
    dv = velocity_change(match.ball_xy, match.ball_present)
    quiet = np.isfinite(dv) & (dv < limit)
    n = len(quiet)
    out = np.zeros(n, dtype=bool)
    if n <= 2 * window:
        return out
    loud = np.concatenate(([0], np.cumsum(~quiet)))         # not-quiet frames among 0..i-1
    t = np.arange(window, n - window)
    out[t] = loud[t + window + 1] - loud[t - window] == 0
    return out


def no_touch_costs(match, cost=NO_TOUCH_COST):
    """(n,) float: the extra cost of a switch into H or A into each frame (0 or `cost`).

    `cost` (NO_TOUCH_COST, 4) on the no_touch_frames(), 0 elsewhere. "Into frame t" means frame
    t is the first frame of the new state, as in mtah.temporal.hmm.whistle_switch_costs. The
    third array of the switch costs that mtah.temporal.hmm.viterbi() and posteriors() take:
    there it is added to H <-> A and to leaving D, never to entering D.

        no_touch_costs(match)[[10, 50]]       # the still ball above -> [0.0, 4.0]
    """
    return np.where(no_touch_frames(match), float(cost), 0.0)
