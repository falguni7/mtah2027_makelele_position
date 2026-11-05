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

In this file: ball_acceleration() measures how sharply the ball's velocity changes on every
frame; snap_changes() moves the start of every control event that changes the team back to the
touch that began it.
"""

import numpy as np

from mtah.possession.control import clean_ball, velocity

# Params
FPS = 25
ACCEL_HALF = 2              # acceleration from velocities +/- this many frames apart
SNAP_BEFORE = 12            # frames (0.48 s) before an event start searched for the touch
SNAP_AFTER = 2              # frames after it, as the event start itself can be a frame early
MIN_TOUCH_ACCEL = 30.0      # m/s^2: below this the strongest acceleration is not a touch


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
