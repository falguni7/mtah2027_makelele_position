"""Control-event rule (roadmap step 2): possession is the team of the last player in control.

    from mtah.possession.control import control_events, possession_from_events
    events = control_events(match)                 # [(start, stop, team)]
    teams = possession_from_events(events, match.graded, dead)   # "H" / "A" / "D" per frame

What is in it, in pipeline order: clean_ball() drops one-frame ball spikes, velocity() gives
smoothed velocities, control_frames() says per frame which team (if any) is in control,
control_events() turns that into events, and possession_from_events() into labels. The events
are also the input of the step-3 temporal model (mtah/temporal/hmm.py) and of the step-4
pseudo-labels (shape_model.py).

CHALLENGE.md: possession changes when an opponent gains *controlled* possession; a ball in flight,
a deflection or a loose ball keeps the previous team. So, per frame with a tracked ball:

  * a player is IN CONTROL when the ball is within CONTROL_RADIUS of them, moves with them
    (relative velocity below CONTROL_REL_SPEED), and no opponent is about as close
    (OPPONENT_MARGIN) -- the last condition keeps duels from counting as control;
  * a CONTROL EVENT is at least MIN_EVENT consecutive control frames by one team;
  * possession is the team of the last event, and a change is placed at the START of the new
    team's event, not when it becomes certain (one second late costs ~0.35 spellF1 for that
    possession).

After a dead-ball stretch, play restarts with whoever takes the restart, so each alive stretch is
filled back from its first event instead of carrying the team from before the stoppage.

Thresholds are set from the measured distributions (work log, step 2), not tuned on any labels.
"""

import numpy as np

from mtah.utils import runs_of

# Params
FPS = 25
MAX_BALL_SPEED = 40.0       # m/s between consecutive frames; faster is a misdetection
CONTROL_RADIUS = 1.25       # metres, ball to player centre
CONTROL_REL_SPEED = 3.0     # m/s, ball velocity minus player velocity
OPPONENT_MARGIN = 0.5       # metres: nearest opponent must be this much farther than the player
MIN_EVENT = 5               # frames (0.2 s); shorter is a touch, not control
VELOCITY_HALF = 2           # velocity from positions +/- this many frames


def clean_ball(match):
    """(ball_xy, present) with single-frame spikes removed (jump in and out above MAX_BALL_SPEED).

    A spike is a tracked ball frame that is reached from the previous frame, and left to the next
    one, faster than MAX_BALL_SPEED (40 m/s): a misdetection, not a kick. Those frames get NaN
    positions and present = False. Copies are returned; the match is not changed. Only lone
    frames are caught: a wrong detection lasting two frames or more is kept.

        # x positions in metres, y = 0, 25 fps; frame 2 jumps 4.9 m (122 m/s) and back
        ball_x  = [0.0, 0.1, 5.0, 0.2, 0.3]
        clean_ball(match)                  # -> x [0.0, 0.1, nan, 0.2, 0.3]
                                           #    present [T, T, F, T, T]
    """
    ball = np.array(match.ball_xy, dtype=np.float64)
    present = np.array(match.ball_present)
    step = np.full(len(ball), np.nan)
    step[1:] = np.hypot(ball[1:, 0] - ball[:-1, 0], ball[1:, 1] - ball[:-1, 1]) * FPS
    jump_in = step > MAX_BALL_SPEED
    jump_out = np.zeros_like(jump_in)
    jump_out[:-1] = jump_in[1:]
    spike = present & jump_in & jump_out
    ball[spike] = np.nan
    present[spike] = False
    return ball, present


def velocity(positions):
    """Velocity over +/- VELOCITY_HALF frames; NaN at the edges and where frame i-2 or i+2 is NaN.

    positions has time on the first axis and metres in the last one, e.g. the ball (n, 2) or all
    players (n, slots, 2); the result has the same shape, in m/s. Each frame's velocity is the
    displacement from 2 frames before to 2 frames after, divided by that 0.16 s, which averages
    out frame-to-frame tracking jitter. The first and last 2 frames are NaN, and so is any frame
    whose position 2 frames before or 2 frames after is NaN (a gap); a NaN in between is bridged.

        x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # metres, moving 0.1 m per frame (2.5 m/s)
        velocity(x)                         # -> [nan, nan, 2.5, 2.5, nan, nan]
    """
    k = VELOCITY_HALF
    out = np.full(positions.shape, np.nan)
    out[k:-k] = (positions[2 * k:] - positions[:-2 * k]) * FPS / (2 * k)
    return out


def control_frames(match):
    """Team in control per frame: 0 home, 1 away, -1 nobody.

    Only graded frames with a (cleaned) tracked ball are considered. A frame counts as control
    for the team of the nearest player when all three hold: the ball is within CONTROL_RADIUS
    (1.25 m) of them, ball and player move together (relative speed below CONTROL_REL_SPEED,
    3 m/s), and the nearest opponent is more than OPPONENT_MARGIN (0.5 m) farther away than the
    player. Every other frame is -1, including ones where velocity is unknown (NaN).

        # ball 0.8 m from a home player, moving with them
        # nearest away player at 3.0 m             -> 0
        # nearest away player at 1.0 m (a duel)    -> -1   (1.0 is not above 0.8 + 0.5)
        # ball 2.0 m from the nearest player       -> -1
    """
    ball, present = clean_ball(match)
    players = np.asarray(match.players_xy, dtype=np.float64)
    frames = np.flatnonzero(present & np.asarray(match.graded))

    p = players[frames]
    distance = np.hypot(p[:, :, 0] - ball[frames, None, 0], p[:, :, 1] - ball[frames, None, 1])
    distance[np.isnan(distance)] = np.inf
    nearest = distance.argmin(axis=1)
    rows = np.arange(len(frames))
    nearest_distance = distance[rows, nearest]
    team = match.team_of_slot[nearest]
    opponent = np.where(match.team_of_slot[None, :] != team[:, None], distance, np.inf).min(axis=1)

    ball_velocity = velocity(ball)[frames]
    player_velocity = velocity(players)[frames, nearest]
    relative = np.hypot(*(ball_velocity - player_velocity).T)

    in_control = ((nearest_distance < CONTROL_RADIUS) & (relative < CONTROL_REL_SPEED)
                  & (opponent > nearest_distance + OPPONENT_MARGIN))
    out = np.full(match.n_frames, -1, dtype=np.int8)
    out[frames[in_control]] = team[in_control]
    return out


def control_events(match):
    """[(start, stop, team)] runs of one team in control lasting at least MIN_EVENT frames.

    The frame-level control of control_frames() turned into events: half-open frame ranges
    (start, stop) of consecutive control by one team (0 home, 1 away), at least MIN_EVENT
    (5 frames, 0.2 s) long so touches and flicks do not count. Sorted by start frame. The input
    of possession_from_events(), the temporal model and the shape-model pseudo-labels.

        # control_frames(match) -> [-1, 0, 0, 0, 0, 0, -1, 1, 1, -1]
        control_events(match)                  # -> [(1, 6, 0)]  (the away run is too short)
    """
    team = control_frames(match)
    events = []
    for code in (0, 1):
        events += [(start, stop, code) for start, stop in runs_of(team == code)
                   if stop - start >= MIN_EVENT]
    return sorted(events)


def possession_from_events(events, graded, dead=None):
    """Per-frame "H" / "A" / "D": team of the last control event, changes at event starts.

    Each alive stretch (graded, not dead) is filled from its own events only: before its first
    event it takes that event's team (the restart), after the last one it keeps the last team.
    An alive stretch with no event at all gets the team of the previous stretch in the same
    graded half, else "D"; nothing carries across half-time. events are (start, stop, team) as
    from control_events(); an event counts from its first alive frame, so one that starts on a
    dead frame and runs into play (a restart taken while the stoppage is still called) changes
    possession where play resumes, and one entirely on dead or ungraded frames is ignored. graded
    and dead are bool masks per frame (dead None: no stoppages). Dead and ungraded frames are
    "D".

        events = [(1, 3, 1), (7, 9, 0)]        # away in control at 1-2, home at 7-8
        graded = [T] * 10
        dead   = [F, F, F, F, F, T, T, F, F, F]
        possession_from_events(events, graded, dead)
        # -> ["A", "A", "A", "A", "A", "D", "D", "H", "H", "H"]
        possession_from_events(events, graded)
        # -> ["A", "A", "A", "A", "A", "A", "A", "H", "H", "H"]
    """
    n = len(graded)
    graded = np.asarray(graded, dtype=bool)
    dead = np.zeros(n, dtype=bool) if dead is None else np.asarray(dead, dtype=bool)
    alive = graded & ~dead
    code = np.full(n, -1, dtype=np.int8)
    for start, stop, team in events:
        inside = np.flatnonzero(alive[start:stop])
        if len(inside):
            code[start + inside[0]] = team
    labels = np.full(n, "D", dtype="<U1")
    for half_start, half_stop in runs_of(graded):
        previous = -1
        for a, b in runs_of(alive[half_start:half_stop]):
            start, stop = half_start + a, half_start + b
            window = code[start:stop]
            known = np.flatnonzero(window >= 0)
            if not len(known):
                if previous >= 0:
                    labels[start:stop] = "HA"[previous]
                continue
            carry = np.maximum.accumulate(np.where(window >= 0, np.arange(len(window)), -1))
            carry[carry < 0] = known[0]
            filled = window[carry]
            labels[start:stop] = np.where(filled == 0, "H", "A")
            previous = int(filled[-1])
    return labels
