"""Control-event rule (roadmap step 2): possession is the team of the last player in control.

    from mtah.possession.control import control_events, possession_from_events
    events = control_events(match)                 # [(start, stop, team)]
    teams = possession_from_events(events, match.graded, dead)

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
    """(ball_xy, present) with single-frame spikes removed (jump in and out above MAX_BALL_SPEED)."""
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
    """Velocity from positions over +/- VELOCITY_HALF frames, NaN at the edges and across gaps."""
    k = VELOCITY_HALF
    out = np.full(positions.shape, np.nan)
    out[k:-k] = (positions[2 * k:] - positions[:-2 * k]) * FPS / (2 * k)
    return out


def control_frames(match):
    """(team in control per frame: 0 home, 1 away, -1 nobody; nearest distance; relative speed)."""
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
    """[(start, stop, team)] runs of one team in control lasting at least MIN_EVENT frames."""
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
    An alive stretch with no event at all gets the team of the previous stretch, else "D".
    """
    n = len(graded)
    dead = np.zeros(n, dtype=bool) if dead is None else np.asarray(dead)
    code = np.full(n, -1, dtype=np.int8)
    for start, _, team in events:
        code[start] = team
    labels = np.full(n, "D", dtype="<U1")
    previous = -1
    for start, stop in runs_of(np.asarray(graded) & ~dead):
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
