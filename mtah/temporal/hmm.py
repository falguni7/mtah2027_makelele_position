"""Temporal model (roadmap step 3): one Viterbi decoding over H / A / D for each half.

    from mtah.temporal.hmm import decode
    labels = decode(match, events, p_dead, dead_labels)

Steps 1 and 2 each made their own hard decisions: the dead-ball detector thresholded its
probability with hysteresis, then the control rule filled possession around the result. Here
both become per-frame evidence (log scores) and one decoding picks the most likely sequence, so
a stoppage boundary can move to where the control evidence says play resumed, and a short dead
call inside a possession has to be strong enough to pay for splitting it.

Evidence per frame (log score added to a state):

  dead ball    DEAD_WEIGHT * log P(state | p_dead) from the step-1 classifier; frames with a
               confident step-1 label (goal, ball out, set-piece start / moving ball) are pinned.
               The weight is below 1 because neighbouring frames are near-duplicates: ~25
               correlated frames per second must not count as 25 independent observations.
  control      during a step-2 control event of team T, the other team and D pay EVENT_COST
               per frame.
  carry        between events, any team other than the last event's pays CARRY_COST per frame,
               so a change lands at the start of the new team's event, not earlier. After a
               confident dead stretch, the "last" team is the next event's (the restart taker).

  shape        (step 4, optional) far from any control event, TEAM_WEIGHT * log P(team) from
               the ball-free team-shape model; the only team evidence inside long ball gaps.

Transitions pay a fixed cost: TEAM_SWITCH for H <-> A, DEAD_SWITCH for entering or leaving D.
A stoppage of n seconds therefore needs about 2 * DEAD_SWITCH of evidence, which acts as a soft
minimum duration.

Values are chosen from these arguments, not tuned on labels.
"""

import numpy as np

from mtah.dead_ball.detector import ALIVE, DEAD
from mtah.utils import runs_of

# Params
FPS = 25
H, A, D = 0, 1, 2
P_CLIP = (0.02, 0.98)
DEAD_WEIGHT = 0.1          # ~ one independent observation every 0.4 s
PINNED = 50.0              # log-score gap that a confident label imposes
EVENT_COST = 5.0
CARRY_COST = 0.02          # per frame, 0.5 per second
TEAM_WEIGHT = 0.05         # shape-model evidence, ~ one observation per 0.8 s
TEAM_SWITCH = 2.0
DEAD_SWITCH = 4.0


def emissions(n, events, p_dead, dead_labels, graded, p_home=None, shape_frames=None):
    """(n, 3) log scores for H, A, D. p_home / shape_frames: step-4 evidence and where to use it."""
    p = np.clip(np.asarray(p_dead, dtype=np.float64), *P_CLIP)
    score = np.zeros((n, 3))
    score[:, D] = DEAD_WEIGHT * np.log(p)
    score[:, H] = score[:, A] = DEAD_WEIGHT * np.log1p(-p)
    pinned_dead = dead_labels == DEAD
    pinned_alive = dead_labels == ALIVE
    score[pinned_dead, H] -= PINNED
    score[pinned_dead, A] -= PINNED
    score[pinned_alive, D] -= PINNED

    # Control events and the carry bias between them.
    in_event = np.full(n, -1, dtype=np.int8)
    event_start = np.full(n, -1, dtype=np.int8)
    for start, stop, team in events:
        in_event[start:stop] = team
        event_start[start] = team
    for team in (H, A):
        rows = in_event == team
        score[rows, 1 - team] -= EVENT_COST
        score[rows, D] -= EVENT_COST

    # Team to carry between events: the last event's, except that across a confident dead
    # stretch it is the next event's. Computed per alive stretch of the confident labels.
    carry = np.full(n, -1, dtype=np.int8)
    for start, stop in runs_of(np.asarray(graded) & ~pinned_dead):
        window = event_start[start:stop]
        known = np.flatnonzero(window >= 0)
        if not len(known):
            continue
        last = np.maximum.accumulate(np.where(window >= 0, np.arange(len(window)), -1))
        last[last < 0] = known[0]
        carry[start:stop] = window[last]
    free = (in_event < 0) & (carry >= 0)
    for team in (H, A):
        score[free & (carry == team), 1 - team] -= CARRY_COST

    if p_home is not None:
        use = np.asarray(shape_frames) & np.isfinite(p_home)
        q = np.clip(np.asarray(p_home, dtype=np.float64)[use], *P_CLIP)
        score[use, H] += TEAM_WEIGHT * np.log(q)
        score[use, A] += TEAM_WEIGHT * np.log1p(-q)
    return score


def transition_matrix():
    """(3, 3) log transition scores, from row state to column state."""
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = TEAM_SWITCH
    cost[H, D] = cost[A, D] = cost[D, H] = cost[D, A] = DEAD_SWITCH
    return -cost


def viterbi(score, transition):
    """Most likely state sequence for one run of frames."""
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    best = score[0].copy()
    trans = transition.tolist()
    for t in range(1, n):
        b0, b1, b2 = best
        row = score[t]
        new = []
        for j in range(3):
            c0 = b0 + trans[0][j]
            c1 = b1 + trans[1][j]
            c2 = b2 + trans[2][j]
            if c0 >= c1 and c0 >= c2:
                back[t, j] = 0
                new.append(c0 + row[j])
            elif c1 >= c2:
                back[t, j] = 1
                new.append(c1 + row[j])
            else:
                back[t, j] = 2
                new.append(c2 + row[j])
        best = new
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(match, events, p_dead, dead_labels, p_home=None, shape_frames=None):
    """Per-frame "H" / "A" / "D" labels; each graded half is decoded on its own."""
    graded = np.asarray(match.graded)
    score = emissions(match.n_frames, events, p_dead, dead_labels, graded, p_home, shape_frames)
    transition = transition_matrix()
    labels = np.full(match.n_frames, "D", dtype="<U1")
    for start, stop in runs_of(graded):
        path = viterbi(score[start:stop], transition)
        labels[start:stop] = np.array(["H", "A", "D"])[path]
    return labels
