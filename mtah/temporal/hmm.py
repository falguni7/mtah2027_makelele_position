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
               confident step-1 label are pinned. DEAD: goal to kick-off (GOALS tags), the second
               half before its kick-off, ball tracked out of the pitch, the first seconds
               after a set-piece tag start. ALIVE: ball tracked and moving inside the pitch,
               short open-play ball gaps, sustained running by the players (ball tracked or
               not). The weight is below 1 because neighbouring frames are near-duplicates: ~25
               correlated frames per second must not count as 25 independent observations.
  control      during a step-2 control event of team T, the other team and D pay EVENT_COST
               per frame.
  carry        between events, any team other than the last event's pays CARRY_COST per frame,
               so a change lands at the start of the new team's event, not earlier. After a
               confident dead stretch, the "last" team is the next event's (the restart taker).

  shape        (step 4, optional) far from any control event, TEAM_WEIGHT * log P(team) from
               the ball-free team-shape model; inside long ball gaps the only team evidence besides
               the carry bias (CARRY_COST per frame still applies on shape frames).

Transitions pay a fixed cost: TEAM_SWITCH for H <-> A, DEAD_SWITCH for entering or leaving D.
The switch cost does not depend on the stoppage length: entering and leaving D costs
2 * DEAD_SWITCH = 8 in total (6 more than a direct H -> A switch when the teams differ). At
p_dead = 0.98 that is about 21 frames (0.8 s) of evidence, so it acts as a soft minimum duration.

Whistles (optional, the referee's whistles from mtah/audio/whistle.py): a whistle says a
stoppage boundary is close, but not which one -- the referee whistles to stop play (a foul,
offside, the end of a half) and to restart it (kick-offs, some free kicks). So near a whistle
the switch into or out of D is made cheaper (WHISTLE_SWITCH instead of DEAD_SWITCH), each in
its own window, because the whistle sits at a different place relative to each boundary:

  entering D   from WHISTLE_ENTER[0] to WHISTLE_ENTER[1] seconds around the whistle (-1 to +1 s):
               the whistle follows the foul or the ball going out within about a second;
  leaving D    from WHISTLE_LEAVE[0] to WHISTLE_LEAVE[1] seconds after it (0 to +3 s): the
               restart follows its whistle within a few seconds.

This never forces a stoppage: it only lowers the price of a boundary where the other evidence
already asks for one, so a stoppage can start or end at the whistle instead of where p_dead
happens to cross over. whistle_switch_costs() builds the per-frame costs.

Values are chosen from these arguments, not tuned on labels.

What is in it: emissions() builds the (frames, 3) evidence table above, transition_matrix() the
switch costs, whistle_switch_costs() the cheaper D switches near whistles, viterbi() finds the
best path through one run of frames, and decode() runs it per graded half and returns the
labels. posteriors() and decode_posteriors() use the same evidence and switch costs for a
different question: not the single best path, but for every frame how likely each of H, A and
D is given everything before and after it (forward-backward), which says how sure the labels
are (used for abstaining with X, mtah/temporal/abstain.py). With the step-4 shape model, and with whistles:

    labels = decode(match, events, p_dead, dead_labels, p_home, shape_frames)
    switch = whistle_switch_costs(match.n_frames, whistle_events)
    labels = decode(match, events, p_dead, dead_labels, p_home, shape_frames, switch)
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
WHISTLE_SWITCH = 1.0       # cost of entering or leaving D near a whistle
WHISTLE_ENTER = (-1.0, 1.0)    # seconds around a whistle where entering D is cheaper
WHISTLE_LEAVE = (0.0, 3.0)     # seconds around a whistle where leaving D is cheaper


def emissions(n, events, p_dead, dead_labels, graded, p_home=None, shape_frames=None):
    """(n, 3) log scores for H, A, D. p_home / shape_frames: step-4 evidence and where to use it.

    One row per frame, one column per state (H = 0, A = 1, D = 2); higher is more likely, and
    only differences within a row matter. Inputs, all per frame over the n frames: p_dead the
    step-1 probability of a dead ball, dead_labels its confident labels (DEAD, ALIVE or unknown),
    graded a bool mask, p_home P(home in possession) from the shape model and shape_frames a bool
    mask of where to trust it; events are (start, stop, team) control events. The terms are the
    ones listed in the module docstring: dead-ball evidence for every frame, PINNED (50) against
    states a confident label rules out, EVENT_COST (5) per frame against the other team and D
    during an event, CARRY_COST (0.02) per frame against changing team between events, and the
    shape evidence where shape_frames is set and p_home is known.

        # 4 frames, p_dead 0.5 everywhere, no confident labels, all graded,
        # one home event at frames 1-2: events = [(1, 3, 0)]
        emissions(4, events, p_dead, dead_labels, graded)
        #           H       A       D
        # -> [[-0.07,  -0.09,  -0.07],    # before the event: carry H, so A pays 0.02
        #     [-0.07,  -5.07,  -5.07],    # in the event: A and D pay 5
        #     [-0.07,  -5.07,  -5.07],
        #     [-0.07,  -0.09,  -0.07]]    # after the event: carry H
    """
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
        # An event counts from its first frame that is not pinned dead: a restart taken while the
        # stoppage is still called changes the carried team where play resumes.
        inside = np.flatnonzero(~pinned_dead[start:stop])
        if len(inside):
            event_start[start + inside[0]] = team
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
    """(3, 3) log transition scores, from row state to column state.

    Staying in a state is free; H <-> A costs TEAM_SWITCH and entering or leaving D costs
    DEAD_SWITCH, as negative log scores. Rows and columns are H, A, D.

        transition_matrix()
        # -> [[ 0, -2, -4],      # from H
        #     [-2,  0, -4],      # from A
        #     [-4, -4,  0]]      # from D
    """
    cost = np.zeros((3, 3))
    cost[H, A] = cost[A, H] = TEAM_SWITCH
    cost[H, D] = cost[A, D] = cost[D, H] = cost[D, A] = DEAD_SWITCH
    return -cost


def whistle_switch_costs(n_frames, whistle_events):
    """(enter, leave): per-frame cost of switching into and out of D, lower near whistles.

    whistle_events are (start_s, stop_s, ...) tuples on the video timeline, as from
    mtah.audio.whistle.detect(). Both arrays are DEAD_SWITCH (4) everywhere, except enter is
    WHISTLE_SWITCH (1) on frames from WHISTLE_ENTER[0] to WHISTLE_ENTER[1] seconds (-1 to +1 s)
    around each whistle's start, and leave is WHISTLE_SWITCH on frames from WHISTLE_LEAVE[0]
    to WHISTLE_LEAVE[1] seconds (0 to +3 s) after it. "Frame t" means the switch happens
    between frame t - 1 and frame t, i.e. frame t is the first one in the new state.

        whistle_events = [(10.0, 10.4, 3800.0, 25.0)]     # a whistle at 10 s (frame 250)
        enter, leave = whistle_switch_costs(500, whistle_events)
        enter[224:227], enter[250], enter[275]  # -> [4, 1, 1], 1, 4     frames 225-274 cheap
        leave[249], leave[250], leave[325]      # -> 4, 1, 4             frames 250-324 cheap
    """
    enter = np.full(n_frames, DEAD_SWITCH)
    leave = np.full(n_frames, DEAD_SWITCH)
    for start_s, *_ in whistle_events:
        for out, (before, after) in ((enter, WHISTLE_ENTER), (leave, WHISTLE_LEAVE)):
            first = max(0, int(round((start_s + before) * FPS)))
            last = min(n_frames, int(round((start_s + after) * FPS)))
            out[first:last] = np.minimum(out[first:last], WHISTLE_SWITCH)
    return enter, leave


def viterbi(score, transition, switch=None):
    """Most likely state sequence for one run of frames.

    score is (frames, 3) log evidence per state as from emissions(), transition the (3, 3) log
    transition scores; returns the int8 states (0 H, 1 A, 2 D) of the path with the highest total
    of evidence plus transitions. There is no start cost: any first state is allowed. On an exact
    tie the lower state (H, then A, then D) wins. A change only happens when the evidence for it
    outweighs the switch cost, so a one-frame blip is smoothed away but a sustained change is not.
    switch, if given, is (enter, leave) from whistle_switch_costs() for these frames: the cost
    of entering D, and of leaving it, into each frame, replacing the fixed DEAD_SWITCH of
    transition (H <-> A keeps its cost).

        transition = transition_matrix()       # H <-> A costs 2
        score = [[0, -3, -5],                  # frame 0 says H
                 [-1, 0, -5],                  # frames 1-3 lean A
                 [-1, 0, -5],
                 [-1, 0, -5]]
        viterbi(score, transition)             # -> [0, 1, 1, 1]  (H A A A, total -2)

        score = [[0, -3, -5], [-1, 0, -5], [0, -3, -5], [0, -3, -5]]   # one A blip
        viterbi(score, transition)             # -> [0, 0, 0, 0]  (H H H H, total -1)

        # D evidence on frames 1-2 worth 3 against H: not enough to pay 4 + 4 for a stoppage,
        # but enough when a whistle makes entering (and leaving) D cost 1 there
        score = [[0, -9, -9], [-3, -9, 0], [-3, -9, 0], [0, -9, -9]]
        viterbi(score, transition)                         # -> [0, 0, 0, 0]
        viterbi(score, transition, ([4, 1, 4, 4], [4, 4, 4, 1]))   # -> [0, 2, 2, 0]
    """
    n = len(score)
    back = np.zeros((n, 3), dtype=np.int8)
    best = list(np.asarray(score[0], dtype=np.float64))
    trans = transition.tolist()
    if switch is not None:
        enter, leave = (np.asarray(costs, dtype=np.float64).tolist() for costs in switch)
    for t in range(1, n):
        b0, b1, b2 = best
        row = score[t]
        if switch is not None:
            trans[0][2] = trans[1][2] = -enter[t]
            trans[2][0] = trans[2][1] = -leave[t]
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


def decode(match, events, p_dead, dead_labels, p_home=None, shape_frames=None, switch=None):
    """Per-frame "H" / "A" / "D" labels; each graded half is decoded on its own.

    The step-3 entry point: builds the evidence with emissions() from the step-2 control events,
    the step-1 dead-ball probability p_dead and confident dead_labels (and optionally the step-4
    p_home with the shape_frames where it applies, and the whistle switch costs, (enter, leave)
    from whistle_switch_costs()), then runs viterbi() separately on every graded run of the
    match, so nothing carries over half-time. Frames outside graded are "D".

        # the 4-frame example of emissions(): one home event at frames 1-2, p_dead 0.5
        decode(match, [(1, 3, 0)], p_dead, dead_labels)   # -> ["H", "H", "H", "H"]
    """
    graded = np.asarray(match.graded)
    score = emissions(match.n_frames, events, p_dead, dead_labels, graded, p_home, shape_frames)
    transition = transition_matrix()
    labels = np.full(match.n_frames, "D", dtype="<U1")
    for start, stop in runs_of(graded):
        costs = None if switch is None else (switch[0][start:stop], switch[1][start:stop])
        path = viterbi(score[start:stop], transition, costs)
        labels[start:stop] = np.array(["H", "A", "D"])[path]
    return labels


def _logsumexp(a, axis):
    """log(sum(exp(a))) along axis, computed without overflow (the largest term factored out).

    a holds log values (log weights of paths); the result is the log of their total, with that
    axis removed. Adding up exp(a) directly would overflow or underflow for the large negative
    scores of long runs, so the largest value is taken out first: log(sum(exp(a - top))) + top.
    Used by posteriors() to add up the weights of all paths into or out of each state.

        _logsumexp(np.log([[1.0, 3.0]]), axis=1)     # -> [log(4)] = [1.386]
    """
    top = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(a - top), axis=axis))


def posteriors(score, transition, switch=None):
    """(frames, 3) probability of H, A and D on every frame, given all the frames of the run.

    The forward-backward counterpart of viterbi(), on the same inputs: score is (frames, 3) log
    evidence from emissions(), transition the (3, 3) log transition scores, switch optional
    (enter, leave) costs from whistle_switch_costs() for these frames, used exactly as in
    viterbi(). The scores are treated as log weights of every possible path; a frame's
    probability of state s is the total weight of the paths that are in s there, divided by the
    total weight of all paths. Each row sums to 1. Where one path is much better than all
    others the result is close to 1 for its state; where the evidence is balanced it is spread.

        transition = transition_matrix()
        score = [[0, -9, -9], [-1, -1, -9], [0, -9, -9]]   # frame 1 is undecided between H and A
        posteriors(score, transition)[1]
        # -> ~[0.98, 0.02, 0.00]   A on frame 1 would cost two team switches (2 + 2), so
        #    that path weighs e^-4 = 0.018 times the all-H path
    """
    score = np.asarray(score, dtype=np.float64)
    n = len(score)
    if switch is not None:
        enter, leave = (np.asarray(c, dtype=np.float64) for c in switch)

    def trans_at(t):
        """(3, 3) log transition scores for the step from frame t - 1 to frame t.

        Without switch costs this is always `transition`. With them, entering D costs enter[t]
        and leaving D costs leave[t] on this step, exactly as in viterbi(); H <-> A keeps its
        fixed cost.

            # enter = [4, 1, 4], leave = [4, 4, 4]: entering D into frame 1 is cheap
            trans_at(1)[0, 2], trans_at(2)[0, 2]      # -> -1, -4
        """
        if switch is None:
            return transition
        m = transition.copy()
        m[0, 2] = m[1, 2] = -enter[t]
        m[2, 0] = m[2, 1] = -leave[t]
        return m

    alpha = np.empty((n, 3))
    alpha[0] = score[0]
    for t in range(1, n):
        alpha[t] = score[t] + _logsumexp(alpha[t - 1][:, None] + trans_at(t), axis=0)
    beta = np.zeros((n, 3))
    for t in range(n - 1, 0, -1):
        beta[t - 1] = _logsumexp(trans_at(t) + (score[t] + beta[t])[None, :], axis=1)
    log_post = alpha + beta
    log_post -= _logsumexp(log_post, axis=1)[:, None]
    return np.exp(log_post)


def decode_posteriors(match, events, p_dead, dead_labels, p_home=None, shape_frames=None,
                      switch=None):
    """(n_frames, 3) probabilities of H, A, D per frame, from the same evidence as decode().

    Same inputs as decode(); each graded half is computed on its own with posteriors(), so
    nothing carries over half-time. Frames outside the graded halves are NaN. Column 0 is H,
    1 A, 2 D, the order of the states.

        post = decode_posteriors(match, events, p_dead, dead_labels, p_home, frames, switch)
        post[50000]          # -> e.g. [0.97, 0.02, 0.01]   sure it is home
        post[60000]          # -> e.g. [0.45, 0.40, 0.15]   unsure
    """
    graded = np.asarray(match.graded)
    score = emissions(match.n_frames, events, p_dead, dead_labels, graded, p_home, shape_frames)
    transition = transition_matrix()
    out = np.full((match.n_frames, 3), np.nan)
    for start, stop in runs_of(graded):
        costs = None if switch is None else (switch[0][start:stop], switch[1][start:stop])
        out[start:stop] = posteriors(score[start:stop], transition, costs)
    return out
