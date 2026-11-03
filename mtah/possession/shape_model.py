"""Team in possession from the players alone (roadmap step 4): pseudo-label self-training.

    from mtah.possession.shape_model import pseudo_labels, out_of_fold
    target = pseudo_labels(match, events)                  # 1 home, 0 away, -1 unknown
    p_home, report = out_of_fold(shape_table, target, match.graded)

Labels come from the step-2 control rule, never from a human: frames inside a control event, and
the frames of a pass between two events of the same team (ball tracked, at most PASS_MAX
frames). They only exist where the ball is tracked, but the features
(mtah/features/team_shape.py) do not use the ball, so the model also answers inside ball gaps.

Predictions are OUT-OF-FOLD over contiguous time blocks: every frame is predicted by a model
that never saw its block. That is what makes the held-out accuracy honest (neighbouring frames
are near-copies, so random splits would leak), and it keeps the confident frames' predictions
from being in-sample when they are used downstream.

What is in it: pseudo_labels() builds the training target, blocks() the time folds, out_of_fold()
trains one gradient-boosting model per fold and predicts P(home) for every frame, and
gap_frames() picks the frames far from any control event, where the step-3 temporal model
(mtah/temporal/hmm.py) uses this P(home) as its only team evidence besides the small carry bias
of the temporal model.

    p_home, report = out_of_fold(shape_table, target, match.graded)
    where = gap_frames(match, events)                      # bool per frame
    labels = decode(match, events, p_dead, dead_labels, p_home, where)   # mtah.temporal.hmm
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.features.team_shape import COLUMNS

# Params
FPS = 25
PASS_MAX = 2 * FPS          # a same-team gap between events of at most this is a pass
N_BLOCKS = 10               # contiguous folds over the graded frames
TRAIN_STRIDE = 5
MODEL = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             random_state=0)


def pseudo_labels(match, events, ball_present=None):
    """int8 per frame: 1 home in possession, 0 away, -1 unknown.

    The training target of the shape model, made from step-2 control events (start, stop, team
    with team 0 home, 1 away) instead of human labels. A frame is labelled when it is inside an
    event, or in the gap between two consecutive events of the same team that is at most
    PASS_MAX (50 frames, 2 s) long with the ball tracked throughout (a pass). Everything else,
    and every frame outside match.graded, is -1. ball_present overrides match.ball_present, e.g.
    to use a cleaned ball. Note the flipped coding: 1 here means home, while events use 0.

        events = [(10, 20, 0), (40, 50, 0), (200, 210, 1)]   # ball tracked throughout
        pseudo_labels(match, events)
        # -> 1 on frames 10-49 (two home events and the pass between them),
        #    0 on frames 200-209, -1 everywhere else
    """
    present = np.asarray(match.ball_present if ball_present is None else ball_present)
    target = np.full(match.n_frames, -1, dtype=np.int8)
    for (start, stop, team), (next_start, _, next_team) in zip(events, events[1:] + [(None,) * 3]):
        target[start:stop] = 1 - team
        if (next_start is not None and next_team == team and next_start - stop <= PASS_MAX
                and present[stop:next_start].all()):
            target[stop:next_start] = 1 - team
    target[~np.asarray(match.graded)] = -1
    return target


def blocks(graded, n_blocks=N_BLOCKS):
    """Block index per frame (-1 off the graded frames): N contiguous chunks of graded frames.

    The folds of out_of_fold(): the graded frames, in time order, cut into n_blocks pieces of
    (nearly) equal size, numbered 0 .. n_blocks - 1. Ungraded frames do not count towards the
    sizes, so a block can span half-time.

        graded = [F, T, T, T, T, F, T]
        blocks(graded, n_blocks=2)             # -> [-1, 0, 0, 0, 1, -1, 1]
    """
    index = np.full(len(graded), -1, dtype=np.int16)
    frames = np.flatnonzero(graded)
    index[frames] = np.arange(len(frames)) * n_blocks // len(frames)
    return index


def out_of_fold(table, target, graded, columns=COLUMNS):
    """(P(home in possession) per frame, per-block report of held-out accuracy).

    table is the per-frame feature table (mtah/features/team_shape.py; one row per frame, the
    given columns are used), target the pseudo_labels() output and graded a bool mask. For each
    time block from blocks() a model is trained on the labelled frames of all other blocks (every
    TRAIN_STRIDE-th one, since neighbours are near-copies) and predicts every frame of its block,
    labelled or not. p_home is float32, NaN off the graded frames. The report has one dict per
    block: its number, how many labelled frames it has, the accuracy of p_home > 0.5 on them and
    the share of them that are home (NaN when the block has no labels).

        p_home, report = out_of_fold(table, target, graded)
        p_home                                 # -> e.g. [nan, 0.91, 0.88, ..., 0.12, nan]
        report[0]                              # -> e.g. {"block": 0, "labelled": 800,
                                               #          "accuracy": 0.8, "home_share": 0.5}
    """
    X = table[columns].to_numpy()
    fold = blocks(np.asarray(graded))
    p_home = np.full(len(X), np.nan, dtype=np.float32)
    report = []
    for k in range(fold.max() + 1):
        train = np.flatnonzero((fold >= 0) & (fold != k) & (target >= 0))[::TRAIN_STRIDE]
        model = HistGradientBoostingClassifier(**MODEL)
        model.fit(X[train], target[train])
        rows = np.flatnonzero(fold == k)
        p_home[rows] = model.predict_proba(X[rows])[:, 1]
        held = rows[target[rows] >= 0]
        report.append({
            "block": k,
            "labelled": len(held),
            "accuracy": float(((p_home[held] > 0.5) == target[held]).mean()) if len(held) else np.nan,
            "home_share": float(target[held].mean()) if len(held) else np.nan,
        })
    return p_home, report


def gap_frames(match, events, min_distance=2 * FPS):
    """Frames more than min_distance from any control event frame, graded: where the shape helps.

    A bool mask per frame: graded frames more than min_distance frames (default 2 s) before the
    start, and at least min_distance frames after the end, of every event. Near an event the
    control rule already knows the team; in these frames, mostly long ball gaps, the shape model
    is the only team evidence, so this is where the temporal model uses it.

        events = [(8, 10, 0)]                  # 20 frames, all graded
        gap_frames(match, events, min_distance=3)
        # -> True on frames 0-4 and 13-19, False on 5-12
    """
    near = np.zeros(match.n_frames, dtype=bool)
    for start, stop, _ in events:
        near[max(0, start - min_distance):stop + min_distance] = True
    return np.asarray(match.graded) & ~near

