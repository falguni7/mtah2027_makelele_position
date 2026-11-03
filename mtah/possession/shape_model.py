"""Team in possession from the players alone (roadmap step 4): pseudo-label self-training.

    from mtah.possession.shape_model import pseudo_labels, out_of_fold
    target = pseudo_labels(match, events)                  # 1 home, 0 away, -1 unknown
    p_home, report = out_of_fold(shape_table, target, match.graded)

Labels come from the step-2 control rule, never from a human: frames inside a control event, and
the frames of a pass between two events of the same team (ball tracked, under PASS_MAX). They
only exist where the ball is tracked, but the features (mtah/features/team_shape.py) do not use
the ball, so the model also answers inside ball gaps.

Predictions are OUT-OF-FOLD over contiguous time blocks: every frame is predicted by a model
that never saw its block. That is what makes the held-out accuracy honest (neighbouring frames
are near-copies, so random splits would leak), and it keeps the confident frames' predictions
from being in-sample when they are used downstream.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.features.team_shape import COLUMNS

# Params
FPS = 25
PASS_MAX = 2 * FPS          # a same-team gap between events shorter than this is a pass
N_BLOCKS = 10               # contiguous folds over the graded frames
TRAIN_STRIDE = 5
MODEL = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             random_state=0)


def pseudo_labels(match, events, ball_present=None):
    """int8 per frame: 1 home in possession, 0 away, -1 unknown."""
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
    """Block index per frame (-1 off the graded frames): N contiguous chunks of graded frames."""
    index = np.full(len(graded), -1, dtype=np.int16)
    frames = np.flatnonzero(graded)
    index[frames] = np.arange(len(frames)) * n_blocks // len(frames)
    return index


def out_of_fold(table, target, graded, columns=COLUMNS):
    """(P(home in possession) per frame, per-block report of held-out accuracy)."""
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
    """Frames at least min_distance from any control event, graded: where only the shape helps."""
    near = np.zeros(match.n_frames, dtype=bool)
    for start, stop, _ in events:
        near[max(0, start - min_distance):stop + min_distance] = True
    return np.asarray(match.graded) & ~near

