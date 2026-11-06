"""Combining two probabilities for the same thing into one, for the temporal model (hmm_v9).

    from mtah.temporal.fusion import average_logodds, calibrate
    p_home = average_logodds(p_home_ours, p_home_dfl)          # two shape models -> one
    p_dead = average_logodds(p_dead, calibrate(p_d_dfl, dead_labels))   # option B for D

hmm_v9 has two opinions on two questions: which team has the ball (our pseudo-label shape model
and the DFL-trained one, mtah/external/shape_transfer.py) and whether the ball is dead (the
dead-ball detector and the DFL model's P(D)). The HMM takes one probability per question per
frame, with a hand-set weight. Averaging the two opinions in log-odds keeps that scale, so the
weights need no new tuning, and a frame where one opinion is very sure and the other unsure
ends up leaning to the sure one, while two opposite sure opinions cancel out.

Before averaging, a probability from another domain may be on a different scale: the DFL model
calls more of our frames dead than our detector does (0.48 vs 0.42 of graded frames). calibrate()
maps it onto our match's scale with a one-input logistic fit on the dead-ball detector's
confident labels (frames the dead-ball rules are sure of; mtah/dead_ball/detector.py). Those
are the easier frames, so this fixes the scale, not the hard cases.

In this file: logit(), average_logodds() and calibrate().
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mtah.dead_ball.detector import ALIVE, DEAD

# Params
EPS = 1e-4                  # probabilities are kept inside [EPS, 1 - EPS] before the log
STRIDE = 5                  # every 5th confident frame is used for the fit (neighbours repeat)


def logit(p):
    """Log-odds log(p / (1 - p)) of each probability, clipped to [EPS, 1 - EPS] first.

    A probability of 0.5 is 0, more likely than not is positive; 0 and 1 would be infinite,
    hence the clipping. NaN stays NaN.

        logit([0.5, 0.88, 0.0])        # -> [0.0, 1.99, -9.21]
    """
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def average_logodds(p, q):
    """The probability whose log-odds is the mean of those of p and q, per frame.

    Where one of the two is NaN (no opinion on that frame), the other is returned as it is;
    where both are NaN the result is NaN. Two equal opinions give that opinion back; 0.9 and 0.5
    give 0.75 (log-odds 2.20 and 0 average to 1.10); 0.9 and 0.1 cancel to 0.5.

        average_logodds([0.9, 0.9, 0.9, nan], [0.9, 0.5, 0.1, 0.3])
        # -> [0.9, 0.75, 0.5, 0.3]
    """
    a, b = logit(p), logit(q)
    mean = np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))
    return (1 / (1 + np.exp(-mean))).astype(np.float32)


def calibrate(p, labels, stride=STRIDE):
    """p mapped onto the scale of the confident dead-ball labels: sigmoid(a * logit(p) + b).

    labels is the detector's confident labels per frame (DEAD, ALIVE or UNKNOWN). On every
    `stride`-th frame labelled DEAD or ALIVE where p is defined, a logistic regression of
    "is DEAD" on logit(p) fits a and b; every frame is then mapped with them (NaN stays NaN).
    If p says dead too often on frames the rules call alive, b comes out negative and pulls it
    down; a > 1 makes it sharper, a < 1 softer.

        p      = [0.9, 0.9, 0.6, 0.6, 0.2, ...]   P(D) from the DFL model
        labels = [DEAD, DEAD, ALIVE, ALIVE, ALIVE, ...]
        calibrate(p, labels)           # -> [1.0, 1.0, 0.0, 0.0, 0.0, ...]: here every p above
                                       #    0.6 is DEAD, so the fit is as sharp as it can be;
                                       #    on real frames the classes overlap and it is softer
    """
    x = logit(p)
    labels = np.asarray(labels)
    rows = np.flatnonzero(np.isin(labels, (DEAD, ALIVE)) & np.isfinite(x))[::stride]
    model = LogisticRegression(C=1e6).fit(x[rows, None], labels[rows] == DEAD)
    out = np.full(len(x), np.nan, dtype=np.float32)
    ok = np.isfinite(x)
    out[ok] = model.predict_proba(x[ok, None])[:, 1]
    return out
