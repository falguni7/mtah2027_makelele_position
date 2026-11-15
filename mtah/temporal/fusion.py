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

Label files can be combined too: bridge_dead() takes a base method's labels and adds another
method's D only where it fills a whole alive stretch between two of the base's own stoppages
(hmm_v19_bridge: hmm_v11_mix plus hmm_v18_doubtB's bridges), or, with sides="any", wherever it
touches one of them (hmm_v20_adjacent).

In this file: logit(), average_logodds(), calibrate() and bridge_dead().
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mtah.dead_ball.detector import ALIVE, DEAD
from mtah.utils import runs_of

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


def bridge_dead(base, other, graded, sides="both"):
    """base's labels, with other's D on alive stretches of base next to base's own D.

    base and other are (n,) label arrays ("H", "A", "D", "X") of the same match, graded the
    bool mask of graded frames. A conversion is a maximal run of graded frames where base says
    H or A and other says D. With sides="both" (hmm_v19_bridge) it is taken over (set to D) only
    when base is D on the graded frame just before it and on the graded frame just after it:
    other then calls the whole alive stretch between two of base's stoppages dead, so the two
    stoppages become one; conversions that touch a stoppage on one side only, or none, are left
    as base has them. With sides="any" (hmm_v20_adjacent) a conversion is taken over when base
    is D on at least one of the two sides: base's stoppages are also made to start earlier or
    end later. Conversions touching no stoppage are always left. Returns a new array; base and
    other are not changed.

        base  = ["D", "H", "H", "D", "A", "A", "H", "A"]
        other = ["D", "D", "D", "D", "D", "A", "D", "D"]
        bridge_dead(base, other, [True] * 8)
        # -> ["D", "D", "D", "D", "A", "A", "H", "A"]
        # frames 1-2 lie between base's D at 0 and 3: bridged. Frame 4 touches D only before,
        # frames 6-7 none: kept.
        bridge_dead(base, other, [True] * 8, sides="any")
        # -> ["D", "D", "D", "D", "D", "A", "H", "A"]     frame 4 now too, frames 6-7 still kept
    """
    if sides not in ("both", "any"):
        raise ValueError("sides must be 'both' or 'any'")
    base = np.asarray(base)
    other = np.asarray(other)
    graded = np.asarray(graded, dtype=bool)
    n = len(base)
    out = base.copy()
    conv = graded & np.isin(base, ["H", "A"]) & (other == "D")
    for start, stop in runs_of(conv):
        before = start > 0 and graded[start - 1] and base[start - 1] == "D"
        after = stop < n and graded[stop] and base[stop] == "D"
        if (before and after) if sides == "both" else (before or after):
            out[start:stop] = "D"
    return out
