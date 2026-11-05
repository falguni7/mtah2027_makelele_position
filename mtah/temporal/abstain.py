"""Abstain with X on frames where the labels are probably wrong, without changing any possession.

    from mtah.temporal.abstain import abstain
    out = abstain(labels, post, graded)      # labels with some frames set to "X"

How the official scorer treats X (evaluation/possession_metrics.py, SUBMISSION.md):

  * the frame-level metric (macroF1 over H, A, D) counts a predicted X on a graded frame as a
    miss for its true class, and never as a false positive for any class;
  * the possession metrics (spellF1, which ranks, and the transition scores) first fill every X
    with the last label before it in its graded half (fill_abstentions), then score.

So an X placed inside one of our label runs, never on its first frame, is filled back with
exactly the label it replaced: our possessions, changes and dead-ball stretches stay the same
frame for frame, and spellF1 cannot change. Only macroF1 changes.

When an X helps macroF1. Take a frame where we say p. If p is right, an X turns a correct frame
into a miss (class p loses a true positive); if p is wrong, it removes a false positive of p
(the true class missed it either way). For class p with F1 = 2 TP / (2 TP + FP + FN), the
first changes F1 by about -(2 / N)(1 - F1 / 2), the second by +(2 / N)(F1 / 2), where N is
2 TP + FP + FN. With q the chance that p is right on that frame, the X helps on average when

    (1 - q) * F1 / 2  >  q * (1 - F1 / 2),   i.e.   q < F1 / 2.

So the threshold follows from the score itself, with nothing tuned: a class we label at F1 0.8
gets X only where we are right less than 40 % of the time. F1 is unknown without labels;
expected_f1() estimates it from the same posteriors, as the F1 the labels would have if the
posteriors were right.

q comes from the temporal model's forward-backward posteriors
(mtah.temporal.hmm.decode_posteriors): the probability of each state on each frame given all
the evidence before and after it. In this file: expected_f1() and abstain().
"""

import numpy as np

from mtah.utils import runs_of

# Params
STATES = ("H", "A", "D")


def expected_f1(labels, post, graded):
    """{state: F1 the labels would have per class if the posteriors were the truth}, graded only.

    labels are "H" / "A" / "D" per frame, post the (n, 3) posteriors in STATES order, graded a
    bool mask. For class k: expected TP is the sum of post[:, k] where we say k, FP the sum of
    1 - post[:, k] there, FN the sum of post[:, k] where we say something else; F1 = 2 TP /
    (2 TP + FP + FN). An estimate, as good as the posteriors are calibrated.

        labels = ["H", "H", "A"], post = [[0.9, 0.1, 0], [0.6, 0.4, 0], [0.2, 0.8, 0]], all graded
        expected_f1(labels, post, graded)
        # H: TP 1.5, FP 0.5, FN 0.2 -> 3.0 / 3.7 = 0.81; A: TP 0.8, FP 0.2, FN 0.5 -> 0.70;
        # D: nothing predicted or expected -> nan
    """
    labels = np.asarray(labels)
    post = np.asarray(post, dtype=np.float64)
    g = np.asarray(graded, dtype=bool)
    out = {}
    for k, state in enumerate(STATES):
        said = g & (labels == state)
        tp = post[said, k].sum()
        fp = (1 - post[said, k]).sum()
        fn = post[g & (labels != state), k].sum()
        out[state] = float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn > 0 else np.nan
    return out


def abstain(labels, post, graded, f1=None):
    """Labels with "X" on frames where the label is probably wrong; possessions stay identical.

    labels are "H" / "A" / "D" per frame (a method's output), post the (n, 3) posteriors from
    decode_posteriors() for the same evidence, graded the bool mask of graded frames. f1 gives
    the per-class F1 used for the thresholds (default: expected_f1() of these labels). A graded
    frame labelled p becomes "X" when post[frame, p] < f1[p] / 2 AND it is not the first frame of
    its run of p inside its graded half, so that the scorer's fill-forward gives back exactly p
    (see the module docstring). Frames outside graded are never touched. Returns a new array.

        labels = ["H", "H", "H", "A", "A"], all graded, f1 = {"H": 0.8, "A": 0.8, "D": 0.5}
        post[:, own label] =  [0.9, 0.3, 0.35, 0.2, 0.9]      # thresholds 0.4 for H and A
        abstain(labels, post, graded, f1)
        # -> ["H", "X", "X", "A", "A"]   frame 3 stays A: it is where A starts
    """
    labels = np.asarray(labels)
    post = np.asarray(post, dtype=np.float64)
    g = np.asarray(graded, dtype=bool)
    f1 = expected_f1(labels, post, g) if f1 is None else f1
    out = labels.copy()
    for start, stop in runs_of(g):
        window = labels[start:stop]
        for k, state in enumerate(STATES):
            if not np.isfinite(f1.get(state, np.nan)):
                continue
            unsure = (window == state) & (post[start:stop, k] < f1[state] / 2)
            first = np.zeros(len(window), dtype=bool)
            first[0] = True
            first[1:] = window[1:] != window[:-1]
            out[start:stop][unsure & ~first] = "X"
    return out
