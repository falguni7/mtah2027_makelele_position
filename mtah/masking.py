"""The masking test: hide the ball where it IS tracked, and see what a method does without it.

41% of graded frames have no ball, in 150 gaps. Where the ball is missing in the real data there
is no way to tell whether a method is right. So we make gaps where we do know something:

  1. pick blocks where the ball is tracked, with lengths like the real gaps (or fixed lengths);
  2. hide the ball there (mask), run the method, and
  3. score the masked output against a reference on windows around the blocks.

The reference is the method's own unmasked output by default -- "how much does losing the ball
change the answer" -- or any fixed labelling, e.g. our best ball-based method. Either way it is
not ground truth: this measures robustness to gaps, not accuracy. It is the tool for comparing
gap-filling methods without touching the public window.

Scores use the official scorer (evaluation/possession_metrics.py): the reference is set to X
outside the windows, so every window is scored as its own run, like a half.
"""

import copy
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
MIN_PRESENT = 0.9        # a block is only placed where the ball is tracked on this share of it
MARGIN = 250             # frames scored either side of a block (10 s): changes leak past its edges
MAX_ATTEMPTS = 2000      # random placements tried per block before giving up


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def real_gap_lengths(match):
    """Lengths in frames of the real ball gaps on graded frames, each half separately."""
    lengths = []
    for start, stop in runs_of(match.graded):
        lengths += [b - a for a, b in runs_of(~match.ball_present[start:stop])]
    return np.array(lengths)


def sample_blocks(match, lengths, rng, min_present=MIN_PRESENT, margin=MARGIN):
    """Non-overlapping blocks [(start, stop)], one per requested length where one fits.

    A block lies inside one graded half, at least `margin` frames from its ends, with the ball
    tracked on at least `min_present` of its frames, and its scoring window (block +/- margin)
    does not overlap another block's window. Longest first, since they are the hardest to place.
    """
    present = np.concatenate(([0], np.cumsum(match.ball_present, dtype=np.int64)))
    halves = runs_of(match.graded)
    taken = np.zeros(match.n_frames, dtype=bool)
    blocks = []
    for length in sorted((int(x) for x in lengths), reverse=True):
        for _ in range(MAX_ATTEMPTS):
            half_start, half_stop = halves[rng.integers(len(halves))]
            low, high = half_start + margin, half_stop - margin - length
            if high <= low:
                break
            start = int(rng.integers(low, high))
            stop = start + length
            if taken[start - margin:stop + margin].any():
                continue
            if present[stop] - present[start] < min_present * length:
                continue
            taken[start - margin:stop + margin] = True
            blocks.append((start, stop))
            break
    return sorted(blocks)


def apply_mask(match, blocks):
    """A copy of match with the ball removed inside the blocks. The original is untouched."""
    masked = copy.copy(match)
    ball_present = np.array(match.ball_present)
    ball_xy = np.array(match.ball_xy)
    for start, stop in blocks:
        ball_present[start:stop] = False
        ball_xy[start:stop] = np.nan
    masked.ball_present = ball_present
    masked.ball_xy = ball_xy
    return masked


def score(reference, prediction, blocks, margin=MARGIN):
    """Official metrics of prediction vs reference, on the windows around the blocks only.

    Also returns frame agreement on the masked frames themselves, the most direct number.
    """
    reference_codes = pm.label_codes(np.asarray(reference))
    prediction_codes = pm.label_codes(np.asarray(prediction))
    window = np.zeros(len(reference_codes), dtype=bool)
    inside = np.zeros(len(reference_codes), dtype=bool)
    for start, stop in blocks:
        window[max(0, start - margin):stop + margin] = True
        inside[start:stop] = True
    # Reference X (ungraded) stays X; everything outside the windows becomes X.
    gt = np.where(window, reference_codes, pm.X)
    result = pm.evaluate(gt, prediction_codes)
    scored = inside & (reference_codes != pm.X)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "agreement": float((reference_codes[scored] == prediction_codes[scored]).mean()),
        "n_blocks": len(blocks),
        "masked_frames": int(inside.sum()),
    }
