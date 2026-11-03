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

The functions, in the order a test uses them:

    real_gap_lengths(match)                 lengths of the real ball gaps, to copy
    sample_blocks(match, lengths, rng)      where to hide the ball
    apply_mask(match, blocks)               a copy of the match with the ball hidden there
    score(reference, prediction, blocks)    how far the masked output moved from the reference

    rng = np.random.default_rng(0)
    blocks = sample_blocks(match, real_gap_lengths(match), rng)
    reference = method(match)
    masked = method(apply_mask(match, blocks))
    score(reference, masked, blocks)        # -> {spell_f1, macro_f1, agreement, n_blocks, ...}
"""

import copy

import numpy as np

import possession_metrics as pm  # evaluation/, on the path via mtah/__init__.py
from mtah.utils import runs_of

# Params
MIN_PRESENT = 0.9        # a block is only placed where the ball is tracked on this share of it
MARGIN = 250             # frames scored either side of a block (10 s): changes leak past its edges
MAX_ATTEMPTS = 2000      # random placements tried per block before giving up


def real_gap_lengths(match):
    """Lengths in frames of the real ball gaps on graded frames, each half separately.

    One entry per stretch of graded frames without a tracked ball. A gap is cut at the end of
    a half, so one that runs across halftime counts as two. The masking test feeds these to
    sample_blocks() so the fake gaps have the same length distribution as the real ones.

        graded       = [F, T, T, T, T, T, F]     # frames 0..6, one half on 1..5
        ball_present = [F, T, F, F, T, F, F]
        real_gap_lengths(match)                  # -> array([2, 1])  frames 2..3 and frame 5
    """
    lengths = []
    for start, stop in runs_of(match.graded):
        lengths += [b - a for a, b in runs_of(~match.ball_present[start:stop])]
    return np.array(lengths)


def sample_blocks(match, lengths, rng, min_present=MIN_PRESENT, margin=MARGIN):
    """Non-overlapping blocks [(start, stop)], one per requested length where one fits.

    A block lies inside one graded half, at least `margin` frames from its ends, with the ball
    tracked on at least `min_present` of its frames, and its scoring window (block +/- margin)
    does not overlap another block's window. Longest first, since they are the hardest to place.

    rng is a numpy Generator (np.random.default_rng), so a seed gives the same blocks every
    time. Each block is a half-open (start, stop) frame range, returned sorted by start. The
    half is drawn among those long enough, in proportion to their free room. A length may still
    get no block: if no half can hold it with the margins, or after MAX_ATTEMPTS placements that
    overlap other blocks or lack the ball, it is skipped, so fewer blocks than lengths can come
    back (the masked_frames column shows how much was masked).

        lengths = [50, 200]                      # e.g. from real_gap_lengths()
        sample_blocks(match, lengths, np.random.default_rng(0), margin=250)
        # -> e.g. [(31020, 31070), (64480, 64680)]   200 placed first; blocks >= 2 * margin apart
    """
    present = np.concatenate(([0], np.cumsum(match.ball_present, dtype=np.int64)))
    halves = runs_of(match.graded)
    taken = np.zeros(match.n_frames, dtype=bool)
    blocks = []
    for length in sorted((int(x) for x in lengths), reverse=True):
        # Only halves long enough for this block, drawn in proportion to the room they have, so
        # every possible start position is equally likely.
        room = np.array([max(0, stop - start - 2 * margin - length) for start, stop in halves],
                        dtype=np.float64)
        if not room.any():
            continue
        for _ in range(MAX_ATTEMPTS):
            half_start, half_stop = halves[rng.choice(len(halves), p=room / room.sum())]
            low, high = half_start + margin, half_stop - margin - length
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
    """A copy of match with the ball removed inside the blocks. The original is untouched.

    Inside every (start, stop) block ball_present becomes False and ball_xy NaN, exactly as in a
    real gap; every other attribute is shared with the original (a shallow copy), and the two
    ball arrays are fresh writable copies, so the cached memory maps are never modified. The
    result can be passed to any method in place of the match.

        match.ball_present[:6]                  # -> [T, T, T, T, T, T]
        masked = apply_mask(match, [(2, 4)])
        masked.ball_present[:6]                 # -> [T, T, F, F, T, T]
        masked.ball_xy[2]                       # -> [NaN, NaN]
        match.ball_present[2]                   # -> True   (unchanged)
    """
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

    reference and prediction are full-length label arrays ("H", "A", "D", "X"); blocks are the
    (start, stop) ranges that were masked. Each window is the block plus `margin` frames either
    side, because hiding the ball changes the answer a little before and after the block too.
    Returns a dict:

        spell_f1        the scorer's spell_f1_mean (the ranking metric) on the windows
        macro_f1        the scorer's per-frame macro F1 over H, A, D on the windows
        agreement       share of masked frames (reference not X) where the two labels are equal
        n_blocks        number of blocks
        masked_frames   number of frames inside the blocks

        reference  = [H, H, H, H, A, A, A, A]   # frames 0..7
        prediction = [H, H, H, H, H, H, A, A]
        score(reference, prediction, [(4, 6)], margin=1)
        # window = frames 3..6, masked = frames 4..5, both wrong -> agreement 0.0
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
