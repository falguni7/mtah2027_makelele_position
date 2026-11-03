"""Small helpers on per-frame arrays, shared by every part of the pipeline.

Everything in the pipeline is a per-frame array (index i is video frame i), and the same two
questions come up everywhere: "where are the stretches where this is true?" and "what is the
average of this around each frame?". This file answers them once:

    runs_of(mask)                      stretches of consecutive True frames, as (start, stop)
    rolling_mean(values, before, after)   NaN-ignoring mean over a window around each frame
    centred(values, half)              the same, symmetric: half frames either side

    from mtah.utils import runs_of, centred
    for start, stop in runs_of(match.graded):      # each graded half
        ...
    smooth_speed = centred(speed, 12)              # +/- 12 frames, about 1 s at 25 fps
"""

import numpy as np


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True; mask must be a numpy bool array.

    Every stretch of consecutive True frames, as (first frame, first frame after it), so
    mask[start:stop] is the stretch and stop - start its length. Used to find ball gaps, the
    graded halves, dead-ball stretches and control events. A plain list does not work.

        mask = np.array([F, T, T, F, F, T, T, T, F])     # frames 0..8
        runs_of(mask)                          # -> [(1, 3), (5, 8)]
    """
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rolling_mean(values, before, after):
    """Mean of values[i - before : i + after + 1], ignoring NaN. NaN where the window is all NaN.

    For every frame i, the average of the frames from `before` frames earlier to `after` frames
    later (both inclusive), skipping missing (NaN) values. The window is cut short at the ends of
    the array rather than padded. Returns a float64 array the same length as values. Lets a
    window look only backwards (after=0) or only forwards (before=0); see centred() for the
    symmetric case.

        values = [1, NaN, 3, 5]
        rolling_mean(values, 1, 0)             # -> [1, 1, 3, 4]      this frame and the one before
        rolling_mean(values, 0, 1)             # -> [1, 3, 4, 5]      this frame and the one after
        rolling_mean([NaN, NaN, 2], 0, 0)      # -> [NaN, NaN, 2]     nothing valid in the window
    """
    values = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.clip(index - before, 0, len(values))
    high = np.clip(index + after + 1, 0, len(values))
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def centred(values, half):
    """Centred rolling mean: values[i - half : i + half + 1], ignoring NaN.

    rolling_mean() with the same number of frames on both sides, so the window is 2 * half + 1
    frames wide and does not shift the signal in time. NaN where the whole window is NaN.

        centred([1, NaN, 3, 5], 1)             # -> [1, 2, 4, 4]
    """
    return rolling_mean(values, half, half)
