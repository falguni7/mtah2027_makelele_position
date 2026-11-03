"""Small helpers on per-frame arrays, shared by every part of the pipeline."""

import numpy as np


def runs_of(mask):
    """[(start, stop)] half-open runs where mask is True.

    Every stretch of consecutive True frames, as (first frame, first frame after it), so
    mask[start:stop] is the stretch and stop - start its length. Used to find ball gaps, the
    graded halves, dead-ball stretches and control events.

        mask = [F, T, T, F, F, T, T, T, F]     # frames 0..8
        runs_of(mask)                          # -> [(1, 3), (5, 8)]
    """
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def rolling_mean(values, before, after):
    """Mean of values[i - before : i + after + 1], ignoring NaN. NaN where the window is all NaN."""
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
    return rolling_mean(values, half, half)
