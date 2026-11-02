"""Possession methods, all with the same signature: method(match) -> labels.

`match` is an mtah.data.Match (or a masked copy of one, see mtah.validation.masking). `labels` is an
(n_frames,) array of "H", "A", "D", "X" -- the submission, one row per video frame.

Register a method in METHODS so scripts can run it by name.
"""

import numpy as np

from baseline import smooth  # starter/, on the path via mtah/__init__.py
from mtah.possession.baseline import baseline, carry_forward, nearest_player_codes


def baseline_dead(window):
    """Baseline team where the ball is seen, the last team carried through gaps, and D only where
    the dead-ball detector says so (step 1)."""
    def method(match):
        from mtah import dead_ball, features
        teams = smooth(nearest_player_codes(match), window)
        graded = np.asarray(match.graded)
        labels = carry_forward(teams, graded)
        is_dead, _, _ = dead_ball.detect(match, features.compute(match))
        labels[is_dead] = "D"
        return labels
    method.__name__ = "baseline{}_dead".format(window)
    return method


METHODS = {
    "baseline_raw": baseline(0),
    "baseline_smooth25": baseline(25),
    "baseline_smooth50": baseline(50),
    "baseline_smooth125": baseline(125),
    "baseline50_dead": baseline_dead(50),
}
