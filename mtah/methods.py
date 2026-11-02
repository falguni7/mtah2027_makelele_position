"""Possession methods, all with the same signature: method(match) -> labels.

`match` is an mtah.cache.Match (or a masked copy of one, see mtah.masking). `labels` is an
(n_frames,) array of "H", "A", "D", "X" -- the submission, one row per video frame.

Register a method in METHODS so scripts can run it by name.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import smooth  # noqa: E402


def nearest_player_codes(match):
    """Team of the player nearest the ball (0 home, 1 away), -1 where there is no ball.

    Step 1 of starter/baseline.py, vectorised. Identical output, including that the first slot
    wins an exact tie.
    """
    # Only frames with a ball, and x and y separately: fewer and smaller temporaries. Allocating
    # full-match temporaries is what made this slow on the cluster nodes, not the arithmetic.
    frames = np.flatnonzero(match.ball_present)
    players = np.asarray(match.players_xy[frames])
    ball = np.asarray(match.ball_xy[frames])
    distance = np.square(players[:, :, 0] - ball[:, 0:1])
    distance += np.square(players[:, :, 1] - ball[:, 1:2])
    distance[np.isnan(distance)] = np.inf
    nearest = distance.argmin(axis=1)
    found = np.isfinite(distance[np.arange(len(frames)), nearest])

    codes = np.full(match.n_frames, -1, dtype=np.int8)
    codes[frames[found]] = match.team_of_slot[nearest[found]]
    return codes


def baseline(window):
    """The official nearest-player baseline with a +/- window majority vote."""
    def method(match):
        return smooth(nearest_player_codes(match), window)
    method.__name__ = "baseline_smooth{}".format(window) if window else "baseline_raw"
    return method


METHODS = {
    "baseline_raw": baseline(0),
    "baseline_smooth25": baseline(25),
    "baseline_smooth50": baseline(50),
    "baseline_smooth125": baseline(125),
}
