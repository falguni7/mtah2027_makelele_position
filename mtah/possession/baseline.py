"""The official nearest-player baseline, vectorised, and carrying the last team through gaps."""

import numpy as np

from baseline import smooth  # starter/, on the path via mtah/__init__.py
from mtah.utils import runs_of


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


def carry_forward(labels, graded):
    """Replace D with the last H / A before it, within each graded half (back-fill at its head).

    The CHALLENGE.md possession rule: a loose ball or a ball out of sight keeps the last team.
    """
    out = labels.copy()
    for start, stop in runs_of(graded):
        window = out[start:stop]
        known = np.flatnonzero(window != "D")
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != "D", np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        out[start:stop] = window[carry]
    return out
