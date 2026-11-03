"""The official nearest-player baseline, vectorised, and carrying the last team through gaps.

The starter's baseline (starter/baseline.py) labels each frame with the team of the player nearest
the ball and smooths that with a majority vote. This file has the same method, fast enough for
whole matches, plus carry_forward(), the CHALLENGE.md rule that a loose or unseen ball keeps the
last team, used to turn any H / A / D labelling into one without unexplained D gaps.

    from mtah.possession.baseline import baseline, carry_forward
    method = baseline(12)                    # +/- 12 frame majority vote; method.__name__ is
    labels = method(match)                   # "baseline_smooth12"; "H" / "A" / "D" per frame
    labels = carry_forward(labels, match.graded)
"""

import numpy as np

from baseline import smooth  # starter/, on the path via mtah/__init__.py
from mtah.utils import runs_of


def nearest_player_codes(match):
    """Team of the player nearest the ball (0 home, 1 away), -1 where there is no ball.

    Step 1 of starter/baseline.py, vectorised: the same decisions as starter/baseline.py
    (verified identical on the full match, all four variants), and the first slot wins an exact
    tie. In an exact tie computed in float32 instead of float64 the two could in principle
    differ. Takes a match (ball_xy, ball_present, players_xy, team_of_slot) and returns
    one int8 code per frame. Frames with a ball but no tracked player are also -1.

        # two slots: slot 0 home at (0, 0), slot 1 away at (10, 0)
        # frame 0: ball at (8, 0); frame 1: no ball; frame 2: ball at (1, 0)
        nearest_player_codes(match)          # -> [1, -1, 0]
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
    """The official nearest-player baseline with a +/- window majority vote.

    Returns a method: a function of a match giving "H" / "A" / "D" per frame, the nearest
    player's team voted over +/- window frames (starter smooth(); ties go to home, D where no
    frame in the window has a ball). window 0 is the raw, unsmoothed baseline. The method is
    named after its window so evaluation tables can tell the variants apart.

        baseline(0).__name__                 # -> "baseline_raw"
        baseline(12).__name__                # -> "baseline_smooth12"
        # nearest codes [0, 0, 1, 0, 0] (one-frame flicker to away)
        baseline(0)(match)                   # -> ["H", "H", "A", "H", "H"]
        baseline(1)(match)                   # -> ["H", "H", "H", "H", "H"]
    """
    def method(match):
        """Labels per frame of match ("H" / "A" / "D"): nearest-player team, majority vote.

            baseline(1)(match)               # match with nearest codes [0, 1, 1, -1]
                                             # -> ["H", "A", "A", "A"]
        """
        return smooth(nearest_player_codes(match), window)
    method.__name__ = "baseline_smooth{}".format(window) if window else "baseline_raw"
    return method


def carry_forward(labels, graded):
    """Replace D with the last H / A before it, within each graded half (back-fill at its head).

    The CHALLENGE.md possession rule: a loose ball or a ball out of sight keeps the last team.
    labels is an array of "H" / "A" / "D", graded a bool mask; a new array is returned. Each
    graded half is filled on its own, so nothing carries across half-time; D at the start of a
    half takes the first team seen in it. A half with no H / A at all stays D, and frames outside
    graded are left as they were. Every D in a half with a team is replaced, including any that
    meant a dead ball.

        labels = ["D", "H", "D", "A", "D"]
        graded = [T, T, T, T, T]
        carry_forward(labels, graded)        # -> ["H", "H", "H", "A", "A"]

        graded = [T, T, F, T, T]             # two halves: frames 0-1 and 3-4
        carry_forward(labels, graded)        # -> ["H", "H", "D", "A", "A"]
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
