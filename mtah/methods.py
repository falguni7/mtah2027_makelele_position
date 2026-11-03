"""Possession methods, all with the same signature: method(match) -> labels.

`match` is an mtah.data.Match (or a masked copy of one, see mtah.validation.masking). `labels` is an
(n_frames,) array of "H", "A", "D", "X" -- the submission, one row per video frame.

Register a method in METHODS so scripts can run it by name.

Each entry is built by a small factory (baseline_dead(50), hmm_v1(), ...) that returns the actual
method, a function of the match only, named after its METHODS key so outputs can be labelled by
method.__name__. The methods built here only ever write "H", "A" and "D"; frames outside the
graded halves come out as "D" (or whatever the underlying step leaves there) and are not scored.
The methods follow the roadmap steps:

    baseline_*         official nearest-player baseline, optionally smoothed (mtah.possession)
    baseline50_dead    baseline teams, last team carried through gaps, D from the step-1 detector
    control_dead       step 2: control events decide the team, step-1 detector decides D
    hmm_v1             step 3: control events and dead-ball probability decoded with Viterbi
    hmm_v2             step 4: hmm_v1 plus the ball-free team-shape model in the gaps
    hmm_v3             hmm_v2 with the extra shape features (pressing, compactness: ideas 9, 10)

    from mtah.data import load
    from mtah.methods import METHODS
    match = load()
    labels = METHODS["control_dead"](match)
    labels[:3]                                 # -> e.g. array(['D', 'D', 'D'])  before kick-off
"""

import numpy as np

from baseline import smooth  # starter/, on the path via mtah/__init__.py
from mtah.possession.baseline import baseline, carry_forward, nearest_player_codes


def baseline_dead(window):
    """Baseline team where the ball is seen, the last team carried through gaps, and D only where
    the dead-ball detector says so (step 1).

    Factory: returns the method for one smoothing window (+/- `window` frames of majority vote
    over the nearest player's team). The baseline alone says D wherever no frame in its +/- window
    has a tracked ball (it has no distance threshold); here those frames instead keep the last
    team (CHALLENGE.md's rule that a loose or unseen ball stays with the last team), and D comes
    only from the dead-ball detector.
    The method is named "baseline<window>_dead".

        method = baseline_dead(50)
        method.__name__                        # -> "baseline50_dead"
        smoothed team per frame                #    H  H  D  D  A  A      (D: no ball seen)
        after carry_forward                    #    H  H  H  H  A  A
        detector says dead on frames 4..5      #    H  H  H  H  D  D  -> returned labels
    """
    def method(match):
        """Labels for one match: smoothed baseline team, carried through gaps, D where dead.

        match is an mtah.data.Match; returns an (n_frames,) array of "H" / "A" / "D". Frames
        outside the graded halves are not touched by the carry and keep the baseline's value.

            labels = baseline_dead(50)(match)
            labels.shape                       # -> (match.n_frames,)
        """
        from mtah import dead_ball, features
        teams = smooth(nearest_player_codes(match), window)
        graded = np.asarray(match.graded)
        labels = carry_forward(teams, graded)
        is_dead, _, _ = dead_ball.detect(match, features.compute(match))
        labels[is_dead] = "D"
        return labels
    method.__name__ = "baseline{}_dead".format(window)
    return method


def control_dead():
    """Control-event rule (step 2) with the step-1 dead-ball detector for D.

    Factory: returns the method named "control_dead". A control event is a run of frames where
    one team's player is in control of the ball (mtah.possession.control); possession is the
    team of the last event, changing at each event's start, and D wherever the detector says
    the ball is dead.

        events = [(10, 20, 0), (40, 45, 1)]    # (start, stop, team): home at 10, away at 40
        labels on frames 0..59, none dead      # -> H from 0 (restart), H on 10..39, A on 40..59
    """
    def method(match):
        """Labels for one match from control events and the dead-ball detector.

        match is an mtah.data.Match; returns an (n_frames,) array of "H" / "A" / "D".

            control_dead()(match)[:3]          # -> e.g. array(['D', 'D', 'D'])  before kick-off
        """
        from mtah import dead_ball, features
        from mtah.possession.control import control_events, possession_from_events
        is_dead, _, _ = dead_ball.detect(match, features.compute(match))
        return possession_from_events(control_events(match), np.asarray(match.graded), is_dead)
    method.__name__ = "control_dead"
    return method


def hmm_v1():
    """Step 3: dead-ball probabilities and control events decoded together with Viterbi.

    Factory: returns the method named "hmm_v1". Instead of applying hard rules frame by frame,
    each graded half is decoded as one sequence of H / A / D states (mtah.temporal.hmm), so a
    single noisy frame cannot flip possession and the D stretches and team changes are placed
    where the evidence as a whole fits best.

        method = hmm_v1()
        method.__name__                        # -> "hmm_v1"
        labels = method(match)                 # (n_frames,) of "H" / "A" / "D"
    """
    def method(match):
        """Labels for one match: Viterbi decoding of control events and P(dead).

        match is an mtah.data.Match; returns an (n_frames,) array of "H" / "A" / "D", with "D"
        outside the graded halves.

            hmm_v1()(match).shape              # -> (match.n_frames,)
        """
        from mtah import dead_ball, features
        from mtah.possession.control import control_events
        from mtah.temporal.hmm import decode
        _, p_dead, dead_labels = dead_ball.detect(match, features.compute(match))
        return decode(match, control_events(match), p_dead, dead_labels)
    method.__name__ = "hmm_v1"
    return method


def shape_evidence(match, events, columns=None):
    """Step 4: out-of-fold P(home) from the ball-free shape model, and the frames to use it on.

    The shape model guesses who has the ball from where the players stand (team shape), without
    looking at the ball. It is trained on pseudo-labels taken from the control events and
    predicted out of fold (each block of the match by a model that never saw it), so the
    probability is never just the training label echoed back. Returns (p_home, frames): p_home
    is a per-frame P(home in possession), NaN where there is no prediction, and frames is a bool
    mask of the graded frames far enough from any control event that the shape is the only team
    evidence there besides the temporal model's carry bias (mtah.possession.shape_model.gap_frames).
    columns picks the shape features the model uses (default: all of team_shape.COLUMNS).

        events = [(100, 120, 0), (900, 910, 1)]        # (start, stop, team)
        p_home, frames = shape_evidence(match, events)
        p_home[500], frames[500]                       # -> e.g. (0.71, True)   mid-gap
        frames[110]                                    # -> False   on a control event
    """
    from mtah.features import team_shape
    from mtah.possession import shape_model
    target = shape_model.pseudo_labels(match, events)
    columns = team_shape.COLUMNS if columns is None else columns
    p_home, _ = shape_model.out_of_fold(team_shape.compute(match), target, match.graded, columns)
    return p_home, shape_model.gap_frames(match, events)


def hmm_v2(version=2):
    """Step 4: hmm_v1 plus team evidence from the shape model away from control events.

    Factory: returns the method named "hmm_v2". Same Viterbi decoding as hmm_v1, but in long
    stretches with no control event (mostly ball gaps) the shape model's P(home) is added as
    evidence for which team has the ball, instead of leaving the decoder to carry the last team.
    version=2 uses the original step-4 shape features (team_shape.COLUMNS_V1); version=3 adds
    depth, hull area, approach and marking rate (team_shape.COLUMNS) and is named "hmm_v3".

        method = hmm_v2()
        method.__name__                        # -> "hmm_v2"
        hmm_v2(version=3).__name__             # -> "hmm_v3"
        labels = method(match)                 # (n_frames,) of "H" / "A" / "D"
    """
    from mtah.features import team_shape
    columns = {2: team_shape.COLUMNS_V1, 3: team_shape.COLUMNS}[version]

    def method(match):
        """Labels for one match: hmm_v1's decoding plus shape-model evidence in the gaps.

        match is an mtah.data.Match; returns an (n_frames,) array of "H" / "A" / "D", with "D"
        outside the graded halves.

            hmm_v2()(match).shape              # -> (match.n_frames,)
        """
        from mtah import dead_ball, features
        from mtah.possession.control import control_events
        from mtah.temporal.hmm import decode
        _, p_dead, dead_labels = dead_ball.detect(match, features.compute(match))
        events = control_events(match)
        p_home, frames = shape_evidence(match, events, columns)
        return decode(match, events, p_dead, dead_labels, p_home, frames)
    method.__name__ = "hmm_v{}".format(version)
    return method


# The official baselines ship as git-tracked files evaluation/results/<name>.csv, and
# scripts/data/check_cache.py compares the cache against them. Scripts never write over them.
OFFICIAL_BASELINES = frozenset({"baseline_raw", "baseline_smooth25", "baseline_smooth50",
                                "baseline_smooth125"})


# Every method by name. Scripts look methods up here (e.g. METHODS["hmm_v2"](match)); the key
# matches the method's __name__.
METHODS = {
    "baseline_raw": baseline(0),
    "baseline_smooth25": baseline(25),
    "baseline_smooth50": baseline(50),
    "baseline_smooth125": baseline(125),
    "baseline50_dead": baseline_dead(50),
    "control_dead": control_dead(),
    "hmm_v1": hmm_v1(),
    "hmm_v2": hmm_v2(),
    "hmm_v3": hmm_v2(version=3),
}
