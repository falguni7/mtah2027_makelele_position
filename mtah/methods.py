"""Possession methods, all with the same signature: method(match) -> labels.

`match` is an mtah.data.Match (or a masked copy of one, see mtah.validation.masking). `labels` is an
(n_frames,) array of "H", "A", "D", "X" -- the submission, one row per video frame.

Register a method in METHODS so scripts can run it by name.

Each entry is built by a small factory (baseline_dead(50), hmm_v1(), ...) that returns the actual
method, a function of the match only, named after its METHODS key so outputs can be labelled by
method.__name__. The methods built here write "H", "A" and "D" (and hmm_v8 also "X"); frames outside the
graded halves come out as "D" (or whatever the underlying step leaves there) and are not scored.
The methods follow the roadmap steps:

    baseline_*         official nearest-player baseline, optionally smoothed (mtah.possession)
    baseline50_dead    baseline teams, last team carried through gaps, D from the step-1 detector
    control_dead       step 2: control events decide the team, step-1 detector decides D
    hmm_v1             step 3: control events and dead-ball probability decoded with Viterbi
    hmm_v2             step 4: hmm_v1 plus the ball-free team-shape model in the gaps
    hmm_v3             hmm_v2 with the extra shape features (pressing, compactness: ideas 9, 10)
    hmm_v4             hmm_v3 with cheaper stoppage boundaries near referee whistles
    hmm_v5             hmm_v4 with whistle timing as inputs of the dead-ball classifier
    hmm_v6             hmm_v4 with weaker dead-ball evidence in ball gaps with no reason to stop
    hmm_v7             hmm_v6 with how the tactical camera turns, in the detector and shape model
    hmm_v8             hmm_v7 with X on frames where it is probably wrong (possessions unchanged)
    hmm_v9_*           hmm_v7 with the DFL-trained shape model (mtah.external.shape_transfer):
                       its team opinion averaged with ours, and / or its P(D) for dead ball,
                       fed into the detector (A) or averaged with it (B); see hmm_v9()

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


# The camera column the shape model gets in hmm_v7 (mtah.features.camera).
SHAPE_CAMERA = ["cam_advance_rate"]


def shape_evidence(match, events, columns=None, camera=None):
    """Step 4: out-of-fold P(home) from the ball-free shape model, and the frames to use it on.

    The shape model guesses who has the ball from where the players stand (team shape), without
    looking at the ball. It is trained on pseudo-labels taken from the control events and
    predicted out of fold (each block of the match by a model that never saw it), so the
    probability is never just the training label echoed back. Returns (p_home, frames): p_home
    is a per-frame P(home in possession), NaN where there is no prediction, and frames is a bool
    mask of the graded frames far enough from any control event that the shape is the only team
    evidence there besides the temporal model's carry bias (mtah.possession.shape_model.gap_frames).
    columns picks the shape features the model uses (default: all of team_shape.COLUMNS).
    camera, if given, is the camera table of mtah.features.camera.compute(match): its
    SHAPE_CAMERA column (cam_advance_rate, how fast the camera's aim moves towards the home
    team's attacking goal) is added to the shape table and to the columns.

        events = [(100, 120, 0), (900, 910, 1)]        # (start, stop, team)
        p_home, frames = shape_evidence(match, events)
        p_home[500], frames[500]                       # -> e.g. (0.71, True)   mid-gap
        frames[110]                                    # -> False   on a control event
    """
    from mtah.features import team_shape
    from mtah.possession import shape_model
    target = shape_model.pseudo_labels(match, events)
    columns = list(team_shape.COLUMNS if columns is None else columns)
    table = team_shape.compute(match)
    if camera is not None:
        table = table.assign(**{c: camera[c].to_numpy() for c in SHAPE_CAMERA})
        columns += [c for c in SHAPE_CAMERA if c not in columns]
    p_home, _ = shape_model.out_of_fold(table, target, match.graded, columns)
    return p_home, shape_model.gap_frames(match, events)


def hmm_v2(version=2, dfl_team=False, dfl_dead=None, soften_dfl="combined", dfl_model="boosting",
           weights=None, name=None):
    """Step 4: hmm_v1 plus team evidence from the shape model away from control events.

    Factory: returns the method named "hmm_v<version>" ("hmm_v2" by default). Same Viterbi decoding as hmm_v1, but in long
    stretches with no control event (mostly ball gaps) the shape model's P(home) is added as
    evidence for which team has the ball, instead of leaving the decoder to carry the last team.
    version=2 uses the original step-4 shape features (team_shape.COLUMNS_V1); version=3 adds
    depth, hull area, approach and marking rate (team_shape.COLUMNS) and is named "hmm_v3";
    version=4 is hmm_v3 plus the referee's whistles (mtah/audio/whistle.py): near a whistle,
    entering or leaving D is cheaper (mtah.temporal.hmm.whistle_switch_costs), so a stoppage
    the other evidence asks for can start or end at the whistle; version=5 is hmm_v4 whose
    dead-ball classifier also gets the time since the last whistle and until the next one
    (mtah.dead_ball.detector.WHISTLE), so whistles become dead-ball evidence of their own;
    version=6 is hmm_v4 (not hmm_v5) where, in ball gaps of 3 s or more with no reason for a
    stoppage (no whistle, ball not near a line, no set-piece or goal tag:
    mtah.dead_ball.explain), the odds of P(dead) are divided by 4 before decoding, so a
    stoppage there needs stronger evidence; version=7 is hmm_v6 plus how the tactical camera
    turns (mtah.features.camera): its turning speed over 1 s and 3 s as dead-ball classifier
    inputs (mtah.dead_ball.detector.CAMERA), as the camera rests in stoppages and turns in play,
    and the speed of its aim towards the home team's goal as a shape-model input
    (SHAPE_CAMERA); version=8 is hmm_v7 with abstention (mtah.temporal.abstain): from the same
    evidence the forward-backward posteriors say how likely each label is, and frames whose
    label is probably wrong (posterior below half its expected F1) become "X", never on the
    first frame of a run, so the scorer's fill-forward gives back hmm_v7's labels exactly and
    only macroF1 can change. Versions 4 to 8 need the extracted sound
    (scripts/data/extract_audio.py).

    The keyword options add the DFL-trained shape model (hmm_v9, built with hmm_v9(); they need
    version 7 or higher, so the unexplained-gap softening exists): dfl_team averages its
    P(home | alive) with ours in log-odds; dfl_dead "A" gives its P(D) to the dead-ball
    detector as an input, "B" averages its P(D), calibrated to our confident labels, with the
    detector's P(dead); soften_dfl says what the unexplained-gap softening applies to with "B":
    "combined" (the averaged P(dead)) or "detector" (only the detector's own part).
    dfl_model picks the DFL-trained model: "boosting" (mtah.external.shape_transfer, hmm_v9) or
    "sequence" (the TCN + LSTM average fine-tuned on our confident frames, out of fold:
    mtah.external.shape_sequence, hmm_v10, built with hmm_v10()); dfl_team="replace" uses its
    P(home | alive) INSTEAD of our shape model's, where it has one. weights, a dict of HMM
    weights (names in mtah.temporal.hmm.WEIGHT_NAMES), replaces those weights for this method's
    evidence and decoding (mtah.temporal.hmm.weights), e.g. the ones chosen on the DFL benchmark
    (DFL_WEIGHTS); the others keep their module values. name replaces the default
    name. Methods with a DFL option have method.needs_dfl = True: they
    need the DFL open data in external/dfl/ (scripts/data/download_external.sh --dfl-only),
    which scripts/run_pipeline.py fetches when it is missing.

        method = hmm_v2()
        method.__name__                        # -> "hmm_v2"
        hmm_v2(version=3).__name__             # -> "hmm_v3"
        hmm_v2(version=4).__name__             # -> "hmm_v4"
        hmm_v2(version=5).__name__             # -> "hmm_v5"
        hmm_v2(version=6).__name__             # -> "hmm_v6"
        hmm_v2(version=7).__name__             # -> "hmm_v7"
        hmm_v2(version=8).__name__             # -> "hmm_v8"
        labels = method(match)                 # (n_frames,) of "H" / "A" / "D"
    """
    from mtah.features import team_shape
    columns = team_shape.COLUMNS_V1 if version == 2 else team_shape.COLUMNS
    use_whistles = version >= 4             # cheaper D boundaries near whistles
    whistle_features = version == 5         # whistle timing in the dead-ball classifier
    soften_unexplained = version >= 6       # weaker P(dead) in gaps with no reason to stop
    use_camera = version >= 7               # camera turning in the detector and shape model
    use_abstain = version >= 8              # X where the label is probably wrong
    use_dfl = bool(dfl_team) or dfl_dead is not None
    if use_dfl and version < 7:
        raise ValueError("the DFL options build on version 7")
    if dfl_dead not in (None, "A", "B") or soften_dfl not in ("combined", "detector"):
        raise ValueError("dfl_dead must be None, 'A' or 'B'; soften_dfl 'combined' or 'detector'")
    if dfl_team not in (False, True, "replace") or dfl_model not in ("boosting", "sequence"):
        raise ValueError("dfl_team must be False, True or 'replace'; dfl_model 'boosting' or "
                         "'sequence'")
    if soften_dfl == "detector" and dfl_dead != "B":
        raise ValueError("soften_dfl='detector' needs dfl_dead='B' (A mixes the two in one model)")

    def _evidence(match):
        """Everything the decoder needs for one match, as a tuple for decode().

        (events, p_dead, dead_labels, p_home, shape_frames, switch): the step-2 control events,
        the step-1 P(dead) (softened in unexplained gaps from version 6) and confident labels,
        the step-4 shape model's P(home) and the frames to use it on, and the whistle switch
        costs (None before version 4). Exposed as method.evidence, so checks can compute the
        posteriors of exactly the evidence the method decodes.

            args = hmm_v2(version=7).evidence(match)
            decode(match, *args)               # -> the same labels as hmm_v7(match)
        """
        from mtah import dead_ball, features
        from mtah.audio.whistle import match_whistles
        from mtah.features import camera as camera_features
        from mtah.possession.control import control_events
        from mtah.temporal.hmm import whistle_switch_costs
        whistles = match_whistles() if use_whistles else None
        table = features.compute(match)
        cam = camera_features.compute(match) if use_camera else None
        dfl = None
        if use_dfl:
            from mtah.temporal.fusion import average_logodds, calibrate
            if dfl_model == "sequence":
                from mtah.external.shape_sequence import for_match
            else:
                from mtah.external.shape_transfer import for_match
            dfl = for_match(match)
        _, p_dead, dead_labels = dead_ball.detect(
            match, table, whistles=whistles if whistle_features else None, camera=cam,
            dfl=dfl if dfl_dead == "A" else None)
        dfl_dead_p = calibrate(dfl[:, 2], dead_labels) if dfl_dead == "B" else None
        if soften_unexplained:
            from mtah.dead_ball import explain
            gaps = explain.explain_gaps(match, table, whistles)
            unexplained = explain.unexplained_frames(match.n_frames, gaps)
            if dfl_dead == "B" and soften_dfl == "detector":
                p_dead = average_logodds(explain.soften(p_dead, unexplained), dfl_dead_p)
            elif dfl_dead == "B":
                p_dead = explain.soften(average_logodds(p_dead, dfl_dead_p), unexplained)
            else:
                p_dead = explain.soften(p_dead, unexplained)
        events = control_events(match)
        p_home, frames = shape_evidence(match, events, columns, camera=cam)
        if dfl_team:
            with np.errstate(invalid="ignore", divide="ignore"):
                theirs = dfl[:, 0] / (dfl[:, 0] + dfl[:, 1])
            if dfl_team == "replace":
                p_home = np.where(np.isfinite(theirs), theirs, p_home).astype(np.float32)
            else:
                p_home = average_logodds(p_home, theirs)
        switch = whistle_switch_costs(match.n_frames, whistles) if use_whistles else None
        return events, p_dead, dead_labels, p_home, frames, switch

    def evidence(match):
        """_evidence() with this method's HMM weights (the whistle switch costs use DEAD_SWITCH).

            hmm_v2(version=7, weights={"DEAD_SWITCH": 8.0}).evidence(match)[5][0].max()
            # -> 8.0: entering D costs 8 away from whistles
        """
        from mtah.temporal.hmm import weights as hmm_weights
        with hmm_weights(weights):
            return _evidence(match)

    def method(match):
        """Labels for one match: hmm_v1's decoding plus shape-model evidence in the gaps.

        match is an mtah.data.Match; returns an (n_frames,) array of "H" / "A" / "D" (and "X"
        from version 8 on), with "D" outside the graded halves.

            hmm_v2()(match).shape              # -> (match.n_frames,)
        """
        from mtah.temporal.hmm import decode, decode_posteriors
        from mtah.temporal.hmm import weights as hmm_weights
        with hmm_weights(weights):
            args = _evidence(match)
            labels = decode(match, *args)
            if use_abstain:
                from mtah.temporal.abstain import abstain
                labels = abstain(labels, decode_posteriors(match, *args), match.graded)
        return labels
    method.evidence = evidence
    method.weights = dict(weights or {})
    method.needs_dfl = use_dfl              # needs external/dfl/ (scripts/run_pipeline.py)
    method.__name__ = name or "hmm_v{}".format(version)
    return method


def hmm_v9(team=True, dead="A", soften="combined"):
    """hmm_v7 plus the shape model trained on the 7 labelled DFL matches (players only).

    Factory over hmm_v2(version=7, ...). The DFL-trained model (mtah.external.shape_transfer)
    gives P(H), P(A), P(D) on every graded frame of our match from the players alone. It joins
    the evidence in up to two places:

      team=True     far from control events, the shape evidence becomes the log-odds average
                    of our shape model's P(home) and the DFL model's P(home | alive)
                    (mtah.temporal.fusion.average_logodds): two models whose mistakes only
                    partly overlap (they agree on 84 % of graded frames)
      dead="A"      the DFL model's P(D) and team margin become inputs of the dead-ball
                    detector, which learns on our confident labels how far to trust them
      dead="B"      the DFL model's P(D), calibrated on our confident labels
                    (mtah.temporal.fusion.calibrate), is averaged in log-odds with the
                    detector's P(dead)
      dead=None     dead ball as in hmm_v7
      soften        with "B": the unexplained-gap softening (hmm_v6) applies to the averaged
                    P(dead) ("combined", option i) or only to the detector's part ("detector",
                    option ii, letting the DFL model's independent evidence through)

    Weights and switch costs are hmm_v7's. The name is "hmm_v9" plus "_team" / "_deadA" /
    "_deadB" for single changes, "A" / "B" for both, and "_ii" for soften="detector".

        hmm_v9().__name__                                  # -> "hmm_v9A"
        hmm_v9(dead="B", soften="detector").__name__       # -> "hmm_v9B_ii"
        hmm_v9(team=False, dead="B").__name__              # -> "hmm_v9_deadB"
        hmm_v9(dead=None).__name__                         # -> "hmm_v9_team"
    """
    if team and dead:
        name = "hmm_v9" + dead
    else:
        name = "hmm_v9_team" if team else "hmm_v9_dead" + str(dead)
    if soften == "detector":
        name += "_ii"
    return hmm_v2(version=7, dfl_team=team, dfl_dead=dead, soften_dfl=soften, name=name)


def hmm_v10(team="replace", dead="A"):
    """hmm_v7 plus the sequence shape model: TCN + LSTM, trained on DFL, fine-tuned on our match.

    Factory over hmm_v2(version=7, dfl_model="sequence", ...). The model
    (mtah.external.shape_sequence.for_match) reads 30 s of team shape at a time, without the
    ball, and gives P(H), P(A), P(D) on every graded frame; it was pretrained on 6 DFL matches
    and fine-tuned on our confident frames out of fold (no frame is predicted by a model that
    trained on its time block's labels). It joins the evidence like the DFL boosting model in
    hmm_v9:

      team="replace"  far from control events, its P(home | alive) is the shape evidence,
                      instead of our shape model's (averaging the two hurt in hmm_v9, and this
                      one has learned from the same pseudo-labels plus DFL)
      team=True       the log-odds average of the two, as in hmm_v9
      dead="A"        its P(D) and team margin become dead-ball detector inputs (the best
                      hmm_v9 option)
      team=False / dead=None   that part as in hmm_v7

    Weights and switch costs are hmm_v7's. Names: "hmm_v10A" for both changes (replace + A),
    "hmm_v10_team" / "hmm_v10_deadA" for one, "_avg" when team=True. Needs PyTorch; the first
    run trains the models (~10 min on a GPU, much longer on a CPU) and caches the result.

        hmm_v10().__name__                                 # -> "hmm_v10A"
        hmm_v10(dead=None).__name__                        # -> "hmm_v10_team"
        hmm_v10(team=False).__name__                       # -> "hmm_v10_deadA"
        hmm_v10(team=True, dead=None).__name__             # -> "hmm_v10_team_avg"
    """
    if team and dead:
        name = "hmm_v10" + dead
    else:
        name = "hmm_v10_team" if team else "hmm_v10_dead" + str(dead)
    if team is True:
        name += "_avg"
    return hmm_v2(version=7, dfl_team=team, dfl_dead=dead, dfl_model="sequence", name=name)


# HMM weights chosen on the DFL whole-pipeline benchmark (scripts/external/dfl_benchmark.py
# --wide, "dead" ball placement, best mean spellF1 over all 7 matches): "own_shape" for the
# variants with our shape model's team evidence (core, deadA_boost), "sequence" for those with
# the sequence model's (A_seq). Several are at the edge of that grid (DEAD_WEIGHT 0.0125,
# TEAM_SWITCH 16). The benchmark has no whistles, camera or Smart Tagging, so whether they
# suit our match is checked there without labels (scripts/validation/v9_check.py, masking).
DFL_WEIGHTS = {
    "own_shape": {"TEAM_WEIGHT": 0.2, "DEAD_WEIGHT": 0.0125, "TEAM_SWITCH": 16.0,
                  "DEAD_SWITCH": 2.0},
    "sequence": {"TEAM_WEIGHT": 0.8, "DEAD_WEIGHT": 0.05, "TEAM_SWITCH": 16.0,
                 "DEAD_SWITCH": 4.0},
}


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
    "hmm_v4": hmm_v2(version=4),
    "hmm_v5": hmm_v2(version=5),
    "hmm_v6": hmm_v2(version=6),
    "hmm_v7": hmm_v2(version=7),
    "hmm_v8": hmm_v2(version=8),
    "hmm_v9_team": hmm_v9(dead=None),
    "hmm_v9_deadA": hmm_v9(team=False, dead="A"),
    "hmm_v9_deadB": hmm_v9(team=False, dead="B"),
    "hmm_v9A": hmm_v9(dead="A"),
    "hmm_v9B": hmm_v9(dead="B"),
    "hmm_v9B_ii": hmm_v9(dead="B", soften="detector"),
    "hmm_v10_team": hmm_v10(dead=None),
    "hmm_v10_deadA": hmm_v10(team=False),
    "hmm_v10A": hmm_v10(),
    # the same methods with the DFL-chosen HMM weights
    "hmm_v7_dflw": hmm_v2(version=7, weights=DFL_WEIGHTS["own_shape"], name="hmm_v7_dflw"),
    "hmm_v9_deadA_dflw": hmm_v2(version=7, dfl_dead="A", weights=DFL_WEIGHTS["own_shape"],
                                name="hmm_v9_deadA_dflw"),
    "hmm_v10A_dflw": hmm_v2(version=7, dfl_team="replace", dfl_dead="A", dfl_model="sequence",
                            weights=DFL_WEIGHTS["sequence"], name="hmm_v10A_dflw"),
}
