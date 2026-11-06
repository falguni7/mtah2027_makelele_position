"""The whole pipeline on DFL matches made to look like ours, scored against DFL's labels.

    from mtah.external import benchmark as bm
    ctx = bm.context()                                   # our ball and player gap statistics
    match, truth = bm.bench_match("DFL-MAT-J03WMX", ctx, "random")
    ev = bm.evidence(match, truth, "core")
    pred = bm.decode(match, ev, bm.DEFAULT)
    bm.score(truth.labels, pred, truth.half)             # -> {"spellF1": ..., "macroF1": ...}

Our match has no labels, so nothing checks the whole method (control events, dead-ball
detector, shape evidence, HMM) against a truth. DFL has labels, and its tracking can be made to
look like ours: the same steps then run on it and are scored. It also gives a place to choose
the HMM's weights that is not our match (scripts/external/dfl_benchmark.py).

How a DFL match is made to look like ours (bench_match):

  players     degraded exactly as for the DFL shape models (adapter.as_match with the same
              seed per match as shape_transfer.build), so the cached DFL model predictions fit
              the same degraded players
  ball        hidden in gaps drawn from our ball gap lengths until the same share of frames as
              ours (~41 % of graded frames) has no ball (adapter.hide_ball). Two placements:
              "random" (uniform starts) and "dead" (a start DEAD_BIAS times as likely on a
              dead-ball frame, as our ball is often lost in stoppages). DEAD_BIAS is a guess,
              not measured: we have no labels to measure it on
  halves      every frame graded except the last frame of the first half, so the HMM decodes
              each half on its own as on our match (that one frame is "D")
  missing     no whistles (no sound), no camera (no video), no Smart Tagging (an empty tag
              table: the set-piece and goal rules of the dead-ball detector never fire). So
              the benchmark runs the core of the method, not all of hmm_v7

Truth: DFL's controlled-possession labels (OR rule, mtah/external/controlled.py), D where
DFL's BallStatus says dead. Scores (score): spellF1 per half as on our match, macroF1, number
of possessions and their mean length.

Variants (VARIANTS), all on the same core (hmm_v3's steps: control events, the dead-ball
detector trained in the match on its own confident rule labels, our shape model trained in the
match out of fold on control-event pseudo-labels, Viterbi):

  core          nothing from the DFL models
  deadA_boost   the DFL boosting model's P(D) and team margin as detector inputs (hmm_v9_deadA)
  deadA_seq     the same from the sequence model (TCN + LSTM average; hmm_v10_deadA)
  team_seq      the sequence model's P(home | alive) instead of our shape model's (hmm_v10_team)
  A_seq         both (hmm_v10A)

The DFL model predictions are each match's leave-one-match-out ones, cached by
scripts/external/dfl_shape.py and dfl_sequence.py: on every DFL match they come from models that
never trained on it. On our match the sequence models are also fine-tuned on our confident
frames; on DFL they are only pretrained. That is one difference to keep in mind.

The HMM weights are module constants of mtah.temporal.hmm; decode() sets them for one call
(hmm.weights()), so a grid can try many without changing the module for anyone else.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import possession_metrics as pm  # evaluation/, on the path via mtah/__init__.py
from mtah import dead_ball, features
from mtah.external import adapter, dfl
from mtah.external import shape_transfer as st
from mtah.possession import shape_model
from mtah.possession.control import control_events
from mtah.temporal import hmm
from mtah.utils import runs_of

# Params
PLACEMENTS = ("random", "dead")
DEAD_BIAS = 4.0             # "dead" placement: a gap start is this much likelier on a dead frame
BALL_SEED = 1000            # ball gaps use seed BALL_SEED + match index (+ 1 for "dead")
VARIANTS = {
    "core": dict(dead=None, team=None),
    "deadA_boost": dict(dead="boosting", team=None),
    "deadA_seq": dict(dead="sequence", team=None),
    "team_seq": dict(dead=None, team="sequence"),
    "A_seq": dict(dead="sequence", team="sequence"),
}
WEIGHTS = hmm.WEIGHT_NAMES
DEFAULT = {w: getattr(hmm, w) for w in WEIGHTS}       # the weights every method uses today
NO_TAGS = pd.DataFrame({"code": pd.Series(dtype=str), "type": pd.Series(dtype=str),
                        "start_frame": pd.Series(dtype=np.int64),
                        "end_frame": pd.Series(dtype=np.int64)})


@dataclass
class Truth:
    """What a benchmark match is scored against: labels (H / A / D per row), half (1 / 2 per
    row), the match index k in dfl.MATCHES and its id."""
    labels: np.ndarray
    half: np.ndarray
    k: int
    match_id: str


def context():
    """What leaves our match for the benchmark: player drop-out lengths and ball gap statistics.

    A dict with "player_gaps" (adapter.presence_gaps) and "ball_lengths", "ball_share"
    (adapter.ball_gaps): lengths in frames and a share, nothing per frame. Loads our match from
    the cache.

        ctx = context()
        ctx["ball_share"]             # -> about 0.41
    """
    from mtah.data import load
    match = load(mmap=False)
    lengths, share = adapter.ball_gaps(match)
    return {"player_gaps": adapter.presence_gaps(match), "ball_lengths": lengths,
            "ball_share": share}


def bench_match(match_id, ctx, placement):
    """(DFLAsMatch made to look like ours, Truth) for one DFL match and ball placement.

    Players degraded with seed k (the match's index in dfl.MATCHES, as in shape_transfer.build),
    ball hidden by adapter.hide_ball with seed BALL_SEED + k (+ 1 for "dead"), graded all but
    the first half's last frame, an empty Smart Tagging table, no half_blocks in meta (so the
    detector's second-half-kick-off rule, which is about our video, does not fire). The labels
    are shape_transfer.build's cached OR labels.

        match, truth = bench_match("DFL-MAT-J03WMX", ctx, "random")
        match.ball_present.mean()     # -> about 0.59 (1 - ctx["ball_share"])
        truth.labels[:3]              # -> array(['A', 'A', 'A'])
    """
    if placement not in PLACEMENTS:
        raise ValueError("placement must be one of {}".format(PLACEMENTS))
    k = dfl.MATCHES.index(match_id)
    m = dfl.load(match_id)
    _, labels = st.build(match_id, ctx["player_gaps"], seed=k)
    match = adapter.as_match(m, np.random.default_rng(k), ctx["player_gaps"])
    rng = np.random.default_rng(BALL_SEED + k + (placement == "dead"))
    hidden = adapter.hide_ball(m.n_frames, ctx["ball_lengths"], ctx["ball_share"], rng,
                               dead=~np.asarray(m.alive),
                               dead_bias=DEAD_BIAS if placement == "dead" else 1.0,
                               half=m.half)
    adapter.with_ball_gaps(match, hidden)
    graded = np.ones(m.n_frames, dtype=bool)
    graded[np.flatnonzero(np.asarray(m.half) == 1)[-1]] = False
    match.graded = graded
    match.smart_tagging = NO_TAGS
    match.meta = {"half_blocks": []}
    return match, Truth(labels=np.asarray(labels), half=np.asarray(m.half), k=k,
                        match_id=match_id)


def dfl_probs(match_id, model):
    """(n, 3) leave-one-match-out P(H), P(A), P(D) of a DFL model on one DFL match.

    model "boosting" reads <match>_shape_oof.npy (scripts/external/dfl_shape.py), "sequence"
    the average of <match>_tcn_oof.npy and <match>_lstm_oof.npy (dfl_sequence.py), as the
    TCN + LSTM average used on our match.

        dfl_probs("DFL-MAT-J03WMX", "sequence").sum(axis=1)[:2]   # -> [1., 1.]
    """
    path = os.path.join(dfl.CACHE, match_id + "_{}_oof.npy")
    if model == "boosting":
        return np.load(path.format("shape"))
    if model == "sequence":
        return (np.load(path.format("tcn")) + np.load(path.format("lstm"))) / 2
    raise ValueError("model must be 'boosting' or 'sequence'")


def evidence(match, truth, variant, team_table=None):
    """The HMM's inputs for one benchmark match and variant, as a dict.

    Keys events, p_dead, dead_labels, p_home, frames: the arguments of hmm.decode after match,
    built as hmm_v3 builds them (methods.hmm_v2, version 3: no whistles, no camera, no
    softening), with the VARIANTS changes. Our shape model reads the ball-free DFL shape table
    (shape_transfer.COLUMNS, cached by shape_transfer.build; pass team_table to reuse one).
    Also "d_auc": how well the detector's P(dead) ranks the true D frames, for the report.

        ev = evidence(match, truth, "deadA_seq")
        len(ev["events"]), ev["p_dead"].shape      # -> e.g. (2400, (145967,))
    """
    spec = VARIANTS[variant]
    if team_table is None:
        team_table, _ = st.build(truth.match_id, None, seed=truth.k)
    table = features.compute(match)
    dead_dfl = dfl_probs(truth.match_id, spec["dead"]) if spec["dead"] else None
    _, p_dead, dead_labels = dead_ball.detect(match, table, dfl=dead_dfl)
    events = control_events(match)
    target = shape_model.pseudo_labels(match, events)
    p_home, _ = shape_model.out_of_fold(team_table, target, match.graded, st.COLUMNS)
    if spec["team"]:
        p = dfl_probs(truth.match_id, spec["team"])
        with np.errstate(invalid="ignore", divide="ignore"):
            theirs = p[:, 0] / (p[:, 0] + p[:, 1])
        p_home = np.where(np.isfinite(theirs), theirs, p_home).astype(np.float32)
    from sklearn.metrics import roc_auc_score
    finite = np.isfinite(p_dead)
    return {"events": events, "p_dead": p_dead, "dead_labels": dead_labels, "p_home": p_home,
            "frames": shape_model.gap_frames(match, events),
            "d_auc": float(roc_auc_score((truth.labels == "D")[finite], p_dead[finite]))}


def decode(match, ev, values):
    """H / A / D labels from the evidence dict of evidence() with the HMM weights `values`.

    Missing names keep their module value. Exactly hmm.decode, only with other weights.

        decode(match, ev, DEFAULT)                    # today's weights
        decode(match, ev, {**DEFAULT, "TEAM_WEIGHT": 0.2})
    """
    with hmm.weights(values):
        return hmm.decode(match, ev["events"], ev["p_dead"], ev["dead_labels"], ev["p_home"],
                          ev["frames"])


def score(labels, pred, half):
    """spellF1 (per half, as on our match), macroF1, possessions and mean length in seconds.

    labels and pred are H / A / D arrays of one match, half 1 / 2 per frame. The same metric
    code as the official scorer (evaluation/possession_metrics.py).

        labels = H H H A A A D D, pred the same, one half
        score(labels, pred, half)     # -> spellF1 1.0, macroF1 1.0, possessions 2,
                                      #    mean_s 0.12 (3 frames at 25 fps)
    """
    gt, pr = pm.label_codes(labels), pm.label_codes(pred)
    runs = [r for h in (1, 2) for r in runs_of(np.asarray(half) == h)]
    cm3 = pm.scored_confusion(pm.confusion(gt, pr))
    stats = pm.possession_stats(pr, runs)
    return {"spellF1": pm.spell_scores(gt, pr, runs)["mean_f1"], "macroF1": pm.macro_f1(cm3),
            "possessions": stats["n_spells"], "mean_s": stats["mean_duration_s"]}
