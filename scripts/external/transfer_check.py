"""Does the DFL-trained shape model work on BAR-ATH? Label-free checks, aggregates only.

    sbatch --time=01:00:00 scripts/slurm/run.sbatch scripts/external/transfer_check.py

The H / A / D shape model of mtah/external/shape_transfer.py is trained on all 7 DFL matches
(players only, degraded like ours, ball hidden) and applied UNCHANGED to our match: the same
feature code (mtah/features/team_shape.py) on the BAR-ATH players, without the camera columns.
There are no labels for our match, so it is compared with what we have, each an opinion and
none ground truth:

1. Team, against the control-event pseudo-labels (frames inside control events and passes
   between them, where the ball is tracked: mtah/possession/shape_model.pseudo_labels). The
   current shape model is trained on exactly these, out of fold; the DFL model never saw them,
   so this is the DFL model's first test on frames we are fairly sure of. Accuracy of the team
   with the larger probability, overall and by confidence.
2. Team, against Smart Tagging's in-possession phases on the frames far from every control
   event (shape_model.gap_frames, where the HMM uses the shape model), all of them and those
   without a tracked ball. Smart Tagging is automatic and has errors; the two models are
   compared on the same frames, so the difference is what counts. The current model's
   0.693 in far gaps without the ball (work log, 2026-11-03) is the number to beat.
3. Dead ball, against the dead-ball detector's confident labels (DEAD / ALIVE rules: set-piece
   tags, the ball out, fast open play ...). AUC of the DFL model's P(D), next to the detector's
   own P(dead) (which is trained on these labels, so its AUC is flattered), overall and inside
   ball gaps of 3 s or more.
4. Dead ball in ball gaps of 3 s or more by reason (mtah/dead_ball/explain.py): mean P(D) of the
   DFL model and the detector's (softened) P(dead) in gaps with a whistle, the ball near a line
   or a set-piece tag, and in unexplained gaps; and P(D) during Smart Tagging's in-possession
   phases (should be low) and in the first 3 s of SET PIECES tags (should be high).
5. Overall: the mean P(D) over the graded frames (the D share the model expects), the
   correlation of P(D) with the detector's P(dead), and how often the two models' team call
   agrees on the graded frames.

The per-frame probabilities are written to outputs/dfl_shape_bar_ath.npy (gitignored, derived
from match data; not opened here), for the HMM step that follows.
"""

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import features  # noqa: E402
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import explain  # noqa: E402
from mtah.dead_ball.detector import ALIVE, DEAD  # noqa: E402
from mtah.external import adapter, dfl  # noqa: E402
from mtah.external import shape_transfer as st  # noqa: E402
from mtah.features import team_shape  # noqa: E402
from mtah.methods import hmm_v2  # noqa: E402
from mtah.possession import shape_model  # noqa: E402
from mtah.validation import tagging  # noqa: E402

# Params
FPS = 25
CONFIDENCE = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
SET_PIECE_DEAD = 3 * FPS
OUT = os.path.join(ROOT, "outputs", "dfl_shape_bar_ath.npy")


def team_call(p_home):
    """"H" where P(home) >= 0.5, "A" where it is below, "" where it is NaN.

        team_call([0.7, 0.2, nan])     # -> ["H", "A", ""]
    """
    p_home = np.asarray(p_home, dtype=np.float64)
    out = np.where(p_home >= 0.5, "H", "A").astype("<U1")
    out[~np.isfinite(p_home)] = ""
    return out


def tag_agreement(calls, tagged, frames):
    """(frames, agree): of the selected frames inside a tagged in-possession phase with a team
    call, how many there are and the share where the call is the tagged team.

        calls = ["H", "A", "H", ""], tagged = ["H", "H", "", "A"], frames all True
        tag_agreement(calls, tagged, frames)    # -> (2, 0.5)
    """
    calls, tagged = np.asarray(calls), np.asarray(tagged)
    use = np.asarray(frames) & (tagged != "") & (calls != "")
    return int(use.sum()), float((calls[use] == tagged[use]).mean()) if use.any() else np.nan


def auc(labels, score, rows):
    """AUC of `score` for DEAD against ALIVE confident labels on the selected rows (NaN if one
    class is missing). 0.5 is chance, 1.0 puts every dead frame above every alive one.

        labels = [DEAD, ALIVE, ALIVE, UNKNOWN], score = [0.9, 0.2, 0.4, 0.5], rows all True
        auc(labels, score, rows)                # -> 1.0
    """
    labels, score = np.asarray(labels), np.asarray(score)
    use = np.asarray(rows) & np.isin(labels, (DEAD, ALIVE)) & np.isfinite(score)
    if len(set(labels[use])) < 2:
        return np.nan
    return float(roc_auc_score(labels[use] == DEAD, score[use]))


if __name__ == '__main__':
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    present = np.asarray(match.ball_present)

    # ------------------------------------------------------------- the model, on all DFL matches
    gaps = adapter.presence_gaps(match)
    built = [st.build(match_id, gaps, seed=k) for k, match_id in enumerate(dfl.MATCHES)]
    model = st.fit([b[0] for b in built], [b[1] for b in built])
    p = np.full((match.n_frames, 3), np.nan, dtype=np.float32)
    table = team_shape.compute(match)
    p[graded] = st.predict(model, table[graded])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, p)
    dfl_home = p[:, 0] / (p[:, 0] + p[:, 1])
    p_d = p[:, 2]

    # ------------------------------------------------------------- what we compare with
    events, p_dead, dead_labels, p_home, far, _ = hmm_v2(version=7).evidence(match)
    target = shape_model.pseudo_labels(match, events)
    tagged = tagging.phase_team(match)

    print("1. team vs control-event pseudo-labels ({} frames)".format(int((target >= 0).sum())))
    labelled = target >= 0
    for name, ph in (("current (out of fold)", p_home), ("DFL model", dfl_home)):
        right = (ph >= 0.5) == (target == 1)
        conf = np.maximum(ph, 1 - ph)
        print("   {:<22} accuracy {:.3f}".format(name, right[labelled].mean()) + "".join(
            "  [{:.1f},{:.1f}) {:.0%} / {:.3f}".format(
                lo, min(hi, 1), (labelled & (conf >= lo) & (conf < hi)).sum() / labelled.sum(),
                right[labelled & (conf >= lo) & (conf < hi)].mean())
            for lo, hi in zip(CONFIDENCE[:-1], CONFIDENCE[1:])))

    print("\n2. team vs Smart Tagging, far from control events")
    for where, rows in (("far frames", far), ("far, no ball", far & ~present)):
        for name, ph in (("current", p_home), ("DFL model", dfl_home)):
            n, agree = tag_agreement(team_call(ph), tagged, rows)
            print("   {:<13} {:<10} frames {:6d}  agree {:.3f}".format(where, name, n, agree))

    gap_list = explain.explain_gaps(match, features.compute(match), match_whistles())
    in_gap = np.zeros(match.n_frames, dtype=bool)
    for g in gap_list:
        in_gap[g["start"]:g["stop"]] = True
    print("\n3. dead ball vs the detector's confident labels (AUC)")
    for where, rows in (("graded", graded), ("in ball gaps >= 3 s", graded & in_gap)):
        print("   {:<20} DEAD {:6d}  ALIVE {:6d}  DFL P(D) {:.3f}  detector P(dead) {:.3f}".format(
            where, int((rows & (dead_labels == DEAD)).sum()), int((rows & (dead_labels == ALIVE)).sum()),
            auc(dead_labels, p_d, rows), auc(dead_labels, p_dead, rows)))

    print("\n4. mean P(D) by situation: DFL model / detector (softened, as hmm_v7 uses it)")
    unexplained = explain.unexplained_frames(match.n_frames, gap_list)
    reasons = {r: np.zeros(match.n_frames, dtype=bool) for r in ("whistle", "line", "tag")}
    for g in gap_list:
        for r in reasons:
            if g[r]:
                reasons[r][g["start"]:g["stop"]] = True
    tags = match.smart_tagging
    set_piece = np.zeros(match.n_frames, dtype=bool)
    for row in tags[tags["code"].str.upper() == "SET PIECES"].itertuples():
        set_piece[row.start_frame:row.start_frame + SET_PIECE_DEAD] = True
    situations = [("gaps with a whistle", reasons["whistle"]), ("gaps with the ball near a line",
                  reasons["line"]), ("gaps with a set-piece tag", reasons["tag"]),
                  ("unexplained gaps", unexplained), ("in-possession phases", tagged != ""),
                  ("first 3 s of SET PIECES", set_piece), ("ball tracked", present)]
    for name, rows in situations:
        rows = rows & graded
        print("   {:<32} frames {:6d}  {:.3f} / {:.3f}".format(
            name, int(rows.sum()), np.nanmean(p_d[rows]), np.nanmean(p_dead[rows])))

    print("\n5. overall: expected D share {:.3f} (DFL P(D) mean), detector {:.3f}; "
          "correlation P(D) vs P(dead) {:.3f}; team calls agree {:.3f}".format(
              np.nanmean(p_d[graded]), np.nanmean(p_dead[graded]),
              np.corrcoef(p_d[graded], p_dead[graded])[0, 1],
              (team_call(dfl_home)[graded] == team_call(p_home)[graded]).mean()))
