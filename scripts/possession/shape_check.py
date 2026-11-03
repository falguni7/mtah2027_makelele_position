"""Check the ball-free team-shape model (step 4). Prints aggregates only.

    python scripts/possession/shape_check.py
    sbatch scripts/slurm/run.sbatch scripts/possession/shape_check.py

The shape model guesses which team has the ball from where the players stand and move, without
the ball, so it can fill ball gaps. It is trained on pseudo-labels from the step-2 control rule:
frames inside a control event and passes between two events of the same team. Nothing is
written to disk.

1. Held-out accuracy on the pseudo-labels (out-of-fold over contiguous time blocks), for each
   feature set in FEATURE_SETS: the step-4 features, each of ideas 9 and 10 added alone, and all.
2. Where it matters -- frames at least 2 s from any control event, i.e. inside ball gaps and
   long loose phases -- agreement with Smart Tagging's in-possession team for: the carry rule
   (control_dead), the shape model on its own with each feature set, hmm_v2 and hmm_v3. Smart
   Tagging's in-possession phases are not used by any of them (the dead-ball detector does use
   the SET PIECES and GOALS tags), so the comparison is independent of what it measures; an
   opinion, not ground truth.

How to read the output:
  pseudo-labels        frames with a pseudo-label, their share of graded frames, and the share
                       of them where the home team has the ball
  feature set row      number of features; overall held-out accuracy (P(home) > 0.5 against
                       the pseudo-label) on all labelled frames; the accuracy of the worst time
                       block; majority class is the accuracy of always guessing the more common
                       team, the number to beat
  block table          for all features: per time block, labelled frames in it, held-out
                       accuracy, home share in that block
  confidence a-b       frames whose max(P(home), P(away)) lies in [a, b): their share of the
                       labelled frames and the accuracy there; accuracy should rise with
                       confidence
  part 2               for all far frames, then far frames without a tracked ball: per method
                       the number of those frames inside an in-possession phase, the share where
                       the method names the tagged team (agree) and the other team (disagree);
                       the rest is D
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.features import team_shape  # noqa: E402
from mtah.methods import METHODS  # noqa: E402
from mtah.possession import shape_model  # noqa: E402
from mtah.possession.control import control_events  # noqa: E402
from mtah.validation import tagging  # noqa: E402


def agreement_on(labels, tagged, frames):
    """(n, agree, disagree) of labels against Smart Tagging, on the selected frames only.

    `tagged` is the per-frame tagged team ("H", "A" or "" outside in-possession phases) and
    `frames` a mask of frames to look at. n counts selected frames that are tagged; agree is the
    share of them where labels name the tagged team, disagree the share where they name the
    other team. D counts as neither, so agree + disagree is below 1 when labels say D.

        labels = np.array(["H", "A", "D", "H"])
        tagged = np.array(["H", "H", "H", ""])
        frames = np.array([True, True, True, True])
        agreement_on(labels, tagged, frames)               # -> (3, 0.333, 0.333)
    """
    covered = frames & (tagged != "")
    labels = np.asarray(labels)[covered]
    return (len(labels), float((labels == tagged[covered]).mean()),
            float(((labels != tagged[covered]) & (labels != "D")).mean()))


# Shape feature sets compared in part 1, each a list of table columns. The shape model is
# trained and scored once per set, with everything else identical, so the difference in
# accuracy is due to the features alone:
#   v1 (step 4)       the original 29 columns (advance, lines, spread, velocity, marking, camera)
#   + idea 9          adds approach and marking_rate: is a team closing in on its opponents
#                     (pressing, likely defending) or moving away from them (likely in possession)
#   + idea 10         adds depth and hull_area: is a team compact (likely defending) or spread
#                     out (likely attacking)
#   all (hmm_v3)      both ideas, the set hmm_v3 uses
# Example row of the output: "v1 + depth, hull_area (idea 10)   35 features  overall 0.869 ...".
FEATURE_SETS = {
    "v1 (step 4)": team_shape.COLUMNS_V1,
    "v1 + approach, marking_rate (idea 9)": team_shape.columns_of(
        team_shape.PER_TEAM_V1 + ["approach", "marking_rate"]),
    "v1 + depth, hull_area (idea 10)": team_shape.columns_of(
        team_shape.PER_TEAM_V1 + ["depth", "hull_area"]),
    "all (hmm_v3)": team_shape.COLUMNS,
}


if __name__ == '__main__':
    match = load(mmap=False)
    events = control_events(match)
    target = shape_model.pseudo_labels(match, events)
    table = team_shape.compute(match)
    labelled = target >= 0
    print("pseudo-labels: {} frames ({:.1%} of graded), home share {:.2f}".format(
        int(labelled.sum()), labelled[np.asarray(match.graded)].mean(), target[labelled].mean()))

    print("\n1. held-out accuracy per feature set")
    p_by_set = {}
    for set_name, columns in FEATURE_SETS.items():
        p_home, report = shape_model.out_of_fold(table, target, match.graded, columns)
        p_by_set[set_name] = p_home
        held = labelled & np.isfinite(p_home)
        blocks = pd.DataFrame(report)["accuracy"]
        print("  {:<38} {:>3} features  overall {:.3f}  worst block {:.3f}  (majority class {:.3f})"
              .format(set_name, len(columns), ((p_home[held] > 0.5) == target[held]).mean(),
                      blocks.min(), max(target[held].mean(), 1 - target[held].mean())))
        confidence = np.maximum(p_home, 1 - p_home)
        for low, high in ((0.5, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)):
            sel = held & (confidence >= low) & (confidence < high)
            print("      confidence {:.2f}-{:.2f}: {:5.1%} of labelled frames, accuracy {:.3f}".format(
                low, high, sel.sum() / held.sum(), ((p_home[sel] > 0.5) == target[sel]).mean()))
    print("  by time block, all features:")
    print(pd.DataFrame(report).set_index("block").round(3).to_string())

    print("\n2. far from control events (>= 2 s): agreement with Smart Tagging")
    tagged = tagging.phase_team(match)
    far = shape_model.gap_frames(match, events)
    ball_missing = ~np.asarray(match.ball_present)
    candidates = [("carry rule (control_dead)", METHODS["control_dead"](match))]
    candidates += [("shape alone, " + set_name, np.where(p > 0.5, "H", "A"))
                   for set_name, p in p_by_set.items()]
    candidates += [("hmm_v2", METHODS["hmm_v2"](match)), ("hmm_v3", METHODS["hmm_v3"](match))]
    for frames_name, frames in (("all far frames", far), ("far and ball missing", far & ball_missing)):
        print("  {} ({} graded frames)".format(frames_name, int(frames.sum())))
        for name, labels in candidates:
            n, agree, disagree = agreement_on(labels, tagged, frames)
            print("    {:<52} tagged frames {:>6}  agree {:.3f}  disagree {:.3f}".format(
                name, n, agree, disagree))
