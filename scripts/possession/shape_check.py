"""Check the ball-free team-shape model (step 4). Prints aggregates only.

    python scripts/possession/shape_check.py
    sbatch scripts/slurm/run.sbatch scripts/possession/shape_check.py

The shape model guesses which team has the ball from where the players stand and move, without
the ball, so it can fill ball gaps. It is trained on pseudo-labels from the step-2 control rule:
frames inside a control event and passes between two events of the same team. Nothing is
written to disk.

1. Held-out accuracy on the pseudo-labels, per contiguous time block (out-of-fold).
2. Where it matters -- frames at least 2 s from any control event, i.e. inside ball gaps and
   long loose phases -- agreement with Smart Tagging's in-possession team for: the carry rule
   (control_dead), the shape model on its own, and hmm_v2. Smart Tagging's in-possession
   phases are not used by any of them (the dead-ball detector does use the SET PIECES and GOALS
   tags), so the comparison is independent of what it measures; an opinion, not ground truth.

How to read the output:
  pseudo-labels        frames with a pseudo-label, their share of graded frames, and the share
                       of them where the home team has the ball
  block table          per time block: block number, labelled frames in it, held-out accuracy
                       (P(home) > 0.5 against the pseudo-label), home share in that block
  overall              held-out accuracy on all labelled frames; majority class is the accuracy
                       of always guessing the more common team, the number to beat
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


if __name__ == '__main__':
    match = load(mmap=False)
    events = control_events(match)
    target = shape_model.pseudo_labels(match, events)
    table = team_shape.compute(match)
    p_home, report = shape_model.out_of_fold(table, target, match.graded)

    labelled = target >= 0
    print("pseudo-labels: {} frames ({:.1%} of graded), home share {:.2f}".format(
        int(labelled.sum()), labelled[np.asarray(match.graded)].mean(), target[labelled].mean()))
    print("\n1. held-out accuracy by time block")
    print(pd.DataFrame(report).set_index("block").round(3).to_string())
    held = labelled & np.isfinite(p_home)
    print("  overall {:.3f} (majority class {:.3f})".format(
        ((p_home[held] > 0.5) == target[held]).mean(), max(target[held].mean(), 1 - target[held].mean())))
    for low, high in ((0.5, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)):
        confidence = np.maximum(p_home, 1 - p_home)
        sel = held & (confidence >= low) & (confidence < high)
        print("  confidence {:.2f}-{:.2f}: {:5.1%} of labelled frames, accuracy {:.3f}".format(
            low, high, sel.sum() / held.sum(), ((p_home[sel] > 0.5) == target[sel]).mean()))

    print("\n2. far from control events (>= 2 s): agreement with Smart Tagging")
    tagged = tagging.phase_team(match)
    far = shape_model.gap_frames(match, events)
    ball_missing = ~np.asarray(match.ball_present)
    shape_only = np.where(p_home > 0.5, "H", "A")
    candidates = (("carry rule (control_dead)", METHODS["control_dead"](match)),
                  ("shape model alone", shape_only),
                  ("hmm_v2", METHODS["hmm_v2"](match)))
    for frames_name, frames in (("all far frames", far), ("far and ball missing", far & ball_missing)):
        print("  {} ({} graded frames)".format(frames_name, int(frames.sum())))
        for name, labels in candidates:
            n, agree, disagree = agreement_on(labels, tagged, frames)
            print("    {:<28} tagged frames {:>6}  agree {:.3f}  disagree {:.3f}".format(
                name, n, agree, disagree))
