"""Does the way the tactical camera turns tell dead from alive, and home from away? Aggregates only.

    python scripts/camera/camera_check.py
    sbatch scripts/slurm/run.sbatch scripts/camera/camera_check.py

Checks the camera features of mtah/features/camera.py before any method uses them. Nothing is
written to disk.

0. Coverage: the share of graded frames with a usable camera value, and with a poor
   homography fit (reprojection error above 1).
1. Dead or alive. The camera's turning speed (cam_turn, cam_turn_3s) and where it points
   (cam_line_dist) on the frames the dead-ball detector is sure about (its confident DEAD and
   ALIVE labels, mtah/dead_ball/detector.py confident_labels), on all of them and on those
   inside ball gaps. For each feature: medians on dead and on alive frames, and the AUC of
   telling them apart (0.5 = no better than a coin, 1 = perfect; shown as at least 0.5, with
   "low" or "high" saying which way dead lies). As a reference, the same for the players' mean
   speed over 3 s (speed_mean_3s), a classifier input already. Then the AUC of cam_turn_3s
   inside thirds of speed_mean_3s: does the camera add anything once the players' speed is
   known? AUCs near 0.5 there mean no.
2. In the long ball gaps (mtah/dead_ball/explain.py): the camera's median turning speed in
   explained and unexplained gaps, against confident dead frames and frames with the ball
   tracked. Like the players' speed in scripts/dead_ball/unexplained_check.py.
3. Home or away. The aim point's speed towards the home team's attacking goal
   (cam_advance_rate) against the shape model's pseudo-labels (control events, home = 1), and
   on the ball-gap frames far from control events against Smart Tagging's in-possession team.
   As a reference, the shape model's existing camera_advance_h (where the camera points, not
   how it moves). AUC above 0.5 means "higher value, more often home".

The confident labels use the ball and the set-piece and goal tags, but none of the camera
features; the Smart Tagging in-possession phases are used only as a check.
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
from mtah.dead_ball.detector import ALIVE, DEAD, confident_labels  # noqa: E402
from mtah.dead_ball.explain import explain_gaps  # noqa: E402
from mtah.features import camera, team_shape  # noqa: E402
from mtah.possession.control import control_events  # noqa: E402
from mtah.possession.shape_model import gap_frames, pseudo_labels  # noqa: E402
from mtah.validation import tagging  # noqa: E402

# Params
FPS = 25
POOR_FIT = 1.0              # reprojection error above this is a poor homography fit


def auc(values, positive):
    """(AUC, side): how well values separate the positive frames from the others.

    values and positive are per frame (NaN values are left out). The AUC is reported as at
    least 0.5, and side says on which side the positives lie: "high" if they tend to have the
    higher values, "low" if the lower. NaN, "-" when either group is empty.

        auc([1, 2, 3, 4], [False, False, True, True])     # -> (1.0, "high")
        auc([4, 3, 2, 1], [False, False, True, True])     # -> (1.0, "low")
    """
    values = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    ok = np.isfinite(values)
    if positive[ok].all() or not positive[ok].any():
        return np.nan, "-"
    a = roc_auc_score(positive[ok], values[ok])
    return (a, "high") if a >= 0.5 else (1 - a, "low")


def gap_mask(n_frames, gaps):
    """Bool mask of the frames of the gap dicts (with "start" and "stop", stop excluded).

    n_frames is the mask's length, gaps a list of gap dicts as from
    mtah.dead_ball.explain.explain_gaps() (only "start" and "stop" are read). Used to pick the
    frames of the explained, or the unexplained, ball gaps for part 2.

        gap_mask(6, [{"start": 1, "stop": 3}])     # -> [F, T, T, F, F, F]
    """
    mask = np.zeros(n_frames, dtype=bool)
    for g in gaps:
        mask[g["start"]:g["stop"]] = True
    return mask


if __name__ == '__main__':
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    present = np.asarray(match.ball_present)
    table = features.compute(match)
    cam = camera.compute(match)

    # ---------------------------------------------------------------- 0. coverage
    error = np.asarray(match.reprojection_error, dtype=np.float64)
    print("0. graded frames with a camera value: {:.3f}; with reprojection error > {:.0f}: "
          "{:.3f}".format(np.isfinite(cam["cam_turn"].to_numpy())[graded].mean(), POOR_FIT,
                          (error[graded] > POOR_FIT).mean()))
    turn = cam["cam_turn"].to_numpy()
    print("   cam_turn on graded frames, px/s: p10/p50/p90 {}; ball-gap frames {}".format(
        np.nanpercentile(turn[graded], [10, 50, 90]).round(0),
        np.nanpercentile(turn[graded & ~present], [10, 50, 90]).round(0)))

    # ---------------------------------------------------------------- 1. dead or alive
    labels = confident_labels(match, table)
    sure = labels != -1
    print("\n1. confident frames: dead {}, alive {} (in ball gaps: dead {}, alive {})".format(
        int((labels == DEAD).sum()), int((labels == ALIVE).sum()),
        int(((labels == DEAD) & ~present).sum()), int(((labels == ALIVE) & ~present).sum())))
    print("   {:<16} {:<8} {:>10} {:>10} {:>12}".format("feature", "frames", "dead p50",
                                                         "alive p50", "AUC (dead)"))
    signals = {name: cam[name].to_numpy() for name in ("cam_turn", "cam_turn_3s",
                                                       "cam_line_dist")}
    signals["speed_mean_3s"] = table["speed_mean_3s"].to_numpy()
    for name, values in signals.items():
        for part, mask in (("all", sure), ("in gaps", sure & ~present)):
            dead = mask & (labels == DEAD)
            alive = mask & (labels == ALIVE)
            a, side = auc(values[mask], (labels == DEAD)[mask])
            print("   {:<16} {:<8} {:>10.2f} {:>10.2f} {:>7.3f} {}".format(
                name, part, np.nanmedian(values[dead]), np.nanmedian(values[alive]), a, side))
    speed = signals["speed_mean_3s"]
    edges = np.nanpercentile(speed[sure], [100 / 3, 200 / 3])
    print("   cam_turn_3s within thirds of speed_mean_3s (players' speed known):")
    for i, (lo, hi) in enumerate(zip([-np.inf, *edges], [*edges, np.inf])):
        mask = sure & (speed >= lo) & (speed < hi)
        a, side = auc(signals["cam_turn_3s"][mask], (labels == DEAD)[mask])
        print("     third {} ({:.2f}-{:.2f} m/s): AUC {:.3f} {}, dead share {:.2f}".format(
            i + 1, max(lo, 0), min(hi, 99), a, side, (labels == DEAD)[mask].mean()))

    # ---------------------------------------------------------------- 2. long ball gaps
    gaps = explain_gaps(match, table, match_whistles())
    yes = gap_mask(match.n_frames, [g for g in gaps if g["whistle"] or g["line"] or g["tag"]])
    no = gap_mask(match.n_frames, [g for g in gaps
                                   if not (g["whistle"] or g["line"] or g["tag"])])
    print("\n2. camera turning speed (px/s, median) in long ball gaps")
    for name, mask in (("explained gaps", yes), ("unexplained gaps", no),
                       ("confident dead", graded & (labels == DEAD)),
                       ("ball tracked", graded & present)):
        print("   {:<18} cam_turn {:6.0f}   cam_turn_3s {:6.0f}   ({:.0f} s)".format(
            name, np.nanmedian(turn[mask]), np.nanmedian(signals["cam_turn_3s"][mask]),
            mask.sum() / FPS))

    # ---------------------------------------------------------------- 3. home or away
    events = control_events(match)
    target = pseudo_labels(match, events)
    labelled = target >= 0
    shape = team_shape.compute(match)
    tagged = tagging.phase_team(match)
    far = gap_frames(match, events) & ~present & (tagged != "")
    print("\n3. home or away: AUC for home (pseudo-labels: {} frames; far ball-gap frames with "
          "a Smart Tagging team: {})".format(int(labelled.sum()), int(far.sum())))
    for name, values in (("cam_advance_rate", cam["cam_advance_rate"].to_numpy()),
                         ("camera_advance_h", shape["camera_advance_h"].to_numpy())):
        a1, s1 = auc(values[labelled], target[labelled] == 1)
        a2, s2 = auc(values[far], tagged[far] == "H")
        print("   {:<18} pseudo-labels {:.3f} {:<4}   tagged gap frames {:.3f} {}".format(
            name, a1, s1, a2, s2))
