"""Are long ball gaps with no reason for a stoppage really open play? Prints aggregates only.

    python scripts/dead_ball/unexplained_check.py [methods...]        # default: hmm_v4
    sbatch scripts/slurm/run.sbatch scripts/dead_ball/unexplained_check.py hmm_v4

mtah/dead_ball/explain.py sorts every ball gap of at least 3 s by its reasons to be a stoppage:
a referee whistle around it, the ball last or next seen near a line, a set-piece or goal tag
around it. A gap with none is "unexplained". The idea (step 3 of the whistle plan): a long D
stretch in an unexplained gap is suspicious, and the methods call too much D in ball gaps. This
script checks whether that holds before any method uses it:

1. The gaps: how many, how long, how many have each reason, how many have none -- by count
   and by frames, and by gap length (3-10 s, 10-30 s, 30 s and more).
2. How much D the step-1 detector (its dead mask) and each method put in explained and in
   unexplained gap frames. If the idea is right, a lot of D sits in unexplained gaps.
3. Independent evidence on what the unexplained gaps are: the share of their frames inside a
   Smart Tagging in-possession phase (BUILD UP, PROGRESSION, ...: play going on) and the
   players' median speed, against the explained gaps and two references -- frames the
   detector is sure are dead (confident DEAD labels) and graded frames with a tracked ball
   (mostly play). Open play looks like the second; stoppages like the first. The set-piece
   and goal tags are used to explain gaps, the in-possession phases are not.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import dead_ball, features  # noqa: E402
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball.detector import DEAD  # noqa: E402
from mtah.dead_ball.explain import explain_gaps  # noqa: E402
from mtah.methods import METHODS  # noqa: E402
from mtah.validation import tagging  # noqa: E402

# Params
FPS = 25
REASONS = ("whistle", "line", "tag")
BUCKETS = ((3, 10), (10, 30), (30, 10 ** 6))     # gap length, seconds


def gap_mask(n_frames, gaps):
    """Bool mask of the frames of the given gaps (dicts with "start" and "stop").

    n_frames is the length of the mask (the match's frame count), gaps a list of gap dicts as
    from mtah.dead_ball.explain.explain_gaps() (only "start" and "stop" are read; stop is
    excluded). Used to pick the frames of the explained, or of the unexplained, gaps, to count
    what the methods label there.

        gap_mask(6, [{"start": 1, "stop": 3}])     # -> [F, T, T, F, F, F]
    """
    mask = np.zeros(n_frames, dtype=bool)
    for g in gaps:
        mask[g["start"]:g["stop"]] = True
    return mask


def explained(g):
    """True if the gap has at least one reason to be a stoppage, False if it is "unexplained".

    g is one gap dict from mtah.dead_ball.explain.explain_gaps(); the reasons are its bool
    fields "whistle" (a referee whistle around the gap), "line" (the ball last or next seen
    near a line) and "tag" (a set-piece or goal tag around it), REASONS above.

        explained({"whistle": False, "line": True, "tag": False})    # -> True
    """
    return any(g[r] for r in REASONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="*", default=["hmm_v4"], choices=sorted(METHODS))
    args = parser.parse_args()

    match = load(mmap=False)
    table = features.compute(match)
    graded = np.asarray(match.graded)
    n = match.n_frames
    gaps = explain_gaps(match, table, match_whistles())
    lengths = np.array([g["stop"] - g["start"] for g in gaps]) / FPS
    yes = [g for g in gaps if explained(g)]
    no = [g for g in gaps if not explained(g)]
    in_yes, in_no = gap_mask(n, yes), gap_mask(n, no)
    missing = graded & ~np.asarray(match.ball_present)

    # ---------------------------------------------------------------- 1. the gaps
    print("1. ball gaps >= 3 s: {} gaps, {:.0f} s ({:.1%} of graded frames without a ball)".format(
        len(gaps), lengths.sum(), (in_yes | in_no).sum() / missing.sum()))
    for r in REASONS:
        has = [g for g in gaps if g[r]]
        print("   {:<8} {:4d} gaps ({:.0%}), {:.0%} of gap frames".format(
            r, len(has), len(has) / len(gaps), gap_mask(n, has).sum() / (in_yes | in_no).sum()))
    print("   none     {:4d} gaps ({:.0%}), {:.0%} of gap frames, {:.0f} s; length p50/p90 "
          "{:.1f} / {:.1f} s".format(
              len(no), len(no) / len(gaps), in_no.sum() / (in_yes | in_no).sum(),
              in_no.sum() / FPS, *np.percentile([(g["stop"] - g["start"]) / FPS for g in no],
                                                [50, 90])))
    for lo, hi in BUCKETS:
        inside = [g for g, s in zip(gaps, lengths) if lo <= s < hi]
        print("   {:>3}-{:<4} s  {:4d} gaps, unexplained {:4d} ({:.0%})".format(
            lo, hi if hi < 10 ** 6 else "", len(inside),
            sum(not explained(g) for g in inside),
            sum(not explained(g) for g in inside) / max(len(inside), 1)))

    # ---------------------------------------------------------------- 2. D in the gaps
    dead, _, labels = dead_ball.detect(match, table)
    print("\n2. D share in gap frames: explained / unexplained (D frames in unexplained gaps)")
    print("   {:<14} {:.3f} / {:.3f}  ({:.0f} s)".format(
        "detector", dead[in_yes].mean(), dead[in_no].mean(), (dead & in_no).sum() / FPS))
    tagged = tagging.phase_team(match)
    results = {}
    for name in args.methods:
        out = METHODS[name](match)
        results[name] = out
        is_d = out == "D"
        print("   {:<14} {:.3f} / {:.3f}  ({:.0f} s)".format(
            name, is_d[in_yes].mean(), is_d[in_no].mean(), (is_d & in_no).sum() / FPS))

    # ---------------------------------------------------------------- 3. what they are
    phase = tagged != ""
    speed = table["speed_mean"].to_numpy()
    refs = [("explained gaps", in_yes), ("unexplained gaps", in_no),
            ("confident dead", graded & (labels == DEAD)),
            ("ball tracked", graded & np.asarray(match.ball_present))]
    print("\n3. in-possession phase share and players' median speed (m/s)")
    for name, mask in refs:
        print("   {:<18} phase {:.3f}   speed {:.2f}   ({:.0f} s)".format(
            name, phase[mask].mean(), np.nanmedian(speed[mask]), mask.sum() / FPS))
    for name, out in results.items():
        for part, mask in (("explained", in_yes), ("unexplained", in_no)):
            a = tagging.agreement(np.where(mask, out, "X"), np.where(mask, tagged, ""))
            print("   {} in {} gaps: tag agree {:.3f} / disagree {:.3f} / D {:.3f} "
                  "({} tagged frames)".format(name, part, a["agree"], a["disagree"],
                                              a["dead"], a["frames"]))
