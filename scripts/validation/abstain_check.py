"""Check X abstention (hmm_v8) before trusting it. Prints aggregates only.

    python scripts/validation/abstain_check.py
    sbatch scripts/slurm/run.sbatch scripts/validation/abstain_check.py

hmm_v8 is hmm_v7 with "X" on the frames whose label is probably wrong
(mtah/temporal/abstain.py), judged by the forward-backward posteriors of the same evidence
(mtah/temporal/hmm.py decode_posteriors). This script checks four things, without labels:

1. Calibration: are the posteriors honest? On frames where Smart Tagging names the team in
   possession and hmm_v7 names a team, frames are binned by the posterior of hmm_v7's label;
   for each bin, how often hmm_v7 agrees with Smart Tagging. If the posterior means anything,
   agreement rises with it. Smart Tagging is automatic and has errors, and in-possession phases
   include short stoppages, so agreement stays below the posterior even when it is honest; the
   trend is what matters.
2. The thresholds: the expected F1 per class (abstain.expected_f1, from the posteriors alone)
   and the threshold F1 / 2 below which a frame becomes X.
3. What is abstained: how many graded frames per label, the share of them inside ball gaps,
   their median posterior, and Smart Tagging agreement on the abstained and on the kept team
   frames. Abstention is worthwhile if agreement is much lower on the abstained frames.
4. The guarantee: hmm_v8's labels, filled the way the scorer fills X
   (possession_metrics.fill_abstentions), must equal hmm_v7's on every graded frame, so that
   spellF1 and the transition scores cannot change. Prints the number of differing frames,
   which must be 0, and the possession counts of both.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.methods import hmm_v2  # noqa: E402
from mtah.temporal.abstain import STATES, abstain, expected_f1  # noqa: E402
from mtah.temporal.hmm import decode, decode_posteriors  # noqa: E402
from mtah.validation import tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0001)


def own_posterior(labels, post):
    """Per frame, the posterior of the frame's own label (NaN for labels other than H / A / D).

        labels = ["H", "D", "X"], post = [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.3, 0.4]]
        own_posterior(labels, post)        # -> [0.7, 0.8, nan]
    """
    labels = np.asarray(labels)
    out = np.full(len(labels), np.nan)
    for k, state in enumerate(STATES):
        rows = labels == state
        out[rows] = np.asarray(post)[rows, k]
    return out


if __name__ == '__main__':
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    present = np.asarray(match.ball_present)
    args = hmm_v2(version=7).evidence(match)
    v7 = decode(match, *args)
    post = decode_posteriors(match, *args)
    v8 = abstain(v7, post, graded)
    own = own_posterior(v7, post)
    tagged = tagging.phase_team(match)
    team = graded & np.isin(v7, ["H", "A"]) & (tagged != "")

    # ---------------------------------------------------------------- 1. calibration
    print("1. calibration: posterior of hmm_v7's team label vs agreement with Smart Tagging")
    print("   {:>12} {:>9} {:>8}".format("posterior", "frames", "agree"))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        rows = team & (own >= lo) & (own < hi)
        print("   {:>5.2f}-{:<6.2f} {:>9d} {:>8.3f}".format(
            lo, min(hi, 1.0), int(rows.sum()),
            (v7[rows] == tagged[rows]).mean() if rows.any() else np.nan))

    # ---------------------------------------------------------------- 2. thresholds
    f1 = expected_f1(v7, post, graded)
    print("\n2. expected F1 (from the posteriors) and threshold F1 / 2: " + "  ".join(
        "{} {:.3f} -> {:.3f}".format(s, f1[s], f1[s] / 2) for s in STATES))

    # ---------------------------------------------------------------- 3. what is abstained
    x = graded & (v8 == "X")
    print("\n3. abstained: {} graded frames ({:.2%}), {:.0%} of them in ball gaps".format(
        int(x.sum()), x.mean() * len(x) / graded.sum(), (x & ~present).sum() / max(x.sum(), 1)))
    for state in STATES:
        rows = x & (v7 == state)
        print("   {}: {:6d} frames ({:.2%} of its graded frames), median posterior {:.2f}".format(
            state, int(rows.sum()), rows.sum() / max((graded & (v7 == state)).sum(), 1),
            np.median(own[rows]) if rows.any() else np.nan))
    for name, rows in (("abstained", team & x), ("kept", team & ~x)):
        print("   team frames tagged by Smart Tagging, {:<9}: {:6d}, agree {:.3f}".format(
            name, int(rows.sum()), (v7[rows] == tagged[rows]).mean() if rows.any() else np.nan))

    # ---------------------------------------------------------------- 4. the guarantee
    runs = pm.valid_runs(np.where(graded, pm.H, pm.X))
    filled = pm.fill_abstentions(pm.label_codes(v8), runs)
    same = pm.label_codes(v7)
    print("\n4. after the scorer's fill-forward, frames differing from hmm_v7: {} (must be 0); "
          "possessions {} / {}".format(
              int((filled[graded] != same[graded]).sum()),
              len(pm.possession_spells(filled, runs)), len(pm.possession_spells(same, runs))))
