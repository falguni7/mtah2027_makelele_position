"""hmm_v9 variants against hmm_v7, without labels. Prints aggregates only.

    sbatch --time=02:00:00 scripts/slurm/run.sbatch scripts/validation/v9_check.py
    python scripts/validation/v9_check.py --methods hmm_v7 hmm_v9A
    python scripts/validation/v9_check.py --methods hmm_v7 hmm_v9_deadA hmm_v10_team \
        hmm_v10_deadA hmm_v10A                                     # hmm_v10 (needs a GPU)

Every method is run on the full match and compared on what we can check without ground truth
(the public window is not used):

  H, A, D            share of graded frames per label
  poss, mean s       possession spells as the official scorer counts them, and their mean length
  tag agree          share of frames inside Smart Tagging's in-possession phases where the method
                     names the tagged team (all graded frames)
  gap agree          the same, only on frames far from every control event without a tracked
                     ball (shape_model.gap_frames, from hmm_v7's events): where team evidence
                     is weakest and the shape models matter
  tag D              share of in-possession-phase frames called D (low is expected: phases can
                     hold short stoppages, but not many)
  unexpl D           share of frames in unexplained ball gaps (no whistle, line or tag;
                     mtah/dead_ball/explain.py) called D
  early, late        share of D 0-3 s and 10-20 s after each SET PIECES tag start: a restart is
                     dead at first (early high; the detector's rules pin the first 3 s, so this
                     is near 1 by construction) and play is on again later (late low)
  changed            frames whose label differs from the first method in the list

Smart Tagging is automatic and has errors, so agreement is consistency, not accuracy; the
masking test (scripts/validation/masking_test.py --methods ...) is run separately.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import features  # noqa: E402
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import explain  # noqa: E402
from mtah.methods import METHODS  # noqa: E402
from mtah.possession import shape_model  # noqa: E402
from mtah.possession.control import control_events  # noqa: E402
from mtah.utils import runs_of  # noqa: E402
from mtah.validation import tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
DEFAULT = ["hmm_v7", "hmm_v9_team", "hmm_v9_deadA", "hmm_v9_deadB", "hmm_v9A", "hmm_v9B",
           "hmm_v9B_ii"]


def restart_shares(labels, starts):
    """(early, late): mean share of D 0-3 s and 10-20 s after each restart frame in `starts`.

        labels D on 0-99, H after; starts = [0]          # 25 fps
        restart_shares(labels, starts)       # -> (1.0, 0.0)   dead for 4 s, alive at 10-20 s
    """
    dead = np.asarray(labels) == "D"
    early = np.mean([dead[s:s + 3 * FPS].mean() for s in starts])
    late = np.mean([dead[s + 10 * FPS:s + 20 * FPS].mean() for s in starts])
    return float(early), float(late)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT, choices=sorted(METHODS))
    args = parser.parse_args()

    match = load(mmap=False)
    graded = np.asarray(match.graded)
    present = np.asarray(match.ball_present)
    tagged = tagging.phase_team(match)
    far = shape_model.gap_frames(match, control_events(match)) & ~present
    gaps = explain.explain_gaps(match, features.compute(match), match_whistles())
    unexplained = explain.unexplained_frames(match.n_frames, gaps) & graded
    tags = match.smart_tagging
    starts = [int(s) for s in tags.loc[tags["code"].str.upper() == "SET PIECES", "start_frame"]
              if graded[int(s)]]
    runs = runs_of(graded)

    print("frames: graded {}, tagged {}, far gaps without ball {} (tagged {}), unexplained {}, "
          "set-piece starts {}".format(int(graded.sum()), int((tagged != "").sum()),
                                       int(far.sum()), int((far & (tagged != "")).sum()),
                                       int(unexplained.sum()), len(starts)))
    head = ["H", "A", "D", "poss", "mean s", "tag agree", "gap agree", "tag D", "unexpl D",
            "early", "late", "changed"]
    print("{:<14}".format("method") + "".join("{:>10}".format(h) for h in head), flush=True)
    first = None
    for name in args.methods:
        labels = np.asarray(METHODS[name](match))
        first = labels if first is None else first
        spells = pm.possession_spells(pm.label_codes(labels), runs)
        lengths = np.array([b - a for a, b, *_ in spells]) / FPS
        on_tag = graded & (tagged != "")
        on_gap = far & (tagged != "")
        early, late = restart_shares(labels, starts)
        values = [(labels[graded] == c).mean() for c in "HAD"] + [
            len(spells), lengths.mean(),
            (labels[on_tag] == tagged[on_tag]).mean(), (labels[on_gap] == tagged[on_gap]).mean(),
            (labels[on_tag] == "D").mean(), (labels[unexplained] == "D").mean(), early, late,
            int((labels[graded] != first[graded]).sum())]
        print("{:<14}".format(name) + "".join(
            "{:>10d}".format(v) if isinstance(v, (int, np.integer)) else "{:>10.3f}".format(v)
            for v in values), flush=True)
