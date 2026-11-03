"""Compare dead-ball detector variants without labels. Prints aggregates only.

    python scripts/dead_ball/dead_variants.py

For each variant:
  restarts     held-out check of dead_check.py (trained without set-piece labels): dead share
               in the 3 s after a set-piece tag start (early, high is good) and 10-20 s after
               (late, low is good)
  dead share   of graded frames, of ball-missing graded frames
  tag D        share called D inside Smart Tagging in-possession phases: all of them, and those
               inside ball gaps far (>= 2 s) from control events. Phases can include short
               stoppages, so this should be low but not zero.
  hmm_v2       the step-4 decoder with this detector: possessions, tag agree / disagree / D
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import features as F  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import detector as D  # noqa: E402
from mtah.methods import shape_evidence  # noqa: E402
from mtah.possession.control import control_events  # noqa: E402
from mtah.possession.shape_model import gap_frames  # noqa: E402
from mtah.temporal.hmm import decode  # noqa: E402
from mtah.validation import tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
VARIANTS = {
    "v1c current": dict(columns=D.COLUMNS, running_alive=False),
    "v2a no gap timing": dict(columns=F.PLAYER + D.GAP_ENDS, running_alive=False),
    "v2b running alive": dict(columns=D.COLUMNS, running_alive=True),
    "v2c both": dict(columns=F.PLAYER + D.GAP_ENDS, running_alive=True),
}


def restart_scores(dead, starts):
    early = np.mean([dead[s:s + 3 * FPS].mean() for s in starts])
    late = np.mean([dead[s + 10 * FPS:s + 20 * FPS].mean() for s in starts])
    return early, late


if __name__ == '__main__':
    match = load(mmap=False)
    table = F.compute(match)
    graded = np.asarray(match.graded)
    missing = graded & ~np.asarray(match.ball_present)
    tags = match.smart_tagging
    starts = tags[(tags["code"].str.upper() == "SET PIECES")
                  & ~tags["type"].str.lower().str.startswith("kick off")]["start_frame"].to_numpy()
    tagged = tagging.phase_team(match)
    in_phase = tagged != ""
    events = control_events(match)
    far_gap = gap_frames(match, events) & missing
    p_home, shape_frames = shape_evidence(match, events)
    runs = pm.valid_runs(np.where(graded, pm.H, pm.X))

    rows = []
    for name, options in VARIANTS.items():
        held_dead, _, _ = D.detect(match, table, use_set_pieces=False, **options)
        early, late = restart_scores(held_dead, starts)
        dead, p_dead, labels = D.detect(match, table, **options)
        labels_hmm = decode(match, events, p_dead, labels, p_home, shape_frames)
        agree = tagging.agreement(labels_hmm, tagged)
        rows.append({
            "variant": name,
            "early": early, "late": late,
            "alive_labels": int((labels == D.ALIVE).sum()),
            "dead": dead[graded].mean(), "dead_in_gaps": dead[missing].mean(),
            "tagD": dead[in_phase].mean(), "tagD_far_gap": dead[in_phase & far_gap].mean(),
            "hmm_poss": len(pm.possession_spells(pm.label_codes(labels_hmm), runs)),
            "hmm_D": (labels_hmm[graded] == "D").mean(),
            "hmm_agree": agree["agree"], "hmm_disagree": agree["disagree"],
            "hmm_tagD": agree["dead"],
        })
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220,
                           "display.max_columns", 30):
        print(pd.DataFrame(rows).set_index("variant"))
