"""Compare dead-ball detector variants without labels. Prints aggregates only.

    python scripts/dead_ball/dead_variants.py
    sbatch scripts/slurm/run.sbatch scripts/dead_ball/dead_variants.py

Each variant (VARIANTS below) is the detector of mtah/dead_ball/detector.py with different
feature columns and with or without the "sustained running counts as alive" labels. Each one is
trained twice (without and with the set-piece labels) and then fed to hmm_v3's decoder (all
shape features, team_shape.COLUMNS; no whistle switch costs).
"v1c step 1" is the first version of the detector (running_alive=False); "v2b running alive
(current)" is the detector every method uses except hmm_v5; "v3 whistle timing" adds the referee-whistle inputs
(time since the last whistle and until the next, mtah/dead_ball/detector.py WHISTLE), as in
hmm_v5; "v4 camera turning" adds how fast the tactical camera turns (detector.CAMERA), as in
hmm_v7. The decoder below is the same for every variant (no whistle switch costs), so the
rows differ only in the detector.
Nothing is written to disk.

For each variant:
  restarts     held-out check of dead_check.py (trained without set-piece labels): dead share
               in the 3 s after a set-piece tag start (early, high is good) and 10-20 s after
               (late, low is good)
  dead share   of graded frames, of ball-missing graded frames
  tag D        share called D inside Smart Tagging in-possession phases: all of them, and those
               inside ball gaps far (>= 2 s) from control events. Phases can include short
               stoppages, so this should be low but not zero.
  hmm_*        hmm_v3's decoder with this detector: possessions, tag agree / disagree / D

The printed table has one row per variant and these columns:
  early, late          the two restart numbers above
  alive_labels         number of frames confidently labelled alive (training examples)
  dead, dead_in_gaps   dead share of graded frames / of graded frames without a tracked ball
  tagD, tagD_far_gap   dead share inside in-possession phases / of those, in far ball gaps
  hmm_poss             possessions in the decoded labelling, counted as the official scorer does
  hmm_D                share of graded frames the decoder labels D
  hmm_agree            share of tagged frames where the decoder names the tagged team
  hmm_disagree         share where it names the other team
  hmm_tagD             share where it says D
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import features as F  # noqa: E402
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import detector as D  # noqa: E402
from mtah.features import camera  # noqa: E402
from mtah.methods import shape_evidence  # noqa: E402
from mtah.possession.control import control_events  # noqa: E402
from mtah.possession.shape_model import gap_frames  # noqa: E402
from mtah.temporal.hmm import decode  # noqa: E402
from mtah.validation import tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
VARIANTS = {
    "v1c step 1": dict(columns=D.COLUMNS, running_alive=False),
    "v2a no gap timing": dict(columns=F.PLAYER + D.GAP_ENDS, running_alive=False),
    "v2b running alive (current)": dict(columns=D.COLUMNS, running_alive=True),
    "v2c both": dict(columns=F.PLAYER + D.GAP_ENDS, running_alive=True),
    "v3 whistle timing": dict(columns=D.COLUMNS, running_alive=True, whistles=True),
    "v4 camera turning": dict(columns=D.COLUMNS, running_alive=True, camera=True),
}


def restart_scores(dead, starts):
    """(early, late): mean dead share 0-3 s and 10-20 s after each set-piece tag start.

    The same two numbers as dead_check.py's restart_table, returned instead of printed. A good
    detector is dead just after a tag start (early high) and alive again later (late low).

        dead = np.zeros(1000, bool); dead[0:75] = True     # dead for the first 3 s only
        restart_scores(dead, starts=[0, 500])              # -> (0.5, 0.0)
    """
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
        if options.get("whistles"):
            options = dict(options, whistles=match_whistles())
        if options.get("camera"):
            options = dict(options, camera=camera.compute(match))
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
