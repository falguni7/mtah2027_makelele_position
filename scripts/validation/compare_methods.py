"""Compare methods on the full match without labels, and write their submissions.

    python scripts/validation/compare_methods.py baseline_smooth50 baseline50_dead control_dead
    sbatch scripts/slurm/run.sbatch scripts/validation/compare_methods.py control_dead hmm_v2

For each method: label shares, possessions (count and mean length, as the official scorer counts
them), and agreement with Smart Tagging's in-possession phases (mtah/validation/tagging.py).
Each submission is written to evaluation/results/<method>.csv. Aggregates only.

The official baselines (baseline_raw, baseline_smooth25, baseline_smooth50, baseline_smooth125,
as in the first example above) are compared but not written: their files in evaluation/results/
are the official, git-tracked ones that scripts/data/check_cache.py compares against.

How to read the output: first the number of graded frames inside Smart Tagging in-possession
phases (the frames the tag_* columns are computed on), then one row per method:
  H, A, D              share of graded frames labelled home, away and dead ball
  possessions          number of possession spells (maximal H or A runs within a half)
  mean_s, median_s     their mean and median length in seconds
  tag_agree            share of tagged frames where the method names the tagged team
  tag_disagree         share where it names the other team
  tag_dead             share where it says D
Smart Tagging is automatic and has errors, so tag_agree is a consistency check, not accuracy.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.methods import METHODS, OFFICIAL_BASELINES  # noqa: E402
from mtah.validation import tagging  # noqa: E402
from baseline import write_submission  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="+", choices=sorted(METHODS))
    args = parser.parse_args()

    match = load(mmap=False)
    graded = np.asarray(match.graded)
    tagged = tagging.phase_team(match)
    gt_like = np.where(graded, pm.H, pm.X)      # only to get the scorer's runs (the halves)
    runs = pm.valid_runs(gt_like)

    rows = []
    for name in args.methods:
        labels = METHODS[name](match)
        if name not in OFFICIAL_BASELINES:
            write_submission(os.path.join(RESULTS_DIR, name + ".csv"), labels)
        codes = pm.label_codes(labels)
        spells = pm.possession_spells(codes, runs)
        lengths = np.array([stop - start for start, stop, _ in spells]) / FPS
        agree = tagging.agreement(labels, tagged)
        rows.append({
            "method": name,
            "H": (labels[graded] == "H").mean(), "A": (labels[graded] == "A").mean(),
            "D": (labels[graded] == "D").mean(),
            "possessions": len(spells), "mean_s": lengths.mean(), "median_s": np.median(lengths),
            "tag_agree": agree["agree"], "tag_disagree": agree["disagree"],
            "tag_dead": agree["dead"],
        })
    print("Smart Tagging in-possession frames: {}\n".format(int((tagged != "").sum())))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200,
                           "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("method"))
