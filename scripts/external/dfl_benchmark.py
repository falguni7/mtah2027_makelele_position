"""The whole method on the 7 DFL matches with our ball gaps, and HMM weights chosen on them.

    sbatch --time=02:00:00 scripts/slurm/run.sbatch scripts/external/dfl_benchmark.py
    python scripts/external/dfl_benchmark.py --quick     # 1 match, 1 placement, 2 weight sets
    python scripts/external/dfl_benchmark.py --wide      # GRID_WIDE: past GRID's edges

mtah/external/benchmark.py makes each DFL match look like ours (degraded players, the ball
hidden in gaps like ours, "random" or "dead" placement, no sound, camera or Smart Tagging) and
runs the core of our method on it with each variant of benchmark.VARIANTS. Every variant is
decoded with every weight set of GRID (all combinations of the four HMM weights) and scored
against DFL's labels. Aggregates only are printed; DFL is public data, and the per-match scores
are saved to cache/external/dfl/benchmark_v1.npz (benchmark_v1_wide.npz with --wide).

Tables:

  truth           the DFL labels' own possessions per match and mean length, for scale
  today           every variant with today's weights (benchmark.DEFAULT): mean over the 7
                  matches of spellF1, macroF1, possessions, mean length; and the detector's D AUC
  chosen          leave-one-match-out: for each match, the weight set with the best mean
                  spellF1 on the other 6 (macroF1 breaks ties) is scored on it; the mean over
                  the 7 is an honest estimate of what choosing weights on DFL gives. "wins" counts
                  the matches where it beats today's weights
  best on all 7   the weight set chosen on all 7 matches, per variant and placement: the one
                  to try on our match (with the label-free checks there, never the public window)

Each (match, placement) is one worker process: it builds the evidence of every variant once
(the dead-ball detector and our shape model train in the match, about a minute) and decodes it
with every weight set (Viterbi, about a second each).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"     # one core per worker; set before numpy / sklearn load

import argparse  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.external import benchmark as bm  # noqa: E402
from mtah.external import dfl  # noqa: E402

# Params
GRID = {"TEAM_WEIGHT": (0.0, 0.025, 0.05, 0.1, 0.2, 0.4),
        "DEAD_WEIGHT": (0.05, 0.1, 0.2, 0.4),
        "TEAM_SWITCH": (1.0, 2.0, 4.0),
        "DEAD_SWITCH": (2.0, 4.0, 8.0)}
# The first run chose weights at three of GRID's edges (TEAM_WEIGHT 0.4, DEAD_WEIGHT 0.05,
# TEAM_SWITCH 4); this grid reaches past them and keeps the values chosen there.
GRID_WIDE = {"TEAM_WEIGHT": (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
             "DEAD_WEIGHT": (0.0125, 0.025, 0.05, 0.1),
             "TEAM_SWITCH": (2.0, 4.0, 8.0, 16.0),
             "DEAD_SWITCH": (2.0, 4.0, 8.0)}
KEYS = ("spellF1", "macroF1", "possessions", "mean_s")
OUT = os.path.join(dfl.CACHE, "benchmark_v1{}.npz")      # "" or "_wide"


def weight_sets(grid=GRID):
    """Every combination of the grid's values as a list of dicts, today's weights first.

        grid = {"TEAM_WEIGHT": (0.05, 0.2), "DEAD_WEIGHT": (0.1,)}
        weight_sets(grid)
        # -> [DEFAULT, {"TEAM_WEIGHT": 0.05, "DEAD_WEIGHT": 0.1},
        #              {"TEAM_WEIGHT": 0.2, "DEAD_WEIGHT": 0.1}]
        # (DEFAULT also carries TEAM_SWITCH 2 and DEAD_SWITCH 4)
    """
    names = list(grid)
    return [dict(bm.DEFAULT)] + [dict(zip(names, values))
                                 for values in itertools.product(*grid.values())]


def run_one(job):
    """Scores of every variant x weight set on one (match, placement): an array and D AUCs.

    job is (match_id, placement, ctx, sets). Returns (match_id, placement, scores, d_auc) with
    scores of shape (variants, weight sets, len(KEYS)) and d_auc a list over the variants.
    """
    match_id, placement, ctx, sets = job
    started = time.time()
    match, truth = bm.bench_match(match_id, ctx, placement)
    out = np.zeros((len(bm.VARIANTS), len(sets), len(KEYS)))
    d_auc = []
    for v, variant in enumerate(bm.VARIANTS):
        ev = bm.evidence(match, truth, variant)
        d_auc.append(ev["d_auc"])
        for s, values in enumerate(sets):
            row = bm.score(truth.labels, bm.decode(match, ev, values), truth.half)
            out[v, s] = [row[k] for k in KEYS]
    print("{} {:<6} done in {:.0f} s".format(match_id, placement, time.time() - started),
          flush=True)
    return match_id, placement, out, d_auc


def choose(spell, macro):
    """Index of the best weight set: highest spellF1, macroF1 breaking (near) ties.

    spell and macro are per-weight-set means. Ties are spellF1 within 1e-9.

        choose(np.array([0.5, 0.6, 0.6]), np.array([0.9, 0.7, 0.8]))     # -> 2
    """
    best = np.flatnonzero(spell >= spell.max() - 1e-9)
    return int(best[np.argmax(macro[best])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="1 match, random placement, 2 weight sets")
    parser.add_argument("--wide", action="store_true", help="GRID_WIDE instead of GRID")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    ctx = bm.context()
    print("our match: ball missing on {:.3f} of graded frames, {} gaps, mean {:.1f} s".format(
        ctx["ball_share"], len(ctx["ball_lengths"]), ctx["ball_lengths"].mean() / 25),
        flush=True)
    matches, placements = list(dfl.MATCHES), list(bm.PLACEMENTS)
    sets = weight_sets(GRID_WIDE if args.wide else GRID)
    if args.quick:
        matches, placements, sets = matches[:1], placements[:1], sets[:2]
    jobs = [(m, p, ctx, sets) for p in placements for m in matches]
    with Pool(min(args.workers, len(jobs))) as pool:
        done = pool.map(run_one, jobs, chunksize=1)

    variants = list(bm.VARIANTS)
    scores = np.zeros((len(placements), len(matches), len(variants), len(sets), len(KEYS)))
    d_auc = np.zeros((len(placements), len(matches), len(variants)))
    for match_id, placement, out, auc in done:
        p, m = placements.index(placement), matches.index(match_id)
        scores[p, m], d_auc[p, m] = out, auc
    if not args.quick:
        np.savez(OUT.format("_wide" if args.wide else ""), scores=scores, d_auc=d_auc, matches=matches, placements=placements,
                 variants=variants, keys=KEYS,
                 sets=np.array([[s[w] for w in bm.WEIGHTS] for s in sets]))

    print("\ntruth: possessions per match / mean s")
    for match_id in matches:
        _, truth = bm.bench_match(match_id, ctx, "random")
        row = bm.score(truth.labels, truth.labels, truth.half)
        print("   {}  {:4d}  {:5.1f}".format(match_id, row["possessions"], row["mean_s"]))

    S, M = KEYS.index("spellF1"), KEYS.index("macroF1")
    for p, placement in enumerate(placements):
        print("\n== ball placement: {} ==".format(placement))
        print("today's weights {}".format(bm.DEFAULT))
        print("{:<14}{:>10}{:>10}{:>8}{:>8}{:>8}".format("variant", "spellF1", "macroF1",
                                                        "poss", "mean s", "D AUC"))
        for v, variant in enumerate(variants):
            mean = scores[p, :, v, 0].mean(axis=0)
            print("{:<14}{:>10.3f}{:>10.3f}{:>8.0f}{:>8.1f}{:>8.3f}".format(
                variant, *mean, d_auc[p, :, v].mean()))
        if len(sets) < 2 or len(matches) < 2:
            continue
        print("chosen (leave one match out) / best on all {}".format(len(matches)))
        print("{:<14}{:>10}{:>10}{:>8}{:>6}   {}".format("variant", "spellF1", "macroF1",
                                                        "poss", "wins", "best on all: "
                                                        + " ".join(bm.WEIGHTS)))
        for v, variant in enumerate(variants):
            test = []
            for m in range(len(matches)):
                rest = [j for j in range(len(matches)) if j != m]
                c = choose(scores[p, rest, v, :, S].mean(axis=0),
                           scores[p, rest, v, :, M].mean(axis=0))
                test.append(scores[p, m, v, c])
            test = np.array(test)
            wins = int((test[:, S] > scores[p, :, v, 0, S]).sum())
            best = choose(scores[p, :, v, :, S].mean(axis=0), scores[p, :, v, :, M].mean(axis=0))
            print("{:<14}{:>10.3f}{:>10.3f}{:>8.0f}{:>6}   {}  (all-7 spellF1 {:.3f})".format(
                variant, test[:, S].mean(), test[:, M].mean(), test[:, 2].mean(),
                "{}/{}".format(wins, len(matches)),
                " ".join("{:g}".format(sets[best][w]) for w in bm.WEIGHTS),
                scores[p, :, v, best, S].mean()))
