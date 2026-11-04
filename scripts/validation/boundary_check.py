"""Compare methods' dead-ball stretches and where their boundaries fall, e.g. with and without
whistles. Prints aggregates only.

    python scripts/validation/boundary_check.py hmm_v3 hmm_v4
    sbatch scripts/slurm/run.sbatch scripts/validation/boundary_check.py hmm_v3 hmm_v4

A dead-ball stretch is a run of D inside a graded half; its start is where play stops and its
end where play restarts. Possessions are split by these stretches, so where they start and
end matters for spellF1. For each method this prints:

  D share          share of graded frames labelled D
  stretches        number of D stretches, and their median length in seconds
  possessions      number of possession spells (maximal H or A runs, as the scorer counts)
  start at whistle share of stretch starts with a referee whistle starting within -1 to +1 s
                   of it (the window in which mtah/temporal/hmm.py makes entering D cheaper),
                   and the same share at random graded frames ("chance")
  end at whistle   the same for stretch ends, whistle within 0 to 3 s before the end

Then, for every method after the first, against the first one:

  frames changed   graded frames whose label differs
  stretches        matched (overlapping a stretch of the first method), new, gone
  start / end moved  for matched stretches whose start (end) moved: how many, and the median
                   and largest move in seconds (negative = earlier than in the first method)

The whistles come from mtah/audio/whistle.py and need the extracted sound
(scripts/data/extract_audio.py). Note: for a method that USES the whistles (hmm_v4), the
"at whistle" shares are no independent check -- they show that the mechanism works, not that
the boundaries are right.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.methods import METHODS  # noqa: E402
from mtah.temporal.hmm import WHISTLE_ENTER, WHISTLE_LEAVE  # noqa: E402
from mtah.utils import runs_of  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
N_RANDOM = 3000


def dead_stretches(labels, graded):
    """[(start, stop)] runs of "D" inside the graded halves (frames, half-open).

        labels = ["H", "D", "D", "A", "D"], graded = [T, T, T, T, F]
        dead_stretches(labels, graded)     # -> [(1, 3)]   frame 4 is not graded
    """
    return runs_of((np.asarray(labels) == "D") & np.asarray(graded, dtype=bool))


def near_whistle(frames, whistle_starts, before, after):
    """Share of frames with a whistle starting in [frame/25 + before, frame/25 + after] seconds.

        frames = [250, 1000], whistle_starts = [10.5]      # a whistle at 10.5 s
        near_whistle(frames, whistle_starts, -1, 1)         # -> 0.5   frame 250 is at 10.0 s
    """
    starts = np.sort(np.asarray(whistle_starts, dtype=np.float64))
    if not len(frames):
        return np.nan
    t = np.asarray(frames) / FPS
    lo = np.searchsorted(starts, t + before, side="left")
    hi = np.searchsorted(starts, t + after, side="right")
    return float(np.mean(hi > lo))


def match_stretches(base, other):
    """(pairs, new, gone): stretches of other matched to overlapping stretches of base.

    pairs is [(base_stretch, other_stretch)] for stretches that overlap (the first overlap of
    each); new counts other's stretches overlapping none of base's, gone base's overlapping none
    of other's.

        base  = [(10, 20), (50, 60)]
        other = [(8, 20), (90, 95)]
        match_stretches(base, other)   # -> ([((10, 20), (8, 20))], 1, 1)
    """
    pairs, used = [], set()
    for b in base:
        for i, o in enumerate(other):
            if i not in used and o[0] < b[1] and b[0] < o[1]:
                pairs.append((b, o))
                used.add(i)
                break
    gone = len(base) - len(pairs)
    return pairs, len(other) - len(used), gone


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("methods", nargs="+", choices=sorted(METHODS))
    args = parser.parse_args()

    match = load(mmap=False)
    graded = np.asarray(match.graded)
    whistle_starts = [e[0] for e in match_whistles()]
    random_frames = np.random.default_rng(0).choice(np.flatnonzero(graded), N_RANDOM,
                                                    replace=False)
    # the windows are relative to the whistle; seen from the boundary they flip sign
    enter_window = (-WHISTLE_ENTER[1], -WHISTLE_ENTER[0])
    leave_window = (-WHISTLE_LEAVE[1], -WHISTLE_LEAVE[0])
    chance_start = near_whistle(random_frames, whistle_starts, *enter_window)
    chance_end = near_whistle(random_frames, whistle_starts, *leave_window)
    gt_like = np.where(graded, pm.H, pm.X)
    runs = pm.valid_runs(gt_like)
    print("whistles: {}; chance of one in the start window {:.3f}, end window {:.3f}\n".format(
        len(whistle_starts), chance_start, chance_end))

    results = {}
    print("{:<10} {:>7} {:>10} {:>9} {:>12} {:>16} {:>16}".format(
        "method", "D share", "stretches", "median s", "possessions", "start at whistle",
        "end at whistle"))
    for name in args.methods:
        labels = METHODS[name](match)
        stretches = dead_stretches(labels, graded)
        results[name] = (labels, stretches)
        spells = pm.possession_spells(pm.label_codes(labels), runs)
        lengths = np.array([b - a for a, b in stretches]) / FPS
        print("{:<10} {:7.3f} {:10d} {:9.1f} {:12d} {:16.3f} {:16.3f}".format(
            name, (labels[graded] == "D").mean(), len(stretches),
            np.median(lengths) if len(lengths) else np.nan, len(spells),
            near_whistle([a for a, _ in stretches], whistle_starts, *enter_window),
            near_whistle([b for _, b in stretches], whistle_starts, *leave_window)))

    base_name = args.methods[0]
    base_labels, base_stretches = results[base_name]
    for name in args.methods[1:]:
        labels, stretches = results[name]
        pairs, new, gone = match_stretches(base_stretches, stretches)
        print("\n{} against {}: frames changed {} ({:.2%} of graded)".format(
            name, base_name, int((labels[graded] != base_labels[graded]).sum()),
            (labels[graded] != base_labels[graded]).mean()))
        print("  stretches: matched {}, new {}, gone {}".format(len(pairs), new, gone))
        for side, index in (("start", 0), ("end", 1)):
            moves = np.array([o[index] - b[index] for b, o in pairs if o[index] != b[index]]) / FPS
            if len(moves):
                print("  {} moved: {} stretches, median {:+.1f} s, largest {:+.1f} s".format(
                    side, len(moves), np.median(moves), moves[np.argmax(np.abs(moves))]))
            else:
                print("  {} moved: none".format(side))
