"""Run the dead-ball detector and check it without labels. Prints aggregates only.

    python scripts/dead_ball/dead_check.py

1. Held-out restarts: train WITHOUT the Smart Tagging set-piece labels, then ask whether the
   detector finds those restarts anyway: dead right after each tag start (the restart comes
   ~5 s later), alive 10-20 s after it. Reference detectors for scale: "ball missing" and
   "players slow".
2. The full detector (all confident labels): how much is dead, in how many stretches, and
   whether stretches end the way restarts do, with the players speeding up.

Writes outputs/dead/{dead,p_dead,labels}.npy for later steps (gitignored).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import features as F  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import detector as D  # noqa: E402
from mtah.utils import runs_of  # noqa: E402

# Params
FPS = 25
EARLY = (0, 3 * FPS)            # after a set-piece tag start: should be dead
LATE = (10 * FPS, 20 * FPS)     # should be alive again
SLOW = 1.0                      # m/s, reference detector "players slow" on the 3 s mean
OUT_DIR = os.path.join(ROOT, "outputs", "dead")


def restart_table(name, dead, starts):
    early = np.mean([dead[s + EARLY[0]:s + EARLY[1]].mean() for s in starts])
    late = np.mean([dead[s + LATE[0]:s + LATE[1]].mean() for s in starts])
    found = np.mean([dead[s + EARLY[0]:s + EARLY[1]].mean() > 0.5 for s in starts])
    print("  {:<34} dead early {:.2f}   dead late {:.2f}   restarts found {:.0%}".format(
        name, early, late, found))


def main():
    start_time = time.time()
    match = load(mmap=False)
    table = F.compute(match)
    graded = np.asarray(match.graded)
    print("features: {} columns in {:.0f} s".format(table.shape[1], time.time() - start_time))

    tags = match.smart_tagging
    set_pieces = tags[(tags["code"].str.upper() == "SET PIECES")
                      & ~tags["type"].str.lower().str.startswith("kick off")]
    starts = set_pieces["start_frame"].to_numpy()

    print("\n1. held-out restarts ({} set-piece tags, not used for training)".format(len(starts)))
    held_dead, _, held_labels = D.detect(match, table, use_set_pieces=False)
    print("  confident labels: dead {} frames, alive {} frames".format(
        int((held_labels == D.DEAD).sum()), int((held_labels == D.ALIVE).sum())))
    restart_table("detector (no set-piece labels)", held_dead, starts)
    restart_table("reference: ball missing", graded & (table["ball_present"].to_numpy() == 0),
                  starts)
    restart_table("reference: players slow", graded & (table["speed_mean_3s"].to_numpy() < SLOW),
                  starts)
    for name, dead in (("detector", held_dead),
                       ("ball missing", graded & (table["ball_present"].to_numpy() == 0)),
                       ("players slow", graded & (table["speed_mean_3s"].to_numpy() < SLOW))):
        print("  {:<34} dead share of graded frames {:.1%}".format(name, dead[graded].mean()))

    print("\n2. full detector")
    dead, p_dead, labels = D.detect(match, table, use_set_pieces=True)
    print("  confident labels: dead {} ({:.1%} of graded), alive {} ({:.1%})".format(
        int((labels == D.DEAD).sum()), (labels == D.DEAD)[graded].mean(),
        int((labels == D.ALIVE).sum()), (labels == D.ALIVE)[graded].mean()))
    stretches = [(a, b) for a, b in runs_of(dead)]
    lengths = np.array([b - a for a, b in stretches]) / FPS
    print("  dead {:.1%} of graded frames, {} stretches, median {:.1f} s, p90 {:.1f} s".format(
        dead[graded].mean(), len(stretches), np.median(lengths), np.percentile(lengths, 90)))
    ball = table["ball_present"].to_numpy() > 0
    print("  dead share where the ball is tracked {:.1%}, where it is missing {:.1%}".format(
        dead[graded & ball].mean(), dead[graded & ~ball].mean()))
    speed = table["speed_mean"].to_numpy()
    rises = []
    for a, b in stretches:
        if b - a >= 3 * FPS and b + 3 * FPS < len(speed):
            rises.append(np.nanmean(speed[b:b + 3 * FPS]) - np.nanmean(speed[b - 3 * FPS:b]))
    rises = np.array(rises)
    print("  stretches >= 3 s whose end is followed by players speeding up: {:.0%} "
          "(median change {:+.2f} m/s)".format((rises > 0).mean(), np.nanmedian(rises)))
    in_gap_found = np.mean([dead[s + EARLY[0]:s + EARLY[1]].mean() > 0.5 for s in starts])
    print("  tagged restarts covered: {:.0%} (they are training labels here)".format(in_gap_found))

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, values in (("dead", dead), ("p_dead", p_dead), ("labels", labels)):
        np.save(os.path.join(OUT_DIR, name + ".npy"), values)
    print("\nwritten to {} in {:.0f} s".format(os.path.relpath(OUT_DIR, ROOT),
                                               time.time() - start_time))


if __name__ == '__main__':
    main()
