"""The train / validation / test split of the tactical match for fine-tuning WASB, and its counts.

    sbatch --time=00:20:00 scripts/slurm/run.sbatch scripts/vision/wasb_split.py

mtah/vision/split.py marks the valid label frames (graded, tracking ball, valid homography,
projected ball inside the image) and splits the match: six 15-minute patches of match time,
1-minute chunks inside each patch drawn to train / validation / test at 70 / 15 / 15 of the
patch's valid frames, and a 10 s buffer around every border between different splits. The
result goes to cache/vision/wasb_split_v1.npz (gitignored: it holds per-frame ball pixels).

Printed, aggregates only: per patch, the chunks of each split and the valid frames and usable
3-frame windows of each split with their shares; the frames lost to buffers; the totals.

    patch 1   chunks 11 / 2 / 2   valid 12991 / 2783 / 2781  (0.70 / 0.15 / 0.15) ...   (e.g.)
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.utils import runs_of  # noqa: E402
from mtah.vision import split as sp  # noqa: E402

NAMES = ("train", "val", "test")

if __name__ == '__main__':
    s = sp.build()
    path = sp.save(s)
    split, patch, chunk, valid, centre = s["split"], s["patch"], s["chunk"], s["valid"], s["centre"]
    print("{:<6}{:>16}{:>24}{:>21}{:>24}{:>14}".format(
        "patch", "chunks t/v/s", "valid frames t/v/s", "shares", "windows t/v/s", "buffer lost"))
    for p in range(1, 7):
        m = patch == p
        ch = [len(np.unique(chunk[m & (split == c)])) for c in range(3)]
        val = [int((m & valid & (split == c)).sum()) for c in range(3)]
        win = [int((m & centre & (split == c)).sum()) for c in range(3)]
        tot = sum(val)
        lost = int((m & valid & (split == sp.NONE)).sum())
        print("{:<6}{:>16}{:>24}{:>21}{:>24}{:>14}".format(
            p, "/".join(map(str, ch)), "/".join(map(str, val)),
            "/".join("{:.2f}".format(v / tot) for v in val), "/".join(map(str, win)), lost))
    graded = patch > 0
    val = [int((valid & (split == c)).sum()) for c in range(3)]
    win = [int((centre & (split == c)).sum()) for c in range(3)]
    print("all: valid {} of which {} = {} ({}); lost to buffers {}; usable windows {} = {}".format(
        int(valid.sum()), "/".join(NAMES), "/".join(map(str, val)),
        "/".join("{:.3f}".format(v / sum(val)) for v in val),
        int((valid & (split == sp.NONE) & graded).sum()), int(centre.sum()), "/".join(map(str, win))))
    print("buffers (borders between different splits): {}".format(
        len(runs_of(graded & (split == sp.NONE)))))
    print("saved", path)
