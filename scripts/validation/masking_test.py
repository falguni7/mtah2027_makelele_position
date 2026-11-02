"""Masking test: how much does each method's output change when the ball is hidden?

    python scripts/validation/masking_test.py                                   # all methods, real gap lengths
    python scripts/validation/masking_test.py --methods baseline_smooth50 --by-length
    python scripts/validation/masking_test.py --reference outputs/best.csv      # fixed reference labelling

Default mode: blocks with lengths drawn from the real gap distribution, covering about
--fraction of the graded frames, repeated for --seeds seeds.
--by-length: one run per fixed block length (1 s ... 2 min), to see where a method breaks down.

Prints aggregates only. See mtah/validation/masking.py for what the numbers mean, and do not read them as
accuracy: the reference is not ground truth.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.validation import masking  # noqa: E402
from mtah.methods import METHODS  # noqa: E402

# Params
FIXED_LENGTHS = (25, 125, 250, 750, 1500, 3000)    # 1 s, 5 s, 10 s, 30 s, 1 min, 2 min


def blocks_for(match, args, rng, gap_lengths, length=None):
    if length is not None:
        count = int(args.fraction * match.graded.sum() / length)
        return masking.sample_blocks(match, [length] * max(count, 1), rng)
    target = args.fraction * match.graded.sum()
    lengths = []
    while sum(lengths) < target:
        lengths.append(int(rng.choice(gap_lengths)))
    return masking.sample_blocks(match, lengths, rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=sorted(METHODS), choices=sorted(METHODS))
    parser.add_argument("--reference", help="frame,label CSV to score against instead of each "
                                            "method's own unmasked output")
    parser.add_argument("--fraction", type=float, default=0.15,
                        help="share of graded frames to mask (default 0.15)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--by-length", action="store_true",
                        help="fixed block lengths instead of the real gap distribution")
    args = parser.parse_args()

    # Read into memory: every method touches every frame, often, and page faults on the
    # network file system cost more than the 130 MB.
    match = load(mmap=False)
    gap_lengths = masking.real_gap_lengths(match)
    print("real gaps: {}, median {:.1f} s, p90 {:.1f} s, max {:.1f} s".format(
        len(gap_lengths), np.median(gap_lengths) / match.fps,
        np.percentile(gap_lengths, 90) / match.fps, gap_lengths.max() / match.fps))
    fixed_reference = (pd.read_csv(args.reference)["label"].to_numpy()
                       if args.reference else None)

    lengths = FIXED_LENGTHS if args.by_length else (None,)
    rows = []
    for name in args.methods:
        method = METHODS[name]
        unmasked = method(match)
        reference = fixed_reference if fixed_reference is not None else unmasked
        for length in lengths:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                blocks = blocks_for(match, args, rng, gap_lengths, length)
                result = masking.score(reference, method(masking.apply_mask(match, blocks)),
                                       blocks)
                if fixed_reference is not None:
                    ceiling = masking.score(reference, unmasked, blocks)
                    result["spell_f1_unmasked"] = ceiling["spell_f1"]
                result.update(method=name, seed=seed,
                              length="real" if length is None else "{:g} s".format(
                                  length / match.fps))
                rows.append(result)

    table = pd.DataFrame(rows)
    columns = ["spell_f1", "macro_f1", "agreement", "n_blocks", "masked_frames"]
    if fixed_reference is not None:
        columns.insert(1, "spell_f1_unmasked")
    summary = table.groupby(["method", "length"], sort=False)[columns].mean()
    if args.seeds > 1:
        summary["spell_f1_sd"] = table.groupby(["method", "length"], sort=False)[
            "spell_f1"].std()
    print("\nreference: {}\n".format(args.reference or "each method's own unmasked output"))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200, "display.max_columns", 20):
        print(summary)


if __name__ == '__main__':
    main()
