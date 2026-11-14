"""Score the pretrained and the fine-tuned WASB once on the test chunks of the tactical match.

    sbatch --time=01:30:00 scripts/slurm/run.sbatch scripts/vision/wasb_test.py
    python scripts/vision/wasb_test.py --checkpoint cache/vision/wasb_smoke.pt --smoke

Run this ONCE, after training is settled: the test chunks (mtah/vision/split.py) are never used
for any choice. Both models see exactly the same test windows, through the evaluation tiles of
mtah/vision/evaluate.py; frames go to the node's $TMPDIR only.

Printed, aggregates only (mtah.vision.evaluate.summary: detected, median px, within / hit at
10, 20, 50 px), for both models on:

  all        every usable test window
  near gap   windows whose middle frame is within 1 s of a graded frame without a tracking ball:
             the hardest labelled frames, closest to the gaps we want to fill
  fast       tracking ball faster than 15 m/s at the middle frame (likely in the air, where the
             ground-projected label may be off)
  slow       slower than 5 m/s
  patch k    each 15-minute patch

--smoke: test windows of patch 1 only, at most 200.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from mtah.data import io  # noqa: E402
from mtah.vision import evaluate, frames, model, split  # noqa: E402

# Params
FPS, PITCH = 25, (105.3, 68.0)
NEAR_GAP = FPS
FAST, SLOW = 15.0, 5.0


def near_gap(no_ball, graded, reach=NEAR_GAP):
    """bool per frame: within `reach` frames of a graded frame without a tracking ball.

        near_gap([F, F, F, T, F, F, F], [T] * 7, 1)    # -> [F, F, T, T, T, F, F]
    """
    gap = np.asarray(no_ball, bool) & np.asarray(graded, bool)
    out = gap.copy()
    for k in range(1, reach + 1):
        out[k:] |= gap[:-k]
        out[:-k] |= gap[k:]
    return out


def ball_speed(ball):
    """m/s per frame from the tracking ball (normalised x, y), centred difference; NaN if missing.

        ball_speed(np.array([[0.0, 0.5], [0.01, 0.5], [0.02, 0.5]]))   # -> [nan, 26.3, nan]
    """
    m = np.asarray(ball, np.float64) * PITCH
    out = np.full(len(m), np.nan)
    out[1:-1] = np.hypot(*(m[2:] - m[:-2]).T) * FPS / 2
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", default=model.FINETUNED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    s = split.load()
    usable = s["centre"] & (s["split"] == split.TEST)
    if args.smoke:
        usable &= s["patch"] == 1
    test_c = np.flatnonzero(usable)
    if args.smoke:
        test_c = test_c[:: max(1, len(test_c) // 200)][:200]
    print("test windows {}; checkpoint {} (iteration {})".format(
        len(test_c), args.checkpoint, torch.load(args.checkpoint, map_location="cpu",
                                                 weights_only=False).get("iter")), flush=True)

    _, ball = io.read_atd_raw()
    graded = s["patch"] > 0
    groups = {"all": np.ones(len(test_c), bool),
              "near gap": near_gap(~np.isfinite(ball[:, 0]), graded)[test_c],
              "fast": ball_speed(ball)[test_c] > FAST,
              "slow": ball_speed(ball)[test_c] < SLOW}
    for p in range(1, 7):
        groups["patch {}".format(p)] = s["patch"][test_c] == p

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wasb_test_frames")
    t = time.time()
    frames.extract(asset_paths(source="tactical")["video"],
                   np.concatenate([test_c - 1, test_c, test_c + 1]), tmp)
    print("extracted in {:.0f} s".format(time.time() - t), flush=True)
    store = frames.FrameStore(tmp)
    pixels = s["pixel"].astype(np.float64)

    res = {}
    for name, weights in (("pretrained", model.PRETRAINED), ("fine-tuned", args.checkpoint)):
        res[name] = evaluate.run(model.build(weights).to("cuda"), store, test_c, pixels)
    for g, m in groups.items():
        if not m.any():
            continue
        print("\n{} ({} windows)".format(g, int(m.sum())))
        for name, r in res.items():
            print("  {:<11} {}".format(name, evaluate.summary(r["score"][m], r["dist"][m])))
