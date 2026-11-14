"""Run the fine-tuned WASB on every graded frame with a tracking ball and save its peak per frame.

    sbatch --time=03:00:00 scripts/slurm/run.sbatch scripts/vision/wasb_ball.py
    python scripts/vision/wasb_ball.py --smoke                     # 300 windows, a pipeline check
    sbatch --time=02:00:00 scripts/slurm/run.sbatch scripts/vision/wasb_ball.py --gaps

Frames: graded, tracking ball present and homography valid on the tactical match (the frames
mtah/vision/ball.py may correct; about 94,000), each as the middle of a 3-frame window. All
three frames are decoded once at half resolution into the node's $TMPDIR (mtah/vision/frames.py),
then scored exactly as the test was (mtah/vision/evaluate.py: the 4 evaluation tiles, the middle
frame's stitched heatmap, its peak in full-resolution pixels).

Writes mtah.vision.ball.CACHE (cache/vision/wasb_ball_v<VERSION>.npz, gitignored, never opened by
Claude): frame, score, u, v per window, plus the checkpoint path and its iteration. Printed:
counts, timing and the share of peaks above 0.5 only.

The model is the one chosen on validation and tested once: cache/vision/runs/r5/best.pt
(radius 5, iteration 1000). Frames from the training chunks are included; there the model leans
towards the tracking label, so it corrects less than on unseen frames.

--smoke: the first 300 such windows, written to cache/vision/wasb_ball_smoke.npz instead.
--gaps: the frames where the tracking ball is MISSING instead (mtah.vision.ball.gap_frames: graded,
no tracking ball, homography valid; 64,643 frames), written to mtah.vision.ball.GAPS_CACHE with the
same keys, for the gap-filling count. Combined with --smoke: the first 300, to
cache/vision/wasb_gaps_smoke.npz.
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
from mtah.data import load  # noqa: E402
from mtah.vision import ball, evaluate, frames, model  # noqa: E402


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gaps", action="store_true")
    args = ap.parse_args()
    started = time.time()
    match = load()
    if args.gaps:
        centres, out, kind = ball.gap_frames(match), ball.GAPS_CACHE, "NO tracking ball"
    else:
        centres, out, kind = ball.candidate_frames(match), ball.CACHE, "tracking ball"
    if args.smoke:
        centres, out = centres[:300], out.replace("_v{}".format(ball.VERSION), "_smoke")
    print("windows {} (graded, {}, valid homography); checkpoint {}".format(
        len(centres), kind, ball.CHECKPOINT), flush=True)

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wasb_ball_frames")
    t = time.time()
    frames.extract(asset_paths(source="tactical")["video"],
                   np.concatenate([centres - 1, centres, centres + 1]), tmp)
    print("extracted in {:.0f} s".format(time.time() - t), flush=True)
    store = frames.FrameStore(tmp)

    t = time.time()
    net = model.build(ball.CHECKPOINT).to("cuda")
    pixels = np.zeros((match.n_frames, 2))              # unused: only score, u, v are kept
    res = evaluate.run(net, store, centres, pixels)
    print("inference in {:.0f} s ({:.0f} windows/s)".format(
        time.time() - t, len(centres) / (time.time() - t)), flush=True)

    ck = torch.load(ball.CHECKPOINT, map_location="cpu", weights_only=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, version=ball.VERSION, frame=centres, score=res["score"].astype(np.float32),
             u=res["u"].astype(np.float32), v=res["v"].astype(np.float32),
             checkpoint=os.path.relpath(ball.CHECKPOINT, ROOT), iteration=ck.get("iter"))
    print("peaks above {}: {:.3f}; saved {}; total {:.0f} min".format(
        evaluate.THRESHOLD, (res["score"] > evaluate.THRESHOLD).mean(), out,
        (time.time() - started) / 60), flush=True)
