"""Fine-tune WASB on the training chunks of the tactical match; pick the checkpoint on validation.

    sbatch --time=06:00:00 scripts/slurm/run.sbatch scripts/vision/wasb_finetune.py --radius 5
    sbatch --time=00:40:00 scripts/slurm/run.sbatch scripts/vision/wasb_finetune.py --smoke

Steps, all inside one job (frames only ever on the node's $TMPDIR):

  1. the split (mtah/vision/split.py, cache/vision/wasb_split_v1.npz): usable 3-frame windows of
     the TRAIN and VAL chunks, labels = tracking ball projected into the image;
  2. one pass over the tactical video writes every frame those windows need at half resolution
     to $TMPDIR (mtah/vision/frames.py; ~67,000 frames, ~25 min);
  3. validation of the PRETRAINED weights on a fixed seeded subset of VAL_WINDOWS validation
     windows (the baseline), with the evaluation tiles used later on the gaps;
  4. training from the pretrained weights: random crops with scale jitter, flips and brightness /
     contrast jitter (mtah/vision/data.py), binary disk targets of radius --radius network px,
     WASB's wbce loss on all 3 heatmaps, Adam with a short warm-up and cosine decay, bfloat16
     autocast; windows drawn at random with replacement;
  5. every VAL_EVERY iterations the same validation, and the weights saved as iter_<N>.pt in the
     run folder cache/vision/runs/<run>/ (gitignored; the last one is the end of training);
     the one with the best hit20 (peak above 0.5 and within 20 px of the tracking ball,
     mtah/vision/evaluate.py) is also copied to best.pt (iteration 0 = the pretrained weights);
  6. at the end, best.pt and the pretrained weights on ALL validation windows.

--radius: the target disk (WASB's default 2.5). With 2.5 the first full run (job 4984740) placed
its peaks much better than the pretrained model but rarely above 0.5: the tracking labels
scatter by several pixels around the ball, more than a 2.5 px disk allows, so a wider disk (5)
lets the model be confident again. --run names the folder (default r<radius>).

The test chunks are not touched here: scripts/vision/wasb_test.py scores them once. Printed:
counts, losses, timings and aggregate validation numbers only.

--smoke: windows of patch 1 only, 60 iterations, 200 validation windows every 30, run folder
cache/vision/runs/smoke. A pipeline check, nothing is chosen from it.
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import asset_paths  # noqa: E402
from mtah.vision import data, evaluate, frames, model, split  # noqa: E402

# Params
ITERS, BATCH, LR, WARMUP = 20000, 32, 1e-4, 500
VAL_EVERY, VAL_WINDOWS = 1000, 3000
WORKERS, SEED = 12, 0


def lr_at(it, iters, lr=LR, warmup=WARMUP):
    """Learning rate at iteration `it`: linear warm-up, then cosine decay to 0.

        lr_at(0, 20000), lr_at(500, 20000), lr_at(20000, 20000)   # -> 2e-07, 1e-04, 0.0
    """
    if it < warmup:
        return lr * (it + 1) / warmup
    t = (it - warmup) / max(1, iters - warmup)
    return lr * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


def validate(net, store, centres, pixels, label):
    """Run evaluate on the windows, print the summary with a label, return it."""
    t = time.time()
    res = evaluate.run(net, store, centres, pixels)
    s = evaluate.summary(res["score"], res["dist"])
    print("  val {:<22} {}  ({:.0f} s)".format(label, s, time.time() - t), flush=True)
    return s


@torch.no_grad()
def check_targets(net, ds, radius, n=200):
    """Smoke check of the augmentation geometry: on n random training crops, where the
    pretrained model is confident (peak > 0.5), its peak should sit on the target disk.
    Prints the share of confident crops and the median peak-to-target distance (network px)."""
    net.eval()
    d = []
    for i in np.random.default_rng(1).choice(len(ds), n, replace=False):
        crops, balls = ds[int(i)]
        y = data.gpu_targets(balls[None].cuda(), r=radius)[0, 1]
        if y.sum() == 0:
            continue
        hm = torch.sigmoid(net(data.gpu_inputs(crops[None].cuda()))[0][0, 1].float())
        if hm.max() > evaluate.THRESHOLD:
            pv, pu = np.unravel_index(int(hm.argmax()), hm.shape)
            tv, tu = np.argwhere(y.cpu().numpy() > 0).mean(axis=0)
            d.append(np.hypot(pu - tu, pv - tv))
    d = np.array(d)
    print("  target check: {} confident crops with a ball; peak-to-target px median {:.1f}, "
          "within 3 px {:.2f}".format(len(d), np.median(d) if len(d) else np.nan,
                                      (d <= 3).mean() if len(d) else np.nan), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--radius", type=float, default=data.RADIUS)
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    iters, val_every, val_n = args.iters, VAL_EVERY, VAL_WINDOWS
    run = "smoke" if args.smoke else (args.run or "r{:g}".format(args.radius))
    if args.smoke:
        iters, val_every, val_n = 60, 30, 200
    run_dir = os.path.join(split.CACHE, "runs", run)
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, "best.pt")
    params = {"iters": iters, "batch": BATCH, "lr": LR, "radius": args.radius}
    print("run {}: {}".format(run, params), flush=True)
    torch.manual_seed(SEED)
    started = time.time()

    s = split.load()
    usable = s["centre"]
    if args.smoke:
        usable = usable & (s["patch"] == 1)
    train_c = np.flatnonzero(usable & (s["split"] == split.TRAIN))
    val_all = np.flatnonzero(usable & (s["split"] == split.VAL))
    rng = np.random.default_rng(SEED)
    val_c = np.sort(rng.choice(val_all, min(val_n, len(val_all)), replace=False))
    print("windows: train {}, validation {} (subset {} for selection)".format(
        len(train_c), len(val_all), len(val_c)), flush=True)

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "wasb_frames")
    need = np.unique(np.concatenate([np.concatenate([c - 1, c, c + 1])
                                     for c in (train_c, val_all)]))
    print("extracting {} frames to {} (free {:.0f} GB)".format(
        len(need), tmp, shutil.disk_usage(os.path.dirname(tmp)).free / 1e9), flush=True)
    t = time.time()
    frames.extract(asset_paths(source="tactical")["video"], need, tmp)
    print("extracted in {:.0f} s, {:.1f} GB".format(
        time.time() - t, os.path.getsize(os.path.join(tmp, "frames.bin")) / 1e9), flush=True)
    store = frames.FrameStore(tmp)
    pixels = s["pixel"].astype(np.float64)

    dev = "cuda"
    net = model.build().to(dev)
    print("pretrained baseline:", flush=True)
    base = validate(net, store, val_c, pixels, "pretrained (subset)")
    best, best_it = base["hit20"], 0
    torch.save({"model_state_dict": net.state_dict(), "iter": 0, "val": base, "params": params},
               out_path)

    ds = data.TrainWindows(store, train_c, pixels, augment=True, seed=SEED)
    if args.smoke:
        check_targets(net, ds, args.radius)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=iters * BATCH,
                                             generator=torch.Generator().manual_seed(SEED))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, sampler=sampler,
                                         num_workers=WORKERS, worker_init_fn=data.worker_init,
                                         pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    losses, t, since = [], time.time(), 0
    for it, (crops, balls) in enumerate(loader):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, iters)
        x = data.gpu_inputs(crops.to(dev, non_blocking=True))
        y = data.gpu_targets(balls.to(dev, non_blocking=True), r=args.radius)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)[0]
        loss = model.wbce(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        done, since = it + 1, since + 1
        if done % 200 == 0 or done == iters:
            print("iter {:6d}  loss {:.5f}  lr {:.2e}  {:.1f} it/s".format(
                done, np.mean(losses[-since:]), lr_at(it, iters), since / (time.time() - t)),
                flush=True)
            t, since = time.time(), 0
        if done % val_every == 0 or done == iters:
            v = validate(net, store, val_c, pixels, "iter {}".format(done))
            ckpt = {"model_state_dict": net.state_dict(), "iter": done, "val": v, "params": params}
            torch.save(ckpt, os.path.join(run_dir, "iter_{:05d}.pt".format(done)))
            if v["hit20"] > best:
                best, best_it = v["hit20"], done
                torch.save(ckpt, out_path)
                print("  -> best so far (hit20 {:.3f})".format(best), flush=True)
            net.train()
            t = time.time()                    # the speed print leaves validation out

    print("\nbest validation hit20 {:.3f} at iteration {} (pretrained {:.3f}); saved {}".format(
        best, best_it, base["hit20"], out_path), flush=True)
    print("all {} validation windows:".format(len(val_all)), flush=True)
    validate(model.build().to(dev), store, val_all, pixels, "pretrained")
    validate(model.build(out_path).to(dev), store, val_all, pixels, "fine-tuned (kept)")
    print("total {:.0f} min".format((time.time() - started) / 60))
