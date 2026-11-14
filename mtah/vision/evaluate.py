"""Score a WASB model on 3-frame windows: the stitched heatmap's peak against the tracking ball.

    from mtah.vision import evaluate
    res = evaluate.run(net, store, centres, pixels)       # per window: score, distance in px
    evaluate.summary(res["score"], res["dist"])           # aggregate numbers

Every window goes through the fixed 4 tiles of the teammate's scheme (mtah.vision.data), the
middle frame's 4 heatmaps are stitched, and the peak is compared with the projected tracking
ball in full-resolution pixels. DataLoader workers only decode the frames; tiling, the network,
stitching and the peak run batched on the GPU. WASB's own detection threshold is THRESHOLD.

The headline number is `hit20`: the share of windows whose peak is above THRESHOLD AND within
20 px of the tracking ball (found, and in the right place). Validation picks the checkpoint by
it; the test reports it once.
"""

import numpy as np
import torch

from mtah.vision import data

# Params
THRESHOLD = 0.5
RADII = (10, 20, 50)
BATCH, WORKERS = 16, 12


@torch.no_grad()
def run(net, store, centres, pixels, batch=BATCH, workers=WORKERS, device="cuda"):
    """{score, u, v, dist} per window centre: the middle heatmap's peak and its distance (px).

        res = run(net, store, [11186], pixels)
        res["score"].shape, res["dist"].shape          # -> (1,), (1,)
    """
    net.eval()
    centres = np.asarray(centres, np.int64)
    loader = torch.utils.data.DataLoader(data.EvalWindows(store, centres), batch_size=batch,
                                         num_workers=workers, pin_memory=True)
    parts = []
    for frames in loader:
        x = data.gpu_tiles(frames.to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)[0]
        hm = torch.sigmoid(logits.float())[:, 1].reshape(len(frames), 4, data.IN_H, data.IN_W)
        parts.append(torch.stack(data.gpu_peaks(hm), 1).cpu().numpy())
    score, u, v = np.concatenate(parts).T if parts else np.zeros((3, 0))
    p = np.asarray(pixels, np.float64)[centres]
    return {"score": score, "u": u, "v": v, "dist": np.hypot(u - p[:, 0], v - p[:, 1])}


def summary(score, dist):
    """Aggregate numbers of one run: detection rate, distances, shares within RADII, hit20.

        summary(np.array([0.9, 0.9, 0.2]), np.array([5.0, 100.0, 3.0]))
        # -> {"n": 3, "detected": 0.667, "median_px": 5.0, "within10": 0.667, "within20": 0.667,
        #     "within50": 0.667, "hit10": 0.333, "hit20": 0.333, "hit50": 0.333}
    """
    score, dist = np.asarray(score), np.asarray(dist)
    det = score > THRESHOLD
    out = {"n": int(len(score)), "detected": round(float(det.mean()), 3),
           "median_px": round(float(np.median(dist)), 1)}
    for r in RADII:
        out["within{}".format(r)] = round(float((dist <= r).mean()), 3)
    for r in RADII:
        out["hit{}".format(r)] = round(float((det & (dist <= r)).mean()), 3)
    return out
