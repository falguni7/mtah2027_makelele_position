"""WASB inputs and targets: training crops with augmentation, and the fixed tiles for evaluation.

    from mtah.vision import data
    ds = data.TrainWindows(store, centres, pixels)       # torch Dataset of random crops
    crops, balls = ds[0]                                 # (3, 288, 512, 3) uint8, (3, 2) ball px
    x = data.gpu_inputs(crops[None].cuda())              # (1, 9, 288, 512) network input
    y = data.gpu_targets(balls[None].cuda())             # (1, 3, 288, 512) binary disks
    score, u, v = data.gpu_peaks(tile_heatmaps)          # 4 fixed tiles -> stitched peak

Workers only decode, warp and jitter uint8 images; normalising, building the targets, tiling,
stitching and finding the peak run batched on the GPU (gpu_*). The numpy versions (normalise,
binary_target, tiles, stitch, peak) are the same arithmetic for one window, kept as reference.

Coordinates. Labels come in full-resolution pixels (u, v) with pixel centres at integers
(mtah.vision.split.ball_pixels). In the half-resolution frame (mtah.vision.frames) pixel i covers
full pixels 2i and 2i + 1, so a point maps as u_half = (u + 0.5) / 2 - 0.5 (to_half / to_full).

Input, as WASB: 3 consecutive frames, oldest first, each RGB / 255 normalised with ImageNet mean
and std, stacked into 9 channels. Target, as WASB's default (`binary_fixed_size`, sigma 2.5): per
frame a disk of radius RADIUS px at the ball, zero when the ball is outside the crop.

Training crops (TrainWindows): from the half-resolution frames, a 512x288 network input covering
a region of k times that size (k uniform in SCALE: scale jitter), placed so that the middle
frame's ball is inside with probability P_BALL (otherwise anywhere: crops without the ball teach
the model to stay quiet on heads, boots and lines), mirrored left-right with probability P_FLIP,
and brightness / contrast jittered by up to JITTER. The same transform applies to all 3 frames.

Evaluation tiles (tiles, stitch): the teammate's scheme exactly, i.e. 4 fixed 512x288 crops of
the half-resolution frame at TILE_ORIGINS (their 1024x576 full-resolution tiles at x 0 / 896,
y 0 / 504, halved); the 4 heatmaps are stitched back with the maximum where tiles overlap.
"""

import cv2
import numpy as np
import torch

# Params
IN_W, IN_H = 512, 288
HALF_W, HALF_H = 960, 540
RADIUS = 2.5
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TILE_ORIGINS = [(0, 0), (448, 0), (0, 252), (448, 252)]       # (x, y) in the half frame
SCALE = (0.8, 1.25)
P_BALL, P_FLIP, JITTER = 0.8, 0.5, 0.2
MARGIN = 8                          # px (network scale): a "ball" crop keeps the ball this far in


def to_half(uv):
    """Full-resolution pixel coordinates -> half-resolution ones.

        to_half(np.array([0.5, 1919.0]))      # -> [0.0, 959.25]   (full pixels 0-1 -> half pixel 0)
    """
    return (np.asarray(uv, np.float64) + 0.5) / 2 - 0.5


def to_full(uv):
    """Half-resolution pixel coordinates -> full-resolution ones (inverse of to_half).

        to_full(np.array([0.0, 959.25]))      # -> [0.5, 1919.0]
    """
    return (np.asarray(uv, np.float64) + 0.5) * 2 - 0.5


def binary_target(cx, cy, w=IN_W, h=IN_H, r=RADIUS):
    """(h, w) float32: 1 within r px of (cx, cy), else 0; all 0 when the point is NaN or outside.

    Same as WASB's utils.heatmap.gen_binary_map (pixel centres at integers).

        t = binary_target(10, 5)
        t.sum(), t[5, 10], t[5, 13]           # -> 21.0, 1.0, 0.0     (21 pixels within 2.5 px)
        binary_target(np.nan, 5).sum()        # -> 0.0
    """
    if not (np.isfinite(cx) and np.isfinite(cy)) or not (-r <= cx < w + r and -r <= cy < h + r):
        return np.zeros((h, w), np.float32)
    y, x = np.mgrid[0:h, 0:w]
    return (((x - cx) ** 2 + (y - cy) ** 2) <= r * r).astype(np.float32)


def normalise(frames):
    """3 RGB uint8 (H, W, 3) frames, oldest first -> (9, H, W) float32 WASB input.

        normalise([np.full((2, 2, 3), 255, np.uint8)] * 3)[0, 0, 0]   # -> (1 - 0.485) / 0.229 = 2.249
    """
    out = [((f.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1) for f in frames]
    return np.ascontiguousarray(np.concatenate(out))


def crop_transform(rng, ball, scale=SCALE, p_ball=P_BALL):
    """2x3 affine from half-frame coordinates to a 512x288 crop; ball is the middle frame's (u, v).

    The crop covers k * (512, 288) half-frame pixels, k uniform in `scale`, inside the frame.
    With probability p_ball (and a ball given) it is placed so the ball lies at least MARGIN
    network pixels inside; otherwise anywhere.

        M = crop_transform(np.random.default_rng(0), (480.0, 270.0))
        x, y = M @ [480, 270, 1]
        MARGIN <= x <= 512 - MARGIN and MARGIN <= y <= 288 - MARGIN     # -> True (usually)
    """
    k = rng.uniform(*scale)
    cw, ch = IN_W * k, IN_H * k
    x_lo, x_hi, y_lo, y_hi = 0.0, HALF_W - cw, 0.0, HALF_H - ch
    if ball is not None and np.all(np.isfinite(ball)) and rng.uniform() < p_ball:
        m = MARGIN * k
        x_lo, x_hi = max(x_lo, ball[0] - cw + m), min(x_hi, ball[0] - m)
        y_lo, y_hi = max(y_lo, ball[1] - ch + m), min(y_hi, ball[1] - m)
        if x_lo > x_hi or y_lo > y_hi:             # ball too close to the frame edge
            x_lo, x_hi, y_lo, y_hi = 0.0, HALF_W - cw, 0.0, HALF_H - ch
    ox, oy = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
    return np.array([[1 / k, 0, -ox / k], [0, 1 / k, -oy / k]], np.float64)


class TrainWindows(torch.utils.data.Dataset):
    """Random training crops of 3-frame windows: item i -> (crops, balls).

    crops is (3, 288, 512, 3) uint8 RGB (already jittered), balls (3, 2) float32 the ball's
    position in each crop (network pixels; may lie outside the crop). The GPU turns them into
    the network input and the targets (gpu_inputs, gpu_targets), so the workers only decode,
    warp and jitter.

    store is a FrameStore holding frames c - 1, c, c + 1 of every centre c; pixels the (n, 2)
    full-resolution ball pixel of every video frame (mtah.vision.split, valid on all 3). Each
    call draws a fresh crop, flip and jitter (augment=False: a fixed centred crop, no jitter).

        ds = TrainWindows(store, [11186, 11190], pixels)
        crops, balls = ds[0]
        crops.shape, crops.dtype, balls.shape       # -> (3, 288, 512, 3), uint8, (3, 2)
    """

    def __init__(self, store, centres, pixels, augment=True, seed=0):
        self.store, self.centres = store, np.asarray(centres, np.int64)
        self.pixels, self.augment, self.seed = np.asarray(pixels, np.float64), augment, seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.centres)

    def __getitem__(self, i):
        c = int(self.centres[i])
        idx = (c - 1, c, c + 1)
        balls = [to_half(self.pixels[f]) for f in idx]
        if self.augment:
            M = crop_transform(self.rng, balls[1])
        else:
            M = crop_transform(np.random.default_rng(c), balls[1], scale=(1, 1), p_ball=1.0)
        flip = self.augment and self.rng.uniform() < P_FLIP
        if flip:
            M = np.array([[-1, 0, IN_W - 1], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
        gain = self.rng.uniform(1 - JITTER, 1 + JITTER) if self.augment else 1.0
        bias = self.rng.uniform(-JITTER, JITTER) * 255 / 4 if self.augment else 0.0
        imgs, pts = [], []
        for f, b in zip(idx, balls):
            img = cv2.warpAffine(self.store[f], M, (IN_W, IN_H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            if self.augment:
                img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            imgs.append(img)
            pts.append(M @ [b[0], b[1], 1.0])
        return torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.array(pts, np.float32))


class EvalWindows(torch.utils.data.Dataset):
    """The 3 half-resolution frames of each window centre: item i -> (3, 540, 960, 3) uint8.

        EvalWindows(store, [11186])[0].shape        # -> (3, 540, 960, 3)
    """

    def __init__(self, store, centres):
        self.store, self.centres = store, np.asarray(centres, np.int64)

    def __len__(self):
        return len(self.centres)

    def __getitem__(self, i):
        c = int(self.centres[i])
        return torch.from_numpy(np.stack([self.store[c - 1], self.store[c], self.store[c + 1]]))


_MEAN_T, _STD_T = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)


def gpu_inputs(crops):
    """(B, 3, H, W, 3) uint8 frames on the GPU -> (B, 9, H, W) float32 WASB input.

    The same arithmetic as normalise(), batched on the device.

        x = torch.full((1, 3, 2, 2, 3), 255, dtype=torch.uint8)
        gpu_inputs(x)[0, 0, 0, 0]                   # -> 2.249 = (1 - 0.485) / 0.229
    """
    b, t, h, w, _ = crops.shape
    x = crops.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w).float().div_(255.0)
    x = (x - _MEAN_T.to(x.device)) / _STD_T.to(x.device)
    return x.reshape(b, t * 3, h, w)


def gpu_targets(balls, h=IN_H, w=IN_W, r=RADIUS):
    """(B, 3, 2) ball positions (network px) -> (B, 3, h, w) binary disks, as binary_target().

        t = gpu_targets(torch.tensor([[[10.0, 5.0], [600.0, 5.0], [float("nan"), 0.0]]]))
        t[0].sum(dim=(1, 2))                        # -> [21., 0., 0.]
    """
    y = torch.arange(h, device=balls.device, dtype=torch.float32).view(1, 1, h, 1)
    x = torch.arange(w, device=balls.device, dtype=torch.float32).view(1, 1, 1, w)
    cx, cy = balls[..., 0, None, None], balls[..., 1, None, None]
    return (((x - cx) ** 2 + (y - cy) ** 2) <= r * r).float()     # NaN compares False


def gpu_tiles(frames):
    """(B, 3, 540, 960, 3) uint8 half frames -> (B * 4, 9, 288, 512) inputs, tiles in TILE_ORIGINS
    order for each window, as tiles().

        gpu_tiles(torch.zeros((2, 3, 540, 960, 3), dtype=torch.uint8)).shape   # -> (8, 9, 288, 512)
    """
    crops = torch.stack([frames[:, :, oy:oy + IN_H, ox:ox + IN_W] for ox, oy in TILE_ORIGINS], 1)
    return gpu_inputs(crops.flatten(0, 1))


def gpu_peaks(heatmaps):
    """(B, 4, 288, 512) tile heatmaps -> (score, u, v), each (B,): stitch (maximum over overlaps)
    and peak in FULL-resolution pixels, as stitch() + peak().

        h = torch.zeros((1, 4, 288, 512)); h[0, 0, 10, 20] = 0.7
        gpu_peaks(h)                                # -> ([0.7], [40.5], [20.5])
    """
    b = heatmaps.shape[0]
    full = torch.zeros((b, HALF_H, HALF_W), device=heatmaps.device, dtype=heatmaps.dtype)
    for k, (ox, oy) in enumerate(TILE_ORIGINS):
        region = full[:, oy:oy + IN_H, ox:ox + IN_W]
        torch.maximum(region, heatmaps[:, k], out=region)
    score, idx = full.flatten(1).max(dim=1)
    v, u = idx // HALF_W, idx % HALF_W
    return score, (u.float() + 0.5) * 2 - 0.5, (v.float() + 0.5) * 2 - 0.5


def worker_init(worker_id):
    """DataLoader worker_init_fn: each worker gets its own random stream (seed + worker id)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.dataset.seed * 1000 + worker_id + 1)


def tiles(frames):
    """3 half-resolution RGB frames (540, 960, 3), oldest first -> (4, 9, 288, 512) fixed tiles.

        tiles([np.zeros((540, 960, 3), np.uint8)] * 3).shape         # -> (4, 9, 288, 512)
    """
    return np.stack([normalise([f[oy:oy + IN_H, ox:ox + IN_W] for f in frames])
                     for ox, oy in TILE_ORIGINS])


def stitch(heatmaps):
    """(4, 288, 512) tile heatmaps -> (540, 960) half-resolution heatmap, maximum over overlaps.

        h = np.zeros((4, 288, 512), np.float32); h[3, 0, 0] = 0.9   # tile 4 starts at (448, 252)
        stitch(h)[252, 448]                                         # -> 0.9
    """
    out = np.zeros((HALF_H, HALF_W), np.float32)
    for (ox, oy), h in zip(TILE_ORIGINS, heatmaps):
        np.maximum(out[oy:oy + IN_H, ox:ox + IN_W], h, out=out[oy:oy + IN_H, ox:ox + IN_W])
    return out


def peak(heatmap):
    """(score, u, v): the stitched heatmap's maximum and its FULL-resolution pixel position.

        h = np.zeros((540, 960), np.float32); h[10, 20] = 0.7
        peak(h)                                     # -> (0.7, 40.5, 20.5)
    """
    v, u = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    fu, fv = to_full(np.array([u, v], np.float64))
    return float(heatmap[v, u]), float(fu), float(fv)
