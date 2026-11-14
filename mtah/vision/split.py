"""Valid label frames of the tactical video and their train / validation / test split.

    from mtah.vision import split
    s = split.build()                    # from the match files (~1 min)
    s["split"][f], s["valid"][f], s["pixel"][f]   # split code, label valid?, ball pixel (u, v)
    split.save(s); split.load()          # cache/vision/wasb_split_v<VERSION>.npz

**Valid frame.** A tactical frame is a usable label when it is graded, the tracking has the ball,
the homography is valid, and the ball's pitch position projected by the homography lands inside
the 1920x1080 image. Its label is that pixel (u, v). A 3-frame WASB window is usable when all its
frames are valid and in the same split (window_centres).

**Split** (agreed 2026-11-11), so that train, validation and test all cover the whole match:

  1. six patches of match time from each kick-off (games.py anchors): 0-15, 15-30, 30-end of
     the first half and 45-60, 60-75, 75-end of the second. The first patch of a half starts at
     the half's first graded frame (so patch 4 also holds the graded frames before the
     second-half kick-off); the last runs to the half's last graded frame (stoppage included);
  2. each patch cut into 1-minute chunks (a last piece shorter than half a chunk joins the one
     before it);
  3. inside each patch the chunks go to train / validation / test, aiming at SHARES of the
     patch's VALID frames: DRAWS random orders of the chunks (seeded) are each filled greedily
     (test first, then validation, the rest train), and the order closest to SHARES is kept.
     The draw looks only at frame counts, never at images or results;
  4. where two neighbouring chunks of a half belong to different splits, BUFFER frames around
     the border are dropped (half on each side), so no training frame sits next to a test frame.

Codes: TRAIN 0, VAL 1, TEST 2, NONE -1 (ungraded, buffer). The split covers every graded frame;
whether a frame is a usable label is `valid`, separately.
"""

import os

import numpy as np

from games import GAME, GAMES, anchor_video_frame, half_video_blocks

# Params
FPS = 25
IMG_W, IMG_H = 1920, 1080
PATCH_MINUTES = (15, 30)            # patch borders after each kick-off; the third runs to the end
CHUNK = 60 * FPS                    # 1-minute chunks
SHARES = (0.70, 0.15, 0.15)         # train, validation, test, of each patch's valid frames
BUFFER = 10 * FPS                   # dropped around a border between different splits
SEED, DRAWS = 0, 2000
VERSION = 1
TRAIN, VAL, TEST, NONE = 0, 1, 2, -1
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "cache", "vision")


def ball_pixels(ball, homographies):
    """(n, 2) image pixel (u, v) of the tracking ball on every frame, NaN where it has none.

    ball is (n, 2) normalised pitch x, y (NaN when missing), homographies (n, 3, 3) mapping
    normalised pitch to pixels (mtah.data.io.read_homography).

        H = [[1920, 0, 0], [0, 1080, 0], [0, 0, 1]]           # pitch (0-1) -> full image
        ball_pixels([[0.5, 0.5], [nan, nan]], [H, H])         # -> [[960, 540], [nan, nan]]
    """
    ball = np.asarray(ball, np.float64)
    H = np.asarray(homographies, np.float64)
    p = np.einsum("nij,nj->ni", H, np.c_[ball, np.ones(len(ball))])
    return p[:, :2] / p[:, 2:3]


def valid_frames(pixel, homography_valid, graded):
    """bool per frame: graded, tracking ball, valid homography, ball pixel inside the image.

        pixel = [[960, 540], [nan, nan], [2000, 500], [10, 10]]
        valid_frames(pixel, [T, T, T, F], [T, T, T, T])       # -> [T, F, F, F]
    """
    u, v = np.asarray(pixel, np.float64).T
    inside = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)       # NaN compares False
    return inside & np.asarray(homography_valid, bool) & np.asarray(graded, bool)


def patches(game=GAME):
    """[(patch, first, last)] video frames (inclusive) of the six patches, in play order.

    Borders at PATCH_MINUTES after each half's kick-off; each half's first patch starts at its
    first graded frame and its last ends at its last graded frame. For BAR-ATH:

        patches()[0], patches()[3]     # -> (1, 11185, 33685), (4, 110808, 136240)
    """
    kickoffs = (anchor_video_frame(GAMES[game]["kickoff"]),
                anchor_video_frame(GAMES[game]["kickoff_2nd"]))
    out = []
    for (half, first, last), kick in zip(half_video_blocks(game), kickoffs):
        edges = [first] + [kick + m * 60 * FPS for m in PATCH_MINUTES] + [last + 1]
        for p in range(len(edges) - 1):
            out.append((3 * (half - 1) + p + 1, edges[p], edges[p + 1] - 1))
    return out


def chunks(first, last, length=CHUNK):
    """[(start, stop)] half-open chunks of `length` frames over first..last (inclusive).

    A last piece shorter than half a chunk joins the chunk before it.

        chunks(0, 239, 100)            # -> [(0, 100), (100, 240)]      the 40-frame tail joins
        chunks(0, 279, 100)            # -> [(0, 100), (100, 200), (200, 280)]
    """
    starts = list(range(first, last + 1, length))
    if len(starts) > 1 and last + 1 - starts[-1] < length / 2:
        starts.pop()
    return [(s, e) for s, e in zip(starts, starts[1:] + [last + 1])]


def assign(counts, shares=SHARES, seed=SEED, draws=DRAWS):
    """Split code per chunk so the valid-frame shares come closest to `shares`.

    counts is the number of valid frames per chunk. Each of `draws` seeded random orders of the
    chunks is filled greedily: chunks go to TEST until it holds its share, then to VAL, the rest
    to TRAIN. The order whose shares are closest to `shares` (sum of absolute differences) wins.

        assign([100] * 20)             # 20 equal chunks -> 14 TRAIN, 3 VAL, 3 TEST (exact 70/15/15)
        np.bincount(assign([100] * 20))                    # -> [14, 3, 3]
    """
    counts = np.asarray(counts, np.float64)
    total = counts.sum()
    target = np.asarray(shares) * total
    rng = np.random.default_rng(seed)
    best, best_err = None, np.inf
    for _ in range(draws):
        code = np.full(len(counts), TRAIN)
        held = {TEST: 0.0, VAL: 0.0}
        for c in rng.permutation(len(counts)):
            for s in (TEST, VAL):
                if held[s] < target[s]:
                    code[c], held[s] = s, held[s] + counts[c]
                    break
        err = np.abs(np.bincount(code, weights=counts, minlength=3) - target).sum()
        if err < best_err:
            best, best_err = code, err
    return best


def with_buffers(split, bounds, buffer=BUFFER):
    """Copy of the per-frame split with NONE around borders between different splits.

    bounds are the (start, stop) chunks of ONE half, in order; where chunk k and k + 1 differ,
    buffer // 2 frames on each side of the border become NONE.

        split = [0] * 10 + [2] * 10                        # TRAIN then TEST, border at 10
        with_buffers(split, [(0, 10), (10, 20)], 4)        # -> frames 8..11 become -1
    """
    out = np.array(split, copy=True)
    half = buffer // 2
    for (s0, e0), (s1, e1) in zip(bounds, bounds[1:]):
        if out[s0] != out[s1]:
            out[max(s0, e0 - half):e0] = NONE
            out[s1:min(e1, s1 + half)] = NONE
    return out


def window_centres(valid, split):
    """bool per frame: a usable 3-frame window is centred here (all 3 valid, same split).

        valid = [T, T, T, T, F]; split = [0, 0, 0, 1, 1]
        window_centres(valid, split)                       # -> [F, T, F, F, F]
    """
    v, s = np.asarray(valid, bool), np.asarray(split)
    out = np.zeros(len(v), bool)
    out[1:-1] = (v[:-2] & v[1:-1] & v[2:] & (s[:-2] == s[1:-1]) & (s[1:-1] == s[2:])
                 & (s[1:-1] != NONE))
    return out


def build(game=GAME):
    """{split, patch, chunk, valid, pixel, centre} per tactical video frame, from the match files.

    split int8 (codes above), patch int8 (1-6, 0 outside), chunk int16 (running index, -1
    outside), valid bool, pixel (n, 2) float32 ball pixel (NaN without a tracking ball), centre
    bool (window_centres).

        s = build()
        (s["split"] == TEST).sum(), s["valid"].sum()       # -> e.g. (22000, 94131)
    """
    from mtah.data import io
    _, ball = io.read_atd_raw(game)
    H, _, hvalid = io.read_homography(game)
    n = len(ball)
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True
    pixel = ball_pixels(ball, H)
    valid = valid_frames(pixel, hvalid, graded)

    split = np.full(n, NONE, np.int8)
    patch = np.zeros(n, np.int8)
    chunk = np.full(n, -1, np.int16)
    half_bounds = {}
    k = 0
    for p, first, last in patches(game):
        bounds = chunks(first, last)
        codes = assign([valid[s:e].sum() for s, e in bounds])
        for (s, e), c in zip(bounds, codes):
            split[s:e], patch[s:e], chunk[s:e] = c, p, k
            k += 1
        half_bounds.setdefault((p - 1) // 3, []).extend(bounds)
    for bounds in half_bounds.values():
        split = with_buffers(split, bounds)
    return {"split": split, "patch": patch, "chunk": chunk, "valid": valid,
            "pixel": pixel.astype(np.float32), "centre": window_centres(valid, split)}


def save(s):
    """Write a build() result to CACHE/wasb_split_v<VERSION>.npz; returns the path."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "wasb_split_v{}.npz".format(VERSION))
    np.savez(path, version=VERSION, **s)
    return path


def load():
    """The saved split as a dict of arrays (see build); raises if missing or of another VERSION."""
    path = os.path.join(CACHE, "wasb_split_v{}.npz".format(VERSION))
    with np.load(path) as data:
        if int(data["version"]) != VERSION:
            raise ValueError("{}: version {} != {}".format(path, int(data["version"]), VERSION))
        return {k: data[k] for k in data.files if k != "version"}
