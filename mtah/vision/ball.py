"""The tracking ball corrected by the fine-tuned WASB, for the step-2 control rule.

    from mtah.vision import ball
    corrected = ball.corrected_match(match)        # match with a corrected ball_xy, else the same
    control_events(corrected)                      # mtah/possession/control.py on it

Why: on the test frames the fine-tuned WASB (cache/vision/runs/r5/best.pt) finds the ball in the
image more precisely than the tracking ball projected with the homography, while the players
line up with the image (the user's visual check of 500 test frames): the tracking ball is a bit
off, not the calibration. The control rule (ball within 1.25 m of a player, moving with them)
is sensitive to exactly that error, so its ball can be corrected where WASB is sure.

The rule, fixed before any result was seen (work log, 2026-11-12):

  1. ACCEPTED frame: graded, tracking ball present, homography valid, a WASB peak with
     score > MIN_SCORE (0.5, WASB's own threshold), within MAX_PX (50) image pixels of the
     projected tracking ball, and its ground point within MAX_M (2 m) of the tracking ball.
     The WASB pixel is taken to the ground with the inverse homography (ground_points).
  2. The correction is the ground OFFSET (WASB minus tracking, metres) on accepted frames. It is
     never applied where the tracking ball is missing: no ball is added, only moved.
  3. Mixing sources frame by frame would add jitter (the control rule reads the ball velocity
     over +/- 2 frames: a 0.5 m jump is ~3 m/s, its relative-speed limit). So, inside each run of
     frames with a tracking ball (smooth_offset): gaps of at most FILL (12 frames, 0.5 s) between
     accepted frames are filled by linear interpolation; elsewhere the offset fades linearly to
     0 over TAPER (12) frames from the nearest accepted frame; then a centred mean over SMOOTH
     (5) frames.
  4. corrected ball = tracking ball + offset.

GAP FILLING (filled_match, hmm_v14_gap), a second, separate use: where the tracking ball is
MISSING, WASB's ground point is used as the ball when all of these hold, fixed before any HMM
result was seen (work log, 2026-11-12, where the counts are):

  1. score > MIN_SCORE (0.5);
  2. the ground point lies on the pitch, with GAP_MARGIN (1 m) to spare;
  3. it is part of a CHAIN of at least MIN_CHAIN (5) consecutive such frames whose every step is
     at most MAX_STEP (1.6 m per frame = 40 m/s, the control rule's spike limit);
  4. the chain is LINKED to the tracking ball: it starts within LINK (12) frames after the
     tracking ball's last frame before the gap (or ends within LINK before its first frame
     after) and the jump is at most JUMP (2 m) + MAX_STEP per frame elapsed; linking continues
     chain to chain with the same rule, forwards from the gap start and backwards from its end.

On BAR-ATH that fills 6,458 of the 64,643 gap frames. The peaks are in GAPS_CACHE
(scripts/vision/wasb_ball.py --gaps). Only frames of the original gaps have peaks there, so on a
masked match (mtah/validation/masking.py) the masked frames are never filled.

The WASB peaks come from scripts/vision/wasb_ball.py (one GPU job, ~1 h), saved to CACHE
(gitignored). A ball at a player's feet is on the ground, where taking the pixel to the ground is
right; a ball in the air lands too far away, but it rarely passes the control rule anyway.
"""

import os

import numpy as np

from mtah import ROOT
from mtah.utils import runs_of

# Params
VERSION = 1
CACHE = os.path.join(ROOT, "cache", "vision", "wasb_ball_v{}.npz".format(VERSION))
GAPS_CACHE = os.path.join(ROOT, "cache", "vision", "wasb_gaps_v{}.npz".format(VERSION))
CHECKPOINT = os.path.join(ROOT, "cache", "vision", "runs", "r5", "best.pt")
MIN_SCORE = 0.5             # WASB's detection threshold
MAX_PX = 50.0               # image px, WASB peak to the projected tracking ball
MAX_M = 2.0                 # metres on the ground, the same two points
FILL = 12                   # frames: gaps between accepted frames filled by interpolation
TAPER = 12                  # frames: the offset fades to 0 over this many frames
SMOOTH = 5                  # frames: centred mean of the offset
GAP_MARGIN = 1.0            # metres: gap filling, ground point this far outside the lines at most
MIN_CHAIN = 5               # frames: gap filling, shortest chain of consistent peaks
MAX_STEP = 1.6              # metres per frame (40 m/s): gap filling, largest step inside a chain
LINK = 12                   # frames: gap filling, largest time between a chain and its anchor
JUMP = 2.0                  # metres: gap filling, jump allowed on top of MAX_STEP per frame


def candidate_frames(match):
    """Frames WASB is run on: graded, tracking ball present, homography valid, not the first or
    last frame of the video (the 3-frame window needs both neighbours).

        candidate_frames(match)[:2]        # -> e.g. [11185, 11186]
    """
    ok = (np.asarray(match.graded) & np.asarray(match.ball_present)
          & np.asarray(match.homography_valid))
    ok[[0, -1]] = False
    return np.flatnonzero(ok)


def gap_frames(match):
    """Frames of the tracking ball's GAPS that WASB is run on: graded, tracking ball missing,
    homography valid, not the first or last frame of the video. The complement of
    candidate_frames within the graded frames with a valid homography; the peaks are saved to
    GAPS_CACHE by scripts/vision/wasb_ball.py --gaps, for the gap-filling count.

        len(gap_frames(match))                 # -> 64643 on BAR-ATH (150 gaps, median 10.7 s)
        np.intersect1d(gap_frames(match), candidate_frames(match)).size      # -> 0
    """
    ok = (np.asarray(match.graded) & ~np.asarray(match.ball_present)
          & np.asarray(match.homography_valid))
    ok[[0, -1]] = False
    return np.flatnonzero(ok)


def load(path=CACHE):
    """{frame, score, u, v} of the saved WASB peaks (see scripts/vision/wasb_ball.py).

        load()["frame"].shape == load()["score"].shape      # -> True
    """
    with np.load(path) as data:
        if int(data["version"]) != VERSION:
            raise ValueError("{}: version {} != {}".format(path, int(data["version"]), VERSION))
        return {k: data[k] for k in ("frame", "score", "u", "v")}


def project(H, xy_norm):
    """(k, 2) normalised pitch points -> (k, 2) image pixels, one 3x3 homography per row (k, 3, 3).

        project(np.diag([1920.0, 1080.0, 1.0])[None], np.array([[0.5, 0.5]]))   # -> [[960., 540.]]
    """
    p = np.einsum("kij,kj->ki", H, np.c_[xy_norm, np.ones(len(xy_norm))])
    with np.errstate(invalid="ignore", divide="ignore"):
        return p[:, :2] / p[:, 2:]


def ground_points(H, uv, pitch):
    """(k, 2) image pixels -> (k, 2) ground points in metres, one homography per row; NaN where
    the pixel lies above the horizon (it does not meet the ground in front of the camera).

        ground_points(np.diag([1920.0, 1080.0, 1.0])[None], np.array([[960.0, 540.0]]),
                      (105.3, 68.0))                                  # -> [[52.65, 34.0]]
    """
    p = np.einsum("kij,kj->ki", np.linalg.inv(H), np.c_[uv, np.ones(len(uv))])
    with np.errstate(invalid="ignore", divide="ignore"):
        g = p[:, :2] / p[:, 2:] * np.asarray(pitch, np.float64)
    g[p[:, 2] <= 0] = np.nan
    return g


def raw_offsets(match, peaks):
    """(n, 2) ground offset (WASB minus tracking, metres) on ACCEPTED frames, NaN elsewhere.

    Accepted (rule 1 of the module docstring): graded, tracking ball, valid homography, score >
    MIN_SCORE, within MAX_PX in the image and MAX_M on the ground. Frames not in `peaks` are not
    accepted.

        # one frame, identity-like homography; tracking ball at pixel (960, 540), WASB at (970, 540)
        raw_offsets(match, {"frame": [f], "score": [0.7], "u": [970.0], "v": [540.0]})[f]
        # -> about [0.55, 0.0]: 10 px = 0.55 m along x on a 1920-px-wide 105.3 m pitch
    """
    n = match.n_frames
    out = np.full((n, 2), np.nan)
    f = np.asarray(peaks["frame"], np.int64)
    score = np.asarray(peaks["score"], np.float64)
    uv = np.c_[peaks["u"], peaks["v"]].astype(np.float64)
    pitch = np.asarray(match.pitch, np.float64)
    ok = (np.asarray(match.graded)[f] & np.asarray(match.ball_present)[f]
          & np.asarray(match.homography_valid)[f] & (score > MIN_SCORE))
    f, uv = f[ok], uv[ok]
    H = np.asarray(match.homography[f], np.float64)
    tracking = np.asarray(match.ball_xy[f], np.float64)
    px = np.linalg.norm(project(H, tracking / pitch) - uv, axis=1)
    offset = ground_points(H, uv, pitch) - tracking
    keep = (px <= MAX_PX) & (np.linalg.norm(offset, axis=1) <= MAX_M)
    out[f[keep]] = offset[keep]
    return out


def smooth_offset(raw, present, fill=FILL, taper=TAPER, smooth=SMOOTH):
    """(n, 2) offsets for every frame from the raw ones (NaN = not accepted); 0 where no ball.

    Inside each run of `present` frames (rule 3 of the module docstring): a gap of at most `fill`
    frames between two accepted frames is interpolated linearly; any other frame takes the
    nearest accepted frame's offset scaled by max(0, 1 - d / (taper + 1)), d frames away (both
    sides added, scaled down if their weights pass 1); then a centred mean over `smooth` frames,
    within the run. Frames outside the runs are 0.

        raw = np.full((8, 2), np.nan); raw[2] = [1.0, 0.0]
        smooth_offset(raw, np.ones(8, bool), fill=2, taper=2, smooth=1)[:, 0]
        # -> [0.33, 0.67, 1.0, 0.67, 0.33, 0.0, 0.0, 0.0]
        raw[4] = [3.0, 0.0]                                  # gap of 1 frame: interpolated
        smooth_offset(raw, np.ones(8, bool), fill=2, taper=2, smooth=1)[:, 0]
        # -> [0.33, 0.67, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
    """
    raw = np.asarray(raw, np.float64)
    out = np.zeros_like(raw)
    for start, stop in runs_of(np.asarray(present, bool)):
        seg = raw[start:stop]
        acc = np.flatnonzero(np.isfinite(seg[:, 0]))
        if not len(acc):
            continue
        idx = np.arange(stop - start)
        nxt = np.searchsorted(acc, idx)                       # first accepted >= idx
        prv = nxt - 1
        has_p = prv >= 0
        has_n = nxt < len(acc)
        p_i = acc[np.clip(prv, 0, len(acc) - 1)]
        n_i = acc[np.clip(nxt, 0, len(acc) - 1)]
        val = np.zeros_like(seg)
        exact = has_n & (n_i == idx)
        val[exact] = seg[idx[exact]]
        gap = n_i - p_i - 1
        interp = ~exact & has_p & has_n & (gap <= fill)
        if interp.any():
            i, p, q = idx[interp], p_i[interp], n_i[interp]
            w = ((i - p) / (q - p))[:, None]
            val[interp] = (1 - w) * seg[p] + w * seg[q]
        rest = ~exact & ~interp
        if rest.any():
            wp = np.where(has_p, np.maximum(0.0, 1 - (idx - p_i) / (taper + 1)), 0.0)
            wn = np.where(has_n, np.maximum(0.0, 1 - (n_i - idx) / (taper + 1)), 0.0)
            scale = np.maximum(1.0, wp + wn)
            t = (wp[:, None] * seg[p_i] + wn[:, None] * seg[n_i]) / scale[:, None]
            val[rest] = t[rest]
        if smooth > 1:
            k = np.ones(smooth)
            num = np.stack([np.convolve(val[:, j], k, "same") for j in range(2)], 1)
            den = np.convolve(np.ones(len(val)), k, "same")[:, None]
            val = num / den
        out[start:stop] = val
    return out


def chains(ok, pts, min_len=MIN_CHAIN, max_step=MAX_STEP):
    """[(start, stop)] runs of `ok` frames whose consecutive points move at most max_step
    metres, kept when at least min_len frames long (a bigger step splits a run). Gap-filling
    rule 3.

        chains(np.array([1, 1, 1, 1, 1, 0, 1, 1], bool), np.zeros((8, 2)), 5)
        # -> [(0, 5)]: the 2-frame run is too short
        chains(np.array([1, 1, 1, 0], bool), np.array([[0, 0], [1, 0], [5, 0], [0, 0.]]), 1)
        # -> [(0, 2), (2, 3)]: the 4 m step splits the run
    """
    out = []
    for s, e in runs_of(np.asarray(ok, bool)):
        cut = s
        for i in range(s + 1, e):
            if np.linalg.norm(pts[i] - pts[i - 1]) > max_step:
                if i - cut >= min_len:
                    out.append((cut, i))
                cut = i
        if e - cut >= min_len:
            out.append((cut, e))
    return out


def linked(chs, start_pt, end_pt, gap_len):
    """Set of chain indices linked to the tracking ball (gap-filling rule 4). chs are
    (start, stop, first point, last point) in gap-relative frames, in time order; the tracking
    ball is start_pt at frame -1 and end_pt at frame gap_len (None when there is none).

        linked([(0, 5, np.zeros(2), np.ones(2))], np.zeros(2), None, 100)       # -> {0}
        linked([(30, 35, np.zeros(2), np.ones(2))], np.zeros(2), None, 100)     # -> set(): too late
        linked([(0, 5, np.full(2, 9.0), np.ones(2))], np.zeros(2), None, 100)   # -> set(): too far
    """
    keep = set()
    if start_pt is not None:
        f, p = -1, start_pt
        for k, (s, e, a, b) in enumerate(chs):
            if s - f > LINK:
                break
            if np.linalg.norm(a - p) <= JUMP + MAX_STEP * (s - f):
                keep.add(k)
                f, p = e - 1, b
    if end_pt is not None:
        f, p = gap_len, end_pt
        for k in range(len(chs) - 1, -1, -1):
            s, e, a, b = chs[k]
            if f - (e - 1) > LINK:
                break
            if np.linalg.norm(b - p) <= JUMP + MAX_STEP * (f - (e - 1)):
                keep.add(k)
                f, p = s, a
    return keep


def gap_fill(match, peaks):
    """(n, 2) ground points (metres) and an (n,) mask of the gap frames filled by rules 1-4.

    Gap frames: graded, no tracking ball, homography valid. A frame missing from `peaks`
    counts as score 0 (never filled).

        pts, filled = gap_fill(match, load(GAPS_CACHE))
        filled.sum(), np.isfinite(pts[filled]).all()          # -> 6458, True on BAR-ATH
    """
    n = match.n_frames
    pitch = np.asarray(match.pitch, np.float64)
    present = np.asarray(match.ball_present)
    gap = np.asarray(match.graded) & ~present & np.asarray(match.homography_valid)
    f = np.asarray(peaks["frame"], np.int64)
    score = np.zeros(n)
    score[f] = peaks["score"]
    pts = np.full((n, 2), np.nan)
    pts[f] = ground_points(np.asarray(match.homography[f], np.float64),
                           np.c_[peaks["u"], peaks["v"]].astype(np.float64), pitch)
    with np.errstate(invalid="ignore"):
        ok = (gap & (score > MIN_SCORE) & np.isfinite(pts[:, 0])
              & (pts >= -GAP_MARGIN).all(axis=1) & (pts <= pitch + GAP_MARGIN).all(axis=1))
    bxy = np.asarray(match.ball_xy, np.float64)
    filled = np.zeros(n, bool)
    for s, e in runs_of(gap):
        chs = [(s + a, s + b) for a, b in chains(ok[s:e], pts[s:e])]
        sp = bxy[s - 1] if s > 0 and present[s - 1] else None
        ep = bxy[e] if e < n and present[e] else None
        info = [(a - s, b - s, pts[a], pts[b - 1]) for a, b in chs]
        for k in linked(info, sp, ep, e - s):
            filled[chs[k][0]:chs[k][1]] = True
    pts[~filled] = np.nan
    return pts, filled


class _CorrectedBall:
    """A match whose ball_xy (and optionally ball_present) is replaced; every other attribute
    is the original match's.

        _CorrectedBall(match, xy).players_xy is match.players_xy               # -> True
    """

    def __init__(self, match, ball_xy, ball_present=None):
        self._match = match
        self.ball_xy = ball_xy
        if ball_present is not None:
            self.ball_present = ball_present

    def __getattr__(self, name):
        return getattr(self._match, name)


def corrected_match(match, peaks=None):
    """The match with its tracking ball moved by the smoothed WASB offset (rules 1-4).

    peaks defaults to load(). Only frames with a tracking ball move; ball_present is unchanged,
    so a masked match (mtah/validation/masking.py) is corrected only where its ball is shown.
    The object has a .correction attribute: {"accepted": frames accepted, "moved": frames moved
    by more than 1 cm, "median_m": median shift of the moved frames}.

        c = corrected_match(match)
        c.ball_xy is not match.ball_xy, c.players_xy is match.players_xy     # -> True, True
    """
    peaks = load() if peaks is None else peaks
    present = np.asarray(match.ball_present)
    raw = raw_offsets(match, peaks)
    offset = smooth_offset(raw, present)
    ball_xy = np.asarray(match.ball_xy, np.float64) + offset
    ball_xy[~present] = np.nan
    out = _CorrectedBall(match, ball_xy.astype(np.float32))
    shift = np.linalg.norm(offset, axis=1)
    moved = present & (shift > 0.01)
    out.correction = {"accepted": int(np.isfinite(raw[:, 0]).sum()), "moved": int(moved.sum()),
                      "median_m": float(np.median(shift[moved])) if moved.any() else 0.0}
    return out


def filled_match(match, peaks=None):
    """The match with the tracking ball's gaps filled by WASB where rules 1-4 of GAP FILLING
    hold (module docstring); those frames get ball_present True. Every other frame is the
    tracking ball as it is. peaks defaults to load(GAPS_CACHE). The object has a .filling
    attribute: {"filled": frames filled, "gaps": gap frames}.

        f = filled_match(match)
        f.filling                                    # -> {"filled": 6458, "gaps": 64643}
        int(f.ball_present.sum() - match.ball_present.sum())                  # -> 6458
    """
    peaks = load(GAPS_CACHE) if peaks is None else peaks
    pts, filled = gap_fill(match, peaks)
    xy = np.asarray(match.ball_xy, np.float32).copy()
    xy[filled] = pts[filled]
    present = np.asarray(match.ball_present).copy()
    present[filled] = True
    out = _CorrectedBall(match, xy, present)
    gap = np.asarray(match.graded) & ~np.asarray(match.ball_present) \
        & np.asarray(match.homography_valid)
    out.filling = {"filled": int(filled.sum()), "gaps": int(gap.sum())}
    return out
