"""A DFL match dressed up as our Match, with its tracking degraded to look like BAR-ATH's.

    from mtah.external import adapter, dfl
    m = dfl.load("DFL-MAT-J03WMX")
    gaps = adapter.presence_gaps(load())                   # from our match: aggregate lengths
    match = adapter.as_match(m, np.random.default_rng(0), gaps)
    table = team_shape.compute(match)                      # the same feature code as for BAR-ATH

Our feature code (mtah/features/team_shape.py) reads a Match from the cache
(mtah/data/cache.py). as_match() builds an object with the same attributes from a DFLMatch, so
the shape features of a DFL match are computed by exactly the code used on BAR-ATH: one
feature definition, two sources.

DFL's tracking is much cleaner than ours: all 22 players on every frame, no noise to speak of.
A model trained on it would see situations our tracking never shows. So the players are
degraded towards BAR-ATH (degrade_players):

  * each player is hidden for short stretches, as our players drop out when they leave the
    camera's view or tracking loses them. The stretch lengths are drawn from our own match's
    player gaps of at most MAX_GAP seconds (longer "gaps" of a slot are identity changes, the
    player continuing on another slot), and spaced so that on average VISIBLE players of 22
    are on the pitch, as in ours (20.5, measured on the graded frames);
  * every position gets a little noise (NOISE metres, smoothed over a few frames so it
    wobbles like a tracking error rather than jumping every frame).

There is no camera on DFL: the homography is invalid on every frame, so the camera features are
NaN (camera_advance_h, camera_y) or 0 (near_camera) and are left out of models trained on DFL.
The ball is kept as it is here (ball_xy, ball_present all True); a model of the players alone
ignores it. The whole-pipeline benchmark (mtah/external/benchmark.py) hides it in gaps like
ours with hide_ball() and with_ball_gaps():

  * the gap lengths are drawn from our match's own ball gaps (ball_gaps(): only the lengths and
    the share of graded frames without a ball leave our match), and gaps are added at random
    places until the same share of the DFL match has no ball;
  * a gap's start can be made more likely on dead frames (dead_bias), as our ball is often lost
    in stoppages (thrown back, the camera turning away), or left uniform (dead_bias 1).

In this file: presence_gaps() (our player gap lengths), degrade_players() (hide and jitter),
ball_gaps() (our ball gap lengths and share), hide_ball() (where to hide a DFL ball),
as_match() (the Match-like object), with DFLAsMatch the object itself, and with_ball_gaps()
(the ball of a DFLAsMatch hidden).
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

from mtah.external import dfl
from mtah.utils import runs_of

# Params
FPS = 25
MAX_GAP = 30 * FPS          # longer slot gaps in our match are identity changes, not drop-outs
VISIBLE = 20.5 / 22         # share of players visible on an average BAR-ATH graded frame
NOISE = 0.15                # metres, standard deviation of the position jitter
NOISE_SMOOTH = 3            # frames either side the jitter is averaged over


def presence_gaps(match):
    """Lengths in frames of our players' short drop-outs: graded stretches a slot is empty.

    For every player slot of our Match, the runs of graded frames without that player between
    two runs with them (so not before a first appearance or after the last), kept when at most
    MAX_GAP frames (30 s). Only these lengths leave the function: they are what
    degrade_players() copies onto DFL players.

        slot present on graded frames  T T F F F T T T F T
        presence_gaps(match)           # -> array([3, 1])
    """
    graded = np.asarray(match.graded)
    present = np.asarray(match.player_present)[graded]
    out = []
    for s in range(present.shape[1]):
        out += [b - a for a, b in runs_of(~present[:, s]) if a > 0 and b < len(present)]
    out = np.array(out, dtype=np.int64)
    return out[out <= MAX_GAP]


def ball_gaps(match):
    """(lengths in frames of our ball gaps, share of graded frames without a ball).

    The gaps are the runs of graded frames of our Match without a tracked ball, every length
    (a gap never crosses half-time, as only graded frames count). Only these lengths and the
    share leave the function: they are what hide_ball() copies onto a DFL match.

        graded all True, ball_present = T F F T T F T T T T
        ball_gaps(match)              # -> (array([2, 1]), 0.3)
    """
    missing = np.asarray(match.graded) & ~np.asarray(match.ball_present)
    lengths = np.array([b - a for a, b in runs_of(missing)], dtype=np.int64)
    return lengths, float(missing.sum() / np.asarray(match.graded).sum())


def hide_ball(n, lengths, share, rng, dead=None, dead_bias=1.0, half=None):
    """Bool mask of n frames where the ball is hidden: gaps of our lengths until `share` is hidden.

    Gaps are added one at a time, each with a length drawn from `lengths` (frames, from
    ball_gaps()) and a start frame drawn at random, until at least `share` of the n frames is
    hidden; gaps that overlap merge. With dead (a bool mask of dead-ball frames) and dead_bias
    above 1, a start on a dead frame is dead_bias times as likely as on an alive one. With half
    (1 / 2 per frame) a gap is cut at the end of its half, as ours never cross half-time.
    rng is a numpy Generator, so a seed gives the same mask every time.

        n = 1000, lengths = [10], share 0.2, uniform starts
        mask = hide_ball(1000, np.array([10]), 0.2, np.random.default_rng(0))
        mask.mean()                   # -> 0.202: stops at the first gap that reaches at
                                      #    least 200 frames, so a little over the share
        # dead = frames 0-99, dead_bias 4: the first 100 frames hold 4 / (4 + 9) of the starts
    """
    weight = np.ones(n)
    if dead is not None:
        weight[np.asarray(dead, dtype=bool)] = dead_bias
    cumulative = np.cumsum(weight)
    half = np.ones(n, dtype=np.int64) if half is None else np.asarray(half)
    stops = np.r_[np.flatnonzero(np.diff(half) != 0) + 1, n]      # first frame after each half
    half_end = stops[np.searchsorted(stops, np.arange(n), side="right")]
    mask = np.zeros(n, dtype=bool)
    target = int(np.ceil(share * n))
    hidden = 0
    while hidden < target:
        draws = rng.random(256) * cumulative[-1]
        starts = np.searchsorted(cumulative, draws, side="right")
        for start, length in zip(starts, rng.choice(lengths, 256)):
            stop = min(start + int(length), half_end[start])
            hidden += int((~mask[start:stop]).sum())
            mask[start:stop] = True
            if hidden >= target:
                break
    return mask


def degrade_players(xy, rng, gaps, visible=VISIBLE, noise=NOISE):
    """Copy of the (n, slots, 2) positions with players hidden in stretches and jittered.

    For each slot, over the frames where the player is on the pitch: hidden stretches with
    lengths drawn from `gaps` (frames), separated by visible stretches drawn from an
    exponential distribution whose mean makes the visible share `visible` on average (mean
    visible = mean gap * visible / (1 - visible)). Hidden frames become NaN. Then every position
    gets Gaussian noise of standard deviation `noise` metres, averaged over +/- NOISE_SMOOTH
    frames (and rescaled to keep that standard deviation), independently per slot and axis.
    rng is a numpy Generator, so a seed gives the same degradation every time.

        xy: 1 player, 1000 frames on the pitch; gaps = [10] frames; visible = 0.9
        out = degrade_players(xy, np.random.default_rng(0), np.array([10]), 0.9)
        np.isnan(out[:, 0, 0]).mean()      # -> about 0.1: ~10 hidden stretches of 10 frames
    """
    out = np.array(xy, dtype=np.float64)
    slots = out.shape[1]
    mean_visible = gaps.mean() * visible / (1 - visible)
    for s in range(slots):
        for a, b in runs_of(np.isfinite(out[:, s, 0])):
            t = a + int(rng.exponential(mean_visible))
            while t < b:
                length = int(rng.choice(gaps))
                out[t:t + length, s] = np.nan
                t += length + int(rng.exponential(mean_visible))
    if noise > 0:
        jitter = rng.normal(0.0, 1.0, out.shape)
        jitter = uniform_filter1d(jitter, 2 * NOISE_SMOOTH + 1, axis=0, mode="nearest")
        jitter *= noise / np.nanstd(jitter)
        out += jitter
    return out


class DFLAsMatch:
    """The attributes of our Match that the feature code reads, filled from a DFLMatch.

    players_xy (n, slots, 2), players_speed (n, slots), player_present (n, slots), team_of_slot,
    goalkeeper_slots, half (1 / 2 on every row), graded (all True), home_attacks_right
    {half: bool}, pitch (105, 68), fps 25, n_frames, ball_xy, ball_present (all True),
    homography (NaN) and homography_valid (all False: there is no camera). Built by as_match().
    """


def as_match(m, rng=None, gaps=None):
    """A DFLAsMatch of the DFLMatch m: degraded when rng and gaps are given, clean otherwise.

    home_attacks_right comes from where the home goalkeeper stands: in a half where the
    keeper's mean x is in the left half of the pitch (x < 52.5), home defends the left goal and
    attacks right.
    Players hidden by degrade_players() are absent (player_present False, position NaN) and
    their speed is NaN.

        match = as_match(dfl.load("DFL-MAT-J03WMX"))
        match.home_attacks_right      # -> {1: False, 2: True}   home keeper at x 90 in half 1
        match.player_present.sum(axis=1).mean()                  # -> about 22 (clean)
    """
    out = DFLAsMatch()
    xy = np.asarray(m.players_xy, dtype=np.float64)
    if rng is not None and gaps is not None:
        xy = degrade_players(xy, rng, gaps)
    present = np.isfinite(xy).all(axis=2)
    out.players_xy = xy.astype(np.float32)
    out.player_present = present
    out.players_speed = np.where(present, m.players_speed, np.nan).astype(np.float32)
    out.team_of_slot = np.asarray(m.slot_team)
    out.goalkeeper_slots = dfl.goalkeeper_slots(m)
    out.half = np.asarray(m.half)
    out.n_frames = m.n_frames
    out.graded = np.ones(m.n_frames, dtype=bool)
    out.pitch = dfl.PITCH
    out.fps = FPS
    keeper = np.asarray(m.players_xy)[:, out.goalkeeper_slots[0], 0]
    out.home_attacks_right = {h: bool(np.nanmean(keeper[out.half == h]) < dfl.PITCH[0] / 2)
                              for h in (1, 2)}
    out.ball_xy = np.asarray(m.ball_xyz)[:, :2]
    out.ball_present = np.ones(m.n_frames, dtype=bool)
    out.homography = np.full((m.n_frames, 3, 3), np.nan)
    out.homography_valid = np.zeros(m.n_frames, dtype=bool)
    return out


def with_ball_gaps(match, hidden):
    """The DFLAsMatch `match` with its ball hidden on the frames of the bool mask `hidden`.

    ball_present becomes False and ball_xy NaN there, as in our cache where the ball is not
    tracked. The match is changed in place and returned.

        match = with_ball_gaps(as_match(m), hide_ball(m.n_frames, lengths, 0.41, rng))
        match.ball_present.mean()     # -> about 0.59
    """
    hidden = np.asarray(hidden, dtype=bool)
    match.ball_present = ~hidden
    match.ball_xy = np.where(hidden[:, None], np.nan, match.ball_xy)
    return match
