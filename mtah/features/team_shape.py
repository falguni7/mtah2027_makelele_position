"""Ball-free features for "which team has the ball", one row per video frame.

    from mtah.data import load
    from mtah.features.team_shape import compute, COLUMNS
    match = load()
    table = compute(match)                    # DataFrame, n_frames rows, columns COLUMNS
    table["advance_diff"]                     # > 0 where home stands higher up than away

None of these use the tracked ball, so they are just as available inside a ball gap as outside
one -- which is the point (roadmap step 4). They describe how each team stands and moves,
oriented by its own attacking direction, plus where the camera points:

  * the team in possession pushes up and spreads out; the defending team drops and compacts;
  * the operator keeps the ball in view, so the centre of the image is a rough ball location.

Per team (suffix _h home, _a away) and as home-minus-away differences (_diff), each averaged
over 1 s (centred, +/- SMOOTH = 12 frames, i.e. +/- 0.5 s):

  advance       mean outfield position along the team's attacking direction (m from halfway)
  front, back   90th / 10th percentile of that, the attacking front and the defensive line
  spread_x/y    standard deviation of outfield positions, length and width
  forward_vel   mean velocity towards the opponent's goal (m/s)
  speed         mean speed (m/s)
  marking       mean distance from each player to the nearest opponent (m)
  near_camera   players within NEAR_CAMERA metres of the camera's pitch point

Camera: the image centre projected onto the pitch, as the home team's oriented position along the
length (camera_advance_h) and the distance from the long axis (camera_y).

Positions are in metres, x measured from halfway along the length and y across the width. Only
outfield players count for a team (goalkeepers are left out), but the opponents for marking
include the goalkeeper. "Oriented" means multiplied by the team's attacking direction, so a
positive value is in the opponent's half for both teams. Off the two graded halves the direction
sign is 0, so advance, front, back, forward_vel and camera_advance_h are 0 there (not NaN);
near_camera is 0 where the homography is invalid. The possession shape model
(mtah.possession.shape_model) uses this table as evidence inside ball gaps.

In this file: home_direction and camera_point (per-frame helpers), velocity, team_columns (one
team's raw features) and compute (both teams, the differences and the camera, smoothed).
"""

import numpy as np
import pandas as pd

from mtah.utils import centred

# Params
FPS = 25
SMOOTH = FPS // 2
NEAR_CAMERA = 15.0
IMAGE_CENTRE = (960.0, 540.0)
VELOCITY_HALF = 2
PER_TEAM = ["advance", "front", "back", "spread_x", "spread_y", "forward_vel", "speed",
            "marking", "near_camera"]
COLUMNS = ([name + "_h" for name in PER_TEAM] + [name + "_a" for name in PER_TEAM]
           + [name + "_diff" for name in PER_TEAM] + ["camera_advance_h", "camera_y"])


def home_direction(match):
    """+1 where home attacks towards x = length, -1 where it attacks towards 0, 0 off the halves.

    The home team's attacking direction on every frame, from the per-half directions in the
    cache; frames outside the two halves get 0. The away team's direction is the negative.
    Multiplying an x by it orients that x for the home team.

        match.home_attacks_right = {1: True, 2: False}
        match.half               = [0, 1, 1, 2, 2]
        home_direction(match)    # -> [0, 1, 1, -1, -1]
    """
    out = np.zeros(match.n_frames)
    half = np.asarray(match.half)
    for number, right in match.home_attacks_right.items():
        out[half == number] = 1.0 if right else -1.0
    return out


def camera_point(match):
    """Pitch position (metres) of the image centre on every frame; NaN where no homography.

    An (n_frames, 2) array of (x, y) in metres from the pitch corner: where on the pitch the
    middle of the broadcast image (IMAGE_CENTRE, pixel 960, 540) points. The operator follows the
    ball, so this is a rough ball location that exists inside ball gaps. Frames whose homography
    is flagged invalid or has non-finite entries are NaN.

        pitch = (105, 68), homography = diag(1920, 1080, 1)    # normalised pitch -> pixels
        camera_point(match)[i]       # -> (52.5, 34.0), the centre spot
    """
    length, width = match.pitch
    matrices = np.asarray(match.homography, dtype=np.float64)
    valid = np.asarray(match.homography_valid) & np.isfinite(matrices).all(axis=(1, 2))
    point = np.full((match.n_frames, 2), np.nan)
    inverse = np.linalg.inv(matrices[valid])
    u, v = IMAGE_CENTRE
    x = inverse[:, 0, 0] * u + inverse[:, 0, 1] * v + inverse[:, 0, 2]
    y = inverse[:, 1, 0] * u + inverse[:, 1, 1] * v + inverse[:, 1, 2]
    w = inverse[:, 2, 0] * u + inverse[:, 2, 1] * v + inverse[:, 2, 2]
    point[valid, 0] = x / w * length
    point[valid, 1] = y / w * width
    return point


def velocity(positions):
    """Per-frame velocity (m/s) of each position, from the displacement over +/- VELOCITY_HALF.

    positions is (n_frames, ...) in metres, usually (n_frames, players, 2); the result has the
    same shape: the centred difference over 2 * VELOCITY_HALF frames, divided by that time. The
    first and last VELOCITY_HALF frames are NaN, and so is any frame where either END of the window
    (frame i-2 or i+2) is NaN; a NaN strictly inside the window does not spread.

        x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]     # one player moving 0.2 m per frame, 25 fps
        velocity(x)                            # -> [nan, nan, 5.0, 5.0, nan, nan]
    """
    k = VELOCITY_HALF
    out = np.full(positions.shape, np.nan)
    out[k:-k] = (positions[2 * k:] - positions[:-2 * k]) * FPS / (2 * k)
    return out


def team_columns(xy, present, speed, sign, opponents_xy, camera):
    """Per-team raw (unsmoothed) features. x is relative to halfway; sign is +1 / -1 / 0 per frame,
    the team's attacking direction.

    xy (n, k, 2), present (n, k) and speed (n, k) are the team's k outfield players; opponents_xy
    (n, m, 2) is every opposing player, NaN where absent; camera (n, 2) is camera_point, also
    relative to halfway. Returns a dict {name in PER_TEAM: (n,) float array}, the quantities in
    the module docstring for this one team, before the +/- 0.5 s (1 s wide) smoothing. near_camera
    is a count, so it is 0 rather than NaN where there is no camera point.

    Toy frame, team attacking right (sign = +1), three outfield players at x = -10, 0, 20:

        advance   -> 3.33     mean of (-10, 0, 20)
        front     -> 16.0     90th percentile
        back      -> -8.0     10th percentile
        spread_x  -> 12.47    standard deviation
        # the same players with sign = -1 give advance -3.33, front 8.0, back -16.0
    """
    x = np.where(present, xy[:, :, 0], np.nan)
    y = np.where(present, xy[:, :, 1], np.nan)
    oriented = x * sign[:, None]
    vel = velocity(np.where(present[:, :, None], xy, np.nan))
    with np.errstate(invalid="ignore"):
        dx = x[:, :, None] - opponents_xy[:, None, :, 0]
        dy = y[:, :, None] - opponents_xy[:, None, :, 1]
        nearest = np.sqrt(np.nanmin(dx * dx + dy * dy, axis=2))
        near_camera = (np.hypot(x - camera[:, 0:1], y - camera[:, 1:2]) < NEAR_CAMERA).sum(axis=1)
        out = {
            "advance": np.nanmean(oriented, axis=1),
            "front": np.nanpercentile(oriented, 90, axis=1),
            "back": np.nanpercentile(oriented, 10, axis=1),
            "spread_x": np.nanstd(x, axis=1),
            "spread_y": np.nanstd(y, axis=1),
            "forward_vel": np.nanmean(vel[:, :, 0] * sign[:, None], axis=1),
            "speed": np.nanmean(np.where(present, speed, np.nan), axis=1),
            "marking": np.nanmean(nearest, axis=1),
            "near_camera": near_camera.astype(np.float64),
        }
    return out


def compute(match):
    """Float32 DataFrame of COLUMNS, indexed by video frame.

    For one Match: team_columns for home (_h) and away (_a), each oriented by its own attacking
    direction and averaged over +/- SMOOTH frames (+/- 0.5 s), then home minus away (_diff), and
    the two camera columns. Row i is video frame i. Off the two graded halves the direction
    sign is 0, so advance, front, back, forward_vel and camera_advance_h are 0 there (not NaN);
    near_camera is 0 where the homography is invalid.

        advance_h = 5.0, advance_a = -3.0    # home pushed up, away dropped deep
        advance_diff                         # -> 8.0, a hint that home has the ball
    """
    length, width = match.pitch
    xy = np.array(match.players_xy, dtype=np.float64)
    xy[:, :, 0] -= length / 2                       # x relative to halfway
    present = np.asarray(match.player_present)
    speed = np.asarray(match.players_speed, dtype=np.float64)
    team = np.asarray(match.team_of_slot)
    outfield = np.ones(len(team), dtype=bool)
    outfield[list(match.goalkeeper_slots)] = False

    sign_h = home_direction(match)
    camera = camera_point(match)
    camera[:, 0] -= length / 2

    columns = {}
    for suffix, side, sign in (("_h", 0, sign_h), ("_a", 1, -sign_h)):
        mine = (team == side) & outfield
        theirs = team != side
        opponents = np.where(present[:, theirs, None], xy[:, theirs], np.nan)
        raw = team_columns(xy[:, mine], present[:, mine], speed[:, mine], sign, opponents, camera)
        for name, values in raw.items():
            columns[name + suffix] = centred(values, SMOOTH)
    for name in PER_TEAM:
        columns[name + "_diff"] = columns[name + "_h"] - columns[name + "_a"]
    columns["camera_advance_h"] = centred(camera[:, 0] * sign_h, SMOOTH)
    columns["camera_y"] = centred(np.abs(camera[:, 1] - width / 2), SMOOTH)
    return pd.DataFrame({name: np.asarray(columns[name], dtype=np.float32) for name in COLUMNS})
