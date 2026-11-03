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
  depth         front minus back: how far the attacking front is from the defensive line (m)
  hull_area     area of the outline around the outfield players (m^2), small when compact
  approach      mean speed of each player towards their nearest opponent (m/s); > 0 closing
                in, as when pressing, < 0 moving away from them, as when finding space
  marking_rate  how fast marking changes (m/s, from the smoothed marking); < 0 the team is
                getting closer to its opponents, > 0 further away

The last four were added later (ideas 9 and 10: a team closing in on its opponents and staying
compact is likely defending, one moving away and spreading out is likely in possession). The
step-4 method hmm_v2 keeps the original set, COLUMNS_V1; hmm_v3 uses all COLUMNS.

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

from scipy.spatial import ConvexHull, QhullError

from mtah.utils import centred

# Params
FPS = 25
SMOOTH = FPS // 2
NEAR_CAMERA = 15.0
IMAGE_CENTRE = (960.0, 540.0)
VELOCITY_HALF = 2
PER_TEAM_V1 = ["advance", "front", "back", "spread_x", "spread_y", "forward_vel", "speed",
               "marking", "near_camera"]
PER_TEAM = PER_TEAM_V1 + ["depth", "hull_area", "approach", "marking_rate"]


def columns_of(per_team):
    """Table column names for a list of per-team features: _h, _a, _diff of each, then camera.

    Every per-team feature appears in the table three times: once for the home team (suffix _h),
    once for the away team (_a) and once as home minus away (_diff), which is often the most
    telling, since possession is about one team relative to the other. The two camera columns
    come last and are always included, as they belong to neither team. This is how a feature set
    is named for the shape model: COLUMNS_V1 (the step-4 set) and COLUMNS (all features) are
    both built with it, and so is any subset used to compare features, e.g. in
    scripts/possession/shape_check.py.

        columns_of(["advance"])
        # -> ["advance_h", "advance_a", "advance_diff", "camera_advance_h", "camera_y"]
    """
    return ([name + "_h" for name in per_team] + [name + "_a" for name in per_team]
            + [name + "_diff" for name in per_team] + ["camera_advance_h", "camera_y"])


COLUMNS_V1 = columns_of(PER_TEAM_V1)      # the step-4 feature set (hmm_v2)
COLUMNS = columns_of(PER_TEAM)            # everything compute() returns (hmm_v3)


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


def hull_area(xy):
    """Area (m^2) of the convex hull of each frame's points; NaN with fewer than 3 points.

    xy is (n_frames, k, 2) with NaN rows for absent players. The convex hull is the outline you
    get by stretching a rubber band around all the players: a compact defensive block has a
    small one, a team spread across the pitch a large one. Points all on one line give 0.

        one frame, players at (0, 0), (10, 0), (0, 10), (2, 2)
        hull_area(xy)                          # -> [50.0]   the (2, 2) player is inside
    """
    out = np.full(len(xy), np.nan)
    for i, points in enumerate(xy):
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 3:
            continue
        try:
            out[i] = ConvexHull(points).volume       # in 2-D, "volume" is the area
        except QhullError:                           # all on one line (or one point)
            out[i] = 0.0
    return out


def team_columns(xy, present, speed, sign, opponents_xy, camera):
    """Per-team raw (unsmoothed) features. x is relative to halfway; sign is +1 / -1 / 0 per frame,
    the team's attacking direction.

    xy (n, k, 2), present (n, k) and speed (n, k) are the team's k outfield players; opponents_xy
    (n, m, 2) is every opposing player, NaN where absent; camera (n, 2) is camera_point, also
    relative to halfway. Returns a dict {name in PER_TEAM: (n,) float array}, the quantities in
    the module docstring for this one team, before the +/- 0.5 s (1 s wide) smoothing. near_camera
    is a count, so it is 0 rather than NaN where there is no camera point. marking_rate is not
    here: compute() takes it from the smoothed marking.

    Toy frame, team attacking right (sign = +1), three outfield players at x = -10, 0, 20:

        advance   -> 3.33     mean of (-10, 0, 20)
        front     -> 16.0     90th percentile
        back      -> -8.0     10th percentile
        spread_x  -> 12.47    standard deviation
        depth     -> 24.0     front - back
        # the same players with sign = -1 give advance -3.33, front 8.0, back -16.0
        # a player at (0, 0) running at (+2, 0) m/s, the nearest opponent at (5, 0):
        #     approach 2.0 for that player (straight at the opponent); at (0, +2) -> ~0.0
    """
    x = np.where(present, xy[:, :, 0], np.nan)
    y = np.where(present, xy[:, :, 1], np.nan)
    oriented = x * sign[:, None]
    vel = velocity(np.where(present[:, :, None], xy, np.nan))
    with np.errstate(invalid="ignore"):
        dx = x[:, :, None] - opponents_xy[:, None, :, 0]
        dy = y[:, :, None] - opponents_xy[:, None, :, 1]
        squared = np.where(np.isfinite(dx * dy), dx * dx + dy * dy, np.inf)
        closest = squared.argmin(axis=2)[:, :, None]
        nearest = np.sqrt(np.take_along_axis(squared, closest, axis=2)[:, :, 0])
        nearest[~np.isfinite(nearest)] = np.nan
        # unit vector from each player to the nearest opponent, and the player's speed along it
        to_x = -np.take_along_axis(dx, closest, axis=2)[:, :, 0] / nearest
        to_y = -np.take_along_axis(dy, closest, axis=2)[:, :, 0] / nearest
        approach = vel[:, :, 0] * to_x + vel[:, :, 1] * to_y
        front = np.nanpercentile(oriented, 90, axis=1)
        back = np.nanpercentile(oriented, 10, axis=1)
        near_camera = (np.hypot(x - camera[:, 0:1], y - camera[:, 1:2]) < NEAR_CAMERA).sum(axis=1)
        out = {
            "advance": np.nanmean(oriented, axis=1),
            "front": front,
            "back": back,
            "spread_x": np.nanstd(x, axis=1),
            "spread_y": np.nanstd(y, axis=1),
            "forward_vel": np.nanmean(vel[:, :, 0] * sign[:, None], axis=1),
            "speed": np.nanmean(np.where(present, speed, np.nan), axis=1),
            "marking": np.nanmean(nearest, axis=1),
            "near_camera": near_camera.astype(np.float64),
            "depth": front - back,
            "hull_area": hull_area(np.stack([x, y], axis=2)),
            "approach": np.nanmean(approach, axis=1),
        }
    return out


def compute(match):
    """Float32 DataFrame of COLUMNS, indexed by video frame.

    For one Match: team_columns for home (_h) and away (_a), each oriented by its own attacking
    direction and averaged over +/- SMOOTH frames (+/- 0.5 s), then home minus away (_diff), and
    the two camera columns. marking_rate is the rate of change of the smoothed marking, over
    +/- VELOCITY_HALF frames. Row i is video frame i. Off the two graded halves the direction
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
        columns["marking_rate" + suffix] = velocity(columns["marking" + suffix])
    for name in PER_TEAM:
        columns[name + "_diff"] = columns[name + "_h"] - columns[name + "_a"]
    columns["camera_advance_h"] = centred(camera[:, 0] * sign_h, SMOOTH)
    columns["camera_y"] = centred(np.abs(camera[:, 1] - width / 2), SMOOTH)
    return pd.DataFrame({name: np.asarray(columns[name], dtype=np.float32) for name in COLUMNS})
