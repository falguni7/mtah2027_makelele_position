"""How the tactical camera turns: per-frame features from the homography, ball-free.

    from mtah.features import camera
    table = camera.compute(match)            # DataFrame, one row per video frame
    table.loc[1000, "cam_turn"]              # how fast the camera turns there, pixels per second

The tactical camera stands in one place and barely zooms, but it rotates all the time: its
operator pans left and right (and tilts a little) to keep the ball in the picture. So the way it
turns says something about the ball even where the tracker has lost it:

  * in open play the ball moves and the camera turns with it; in a stoppage (a throw-in being
    taken, a free kick being set up, a goal celebration) it turns slowly or rests;
  * when one team attacks, the camera swings towards that team's attacking goal;
  * when the ball goes out or is placed for a restart, the camera points near a line.

How the turning is measured. The homography maps the pitch to the image on every frame. Take
the pitch point at the image centre on frame t - LAG and see where that same point appears in
the image on frame t + LAG: if the camera turned right, the point has moved left in the image,
by an amount that depends on how fast the camera turned, not on how far away the point is. So
the measure is in pixels per second, close to proportional to the camera's turning speed in
degrees per second. Measuring the aim point's speed on the pitch in m/s instead would be
misleading: the same turn moves the aim point much further on the far side of the pitch than
near the camera.

Columns of compute(), each smoothed over +/- SMOOTH frames (0.5 s):

  cam_pan            sideways turn, pixels per second; positive = turning right in the image
  cam_tilt           up-down turn, pixels per second; positive = tilting down
  cam_turn           overall turning speed, hypot(cam_pan, cam_tilt), pixels per second
  cam_turn_3s        cam_turn averaged over +/- 1.5 s, the slower trend
  cam_advance_rate   how fast the aim point moves towards the home team's attacking goal, m/s
                     along the pitch's length (negative = towards the away team's); 0 off the
                     two halves, where there is no attacking direction
  cam_line_dist      distance of the aim point from the nearest pitch line, metres (negative
                     outside the pitch)

Before smoothing, a value is NaN where a homography it needs is invalid or not finite; the
smoothing ignores NaN, so after it a column is NaN only where every frame of its window was. The camera's aim point is
mtah.features.team_shape.camera_point (the pitch point of the image centre). Nothing here reads
the ball, the players or the tags.
"""

import numpy as np
import pandas as pd

from mtah.features.frame import line_distance
from mtah.features.team_shape import IMAGE_CENTRE, camera_point, home_direction
from mtah.utils import centred

# Params
FPS = 25
LAG = 2                     # frames before and after: the turn is measured over 4 frames (0.16 s)
SMOOTH = FPS // 2           # +/- 0.5 s smoothing of every column
TREND = 3 * FPS // 2        # +/- 1.5 s for cam_turn_3s
COLUMNS = ["cam_pan", "cam_tilt", "cam_turn", "cam_turn_3s", "cam_advance_rate",
           "cam_line_dist"]


def project(matrices, points, length, width):
    """Image pixels (u, v) of pitch points (x, y) in metres, through per-frame homographies.

    matrices is (n, 3, 3), each mapping the NORMALISED pitch (x / length, y / width, in 0..1) to
    pixels, as in the cache; points is (n, 2) in metres, one point per matrix. Returns (n, 2)
    pixels: u = (h00 x' + h01 y' + h02) / w, v = (h10 x' + h11 y' + h12) / w, with
    w = h20 x' + h21 y' + h22 and x', y' the normalised point. NaN in, NaN out.

        H = diag(1920, 1080, 1)                  # normalised pitch -> a 1920 x 1080 image
        project([H], [(52.5, 34.0)], 105, 68)    # -> [[960, 540]]   centre spot -> image centre
        project([H], [(105.0, 0.0)], 105, 68)    # -> [[1920, 0]]
    """
    m = np.asarray(matrices, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0] / length, p[:, 1] / width
    w = m[:, 2, 0] * x + m[:, 2, 1] * y + m[:, 2, 2]
    u = (m[:, 0, 0] * x + m[:, 0, 1] * y + m[:, 0, 2]) / w
    v = (m[:, 1, 0] * x + m[:, 1, 1] * y + m[:, 1, 2]) / w
    return np.column_stack([u, v])


def turn_rate(matrices, aim, length, width, lag=LAG):
    """(pan, tilt) per frame, pixels per second, unsmoothed: how fast the camera turns.

    aim is (n, 2), the pitch point at the image centre on every frame (camera_point), NaN where
    unknown; matrices the (n, 3, 3) homographies. For frame t, the aim point of frame t - lag is
    projected into the image of frame t + lag. The camera turned right by as much as that point
    moved left, so pan = -(u - centre_u) / (2 lag / 25) and tilt = -(v - centre_v) / (2 lag / 25).
    The first and last `lag` frames, and frames where either homography or aim point is
    missing, are NaN.

        # the image slides 10 pixels to the left per frame: the camera turns right
        # (H_t = H_0 shifted by -10 t pixels in u), lag = 2 -> pan = 10 * 25 = 250 px/s
        pan, tilt = turn_rate(matrices, aim, 105, 68)
        pan[2:-2]                                # -> [250, 250, ...]
        tilt[2:-2]                               # -> [0, 0, ...]
    """
    n = len(aim)
    pan = np.full(n, np.nan)
    tilt = np.full(n, np.nan)
    if n <= 2 * lag:
        return pan, tilt
    later = np.asarray(matrices, dtype=np.float64)[2 * lag:]
    earlier_aim = np.asarray(aim, dtype=np.float64)[:-2 * lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        pixel = project(later, earlier_aim, length, width)
    seconds = 2 * lag / FPS
    pan[lag:-lag] = -(pixel[:, 0] - IMAGE_CENTRE[0]) / seconds
    tilt[lag:-lag] = -(pixel[:, 1] - IMAGE_CENTRE[1]) / seconds
    return pan, tilt


def compute(match):
    """All camera columns (COLUMNS) as a float32 DataFrame indexed by video frame.

    match is the cached Match (homography, homography_valid, pitch, half, home_attacks_right).
    The turning speed comes from turn_rate(), the aim point from team_shape.camera_point(); see
    the module docstring for the meaning of each column. Frames whose homography is flagged
    invalid count as missing: they give no turn or aim value of their own, and a column is NaN
    where its whole smoothing window is missing.

        table = compute(match)
        list(table.columns)          # -> COLUMNS
        table.shape                  # -> (match.n_frames, 6)
        np.nanmedian(table["cam_turn"][match.graded])   # -> ~4 px/s on this match: the camera
                                                        #    is still most of the time
    """
    length, width = match.pitch
    aim = camera_point(match)
    matrices = np.asarray(match.homography, dtype=np.float64).copy()
    matrices[~np.isfinite(aim).all(axis=1)] = np.nan
    pan, tilt = turn_rate(matrices, aim, length, width)
    turn = np.hypot(pan, tilt)

    sign = home_direction(match)
    advance = np.full(match.n_frames, np.nan)
    advance[1:-1] = (aim[2:, 0] - aim[:-2, 0]) * FPS / 2
    advance *= sign

    columns = {
        "cam_pan": centred(pan, SMOOTH),
        "cam_tilt": centred(tilt, SMOOTH),
        "cam_turn": centred(turn, SMOOTH),
        "cam_turn_3s": centred(turn, TREND),
        "cam_advance_rate": centred(advance, SMOOTH),
        "cam_line_dist": centred(line_distance(aim[:, 0], aim[:, 1], length, width), SMOOTH),
    }
    return pd.DataFrame({name: np.asarray(columns[name], dtype=np.float32) for name in COLUMNS})
