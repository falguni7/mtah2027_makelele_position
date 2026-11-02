"""Per-frame features, one row per video frame, computed from the cache.

    from mtah.features import compute
    table = compute(match)                    # DataFrame, n_frames rows

Three groups, kept apart on purpose (see PLAYER, BALL, TAGGING), because the dead-ball detector
builds its confident labels from the ball and the tagging and must not then learn from the same
columns: a model trained on the signals that defined its labels only learns to repeat them.

    PLAYER    how the 22 players move and stand. Defined on every frame with players.
    BALL      the tracked ball and the gap it is in, if any.
    TAGGING   Smart Tagging intervals.

Windows are centred unless named past / future: the whole match is available offline, so a
frame may look ahead. Times are in frames at 25 fps.
"""

import numpy as np
import pandas as pd

from mtah.utils import centred, rolling_mean, runs_of

# Params
FPS = 25
WALK_SPEED = 1.5          # m/s; below this a player is walking or standing
RUN_SPEED = 4.0           # m/s; above this a player is running
LINE_MARGIN = 1.5         # metres from a touchline that counts as "at the line"
OUT_MARGIN = 1.0          # metres beyond a line before the ball counts as out
BOX_DEPTH, BOX_WIDTH = 16.5, 40.3
SET_PIECE_BEFORE = 20 * FPS   # Smart Tagging set-piece window, relative to the tag start
SET_PIECE_AFTER = 5 * FPS
IN_POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION")

PLAYER = ["n_players", "speed_mean", "speed_mean_1s", "speed_mean_3s", "speed_mean_8s",
          "speed_past_3s", "speed_future_3s", "speed_trend_3s", "walk_share_3s", "run_share_3s",
          "speed_max_1s", "spread_x", "spread_y", "centroid_from_halfway", "n_at_touchline",
          "n_outside", "n_in_boxes", "own_half_share"]
BALL = ["ball_present", "ball_present_3s", "ball_speed_1s", "ball_line_dist", "ball_out",
        "gap_length", "since_seen", "until_seen", "last_seen_line_dist", "next_seen_line_dist"]
TAGGING = ["set_piece_window", "goal_tag", "in_possession_tag"]


def line_distance(x, y, length, width):
    """Distance to the nearest pitch line, negative outside the pitch."""
    return np.minimum.reduce([x, length - x, y, width - y])


def player_features(match):
    length, width = match.pitch
    present = np.asarray(match.player_present)
    x = np.asarray(match.players_xy[:, :, 0], dtype=np.float64)
    y = np.asarray(match.players_xy[:, :, 1], dtype=np.float64)
    speed = np.where(present, np.asarray(match.players_speed, dtype=np.float64), np.nan)
    n_players = present.sum(axis=1)

    with np.errstate(invalid="ignore"):
        has_speed = (~np.isnan(speed)).sum(axis=1)
        speed_sum = np.nansum(speed, axis=1)
        speed_mean = np.where(has_speed > 0, speed_sum / np.maximum(has_speed, 1), np.nan)
        walking = np.where(has_speed > 0, (speed < WALK_SPEED).sum(axis=1) / np.maximum(has_speed, 1), np.nan)
        running = np.where(has_speed > 0, (speed > RUN_SPEED).sum(axis=1) / np.maximum(has_speed, 1), np.nan)
        speed_max = np.where(has_speed > 0, np.nanmax(np.where(np.isnan(speed), -np.inf, speed), axis=1), np.nan)
        spread_x = np.nanstd(np.where(present, x, np.nan), axis=1)
        spread_y = np.nanstd(np.where(present, y, np.nan), axis=1)
        centroid = np.nanmean(np.where(present, x, np.nan), axis=1)

    at_touchline = present & ((y < LINE_MARGIN) | (y > width - LINE_MARGIN))
    outside = present & ((x < -OUT_MARGIN) | (x > length + OUT_MARGIN)
                         | (y < -OUT_MARGIN) | (y > width + OUT_MARGIN))
    in_box_y = np.abs(y - width / 2) < BOX_WIDTH / 2
    in_boxes = present & in_box_y & ((x < BOX_DEPTH) | (x > length - BOX_DEPTH))

    # Share of players in their own half: 1.0 at a kick-off. Direction from the cache, per half.
    own_half = np.full(match.n_frames, np.nan)
    home = match.team_of_slot == 0
    for number, right in match.home_attacks_right.items():
        frames = np.asarray(match.half) == number
        home_own = x[frames] < length / 2 if right else x[frames] > length / 2
        own = np.where(home, home_own, ~home_own) & present[frames]
        own_half[frames] = own.sum(axis=1) / np.maximum(present[frames].sum(axis=1), 1)

    three = 3 * FPS
    past = rolling_mean(speed_mean, three, -1)
    future = rolling_mean(speed_mean, -1, three)
    return {
        "n_players": n_players.astype(np.float64),
        "speed_mean": speed_mean,
        "speed_mean_1s": centred(speed_mean, FPS // 2),
        "speed_mean_3s": centred(speed_mean, three // 2),
        "speed_mean_8s": centred(speed_mean, 4 * FPS),
        "speed_past_3s": past,
        "speed_future_3s": future,
        "speed_trend_3s": future - past,
        "walk_share_3s": centred(walking, three // 2),
        "run_share_3s": centred(running, three // 2),
        "speed_max_1s": centred(speed_max, FPS // 2),
        "spread_x": centred(spread_x, FPS // 2),
        "spread_y": centred(spread_y, FPS // 2),
        "centroid_from_halfway": np.abs(centred(centroid, FPS // 2) - length / 2),
        "n_at_touchline": centred(at_touchline.sum(axis=1), FPS // 2),
        "n_outside": centred(outside.sum(axis=1), FPS // 2),
        "n_in_boxes": centred(in_boxes.sum(axis=1), FPS // 2),
        "own_half_share": centred(own_half, FPS // 2),
    }


def ball_features(match):
    length, width = match.pitch
    present = np.asarray(match.ball_present)
    bx = np.asarray(match.ball_xy[:, 0], dtype=np.float64)
    by = np.asarray(match.ball_xy[:, 1], dtype=np.float64)
    n = match.n_frames

    # Speed from positions over 1 s, centred; the ATD ball speed field is not usable.
    half = FPS // 2
    speed = np.full(n, np.nan)
    speed[half:n - half] = np.hypot(bx[2 * half:] - bx[:-2 * half],
                                    by[2 * half:] - by[:-2 * half]) / (2 * half / FPS)
    line = line_distance(bx, by, length, width)

    gap_length = np.zeros(n)
    since = np.zeros(n)
    until = np.zeros(n)
    last_line = np.full(n, np.nan)
    next_line = np.full(n, np.nan)
    for start, stop in runs_of(~present):
        span = np.arange(stop - start)
        gap_length[start:stop] = stop - start
        since[start:stop] = span + 1
        until[start:stop] = stop - start - span
        if start > 0:
            last_line[start:stop] = line[start - 1]
        if stop < n:
            next_line[start:stop] = line[stop]

    return {
        "ball_present": present.astype(np.float64),
        "ball_present_3s": centred(present.astype(np.float64), 3 * FPS // 2),
        "ball_speed_1s": speed,
        "ball_line_dist": line,
        "ball_out": (present & (line < -OUT_MARGIN)).astype(np.float64),
        "gap_length": gap_length,
        "since_seen": since,
        "until_seen": until,
        "last_seen_line_dist": last_line,
        "next_seen_line_dist": next_line,
    }


def tagging_features(match):
    n = match.n_frames
    out = {name: np.zeros(n) for name in TAGGING}
    tags = match.smart_tagging
    if tags is None:
        return out
    code = tags["code"].str.upper()
    for row in tags[code == "SET PIECES"].itertuples():
        out["set_piece_window"][max(0, row.start_frame - SET_PIECE_BEFORE):
                                row.start_frame + SET_PIECE_AFTER] = 1
    for row in tags[code == "GOALS"].itertuples():
        out["goal_tag"][row.start_frame:row.end_frame] = 1
    for row in tags[code.isin(IN_POSSESSION_CODES)].itertuples():
        out["in_possession_tag"][row.start_frame:row.end_frame] = 1
    return out


def compute(match):
    """All features as a float32 DataFrame indexed by video frame."""
    columns = {}
    columns.update(player_features(match))
    columns.update(ball_features(match))
    columns.update(tagging_features(match))
    return pd.DataFrame({name: np.asarray(values, dtype=np.float32)
                         for name, values in columns.items()})[PLAYER + BALL + TAGGING]
