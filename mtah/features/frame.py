"""Per-frame features, one row per video frame, computed from the cache.

This is the shared feature table of the pipeline: compute() turns one cached Match (positions,
speeds, ball, Smart Tagging) into a float32 DataFrame with one row per video frame and one
column per feature. The dead-ball detector (mtah.dead_ball) reads it, and so do the checks in
scripts/. The ball-free team-shape features for possession live next door in team_shape.py.

    from mtah.data import load
    from mtah.features import compute, PLAYER
    match = load()                            # the cached Match
    table = compute(match)                    # DataFrame, n_frames rows
    table[PLAYER]                             # only the player-movement columns

Three groups, kept apart on purpose (see PLAYER, BALL, TAGGING), because the dead-ball detector
builds its confident labels from the ball and the tagging and must not then learn from the same
columns: a model trained on the signals that defined its labels only learns to repeat them.

    PLAYER    how the 22 players move and stand. Defined on every frame with players.
    BALL      the tracked ball and the gap it is in, if any.
    TAGGING   Smart Tagging intervals.

Windows are centred unless named past / future: the whole match is available offline, so a
frame may look ahead. Times are in frames at 25 fps. Positions are in metres with x along the
length (0 at one goal line) and y across the width (0 at one touchline). Near the start and end
of the match a centred window is simply cut short, so the mean is over fewer frames.

In this file: line_distance (helper), player_features, ball_features and tagging_features (one
per group, each a dict of column name -> per-frame array) and compute (all of them, as a table).
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
    """Distance to the nearest pitch line, negative outside the pitch.

    For points (x, y) in metres on a length x width pitch, the distance to the closest of the
    two goal lines and two touchlines: positive inside, zero on a line, negative beyond one (by
    how far it is past that line). NaN in, NaN out. Used for the ball's distance to the lines
    (ball_line_dist, ball_out and the gap-end columns).

        length, width = 105, 68
        line_distance(10, 30, 105, 68)       # -> 10   (10 m from the left goal line)
        line_distance(50, 66, 105, 68)       # -> 2    (2 m from the far touchline)
        line_distance(-2, 30, 105, 68)       # -> -2   (2 m behind the left goal line)
    """
    return np.minimum.reduce([x, length - x, y, width - y])


def player_features(match):
    """The PLAYER columns: how the players move and where they stand, per frame.

    Takes a Match and returns a dict {column name: float array of n_frames}, in PLAYER order.
    Every present player counts, both teams and goalkeepers included. "+/- 0.5 s" etc. is the
    centred mean of the per-frame value over that window. On a frame without players the speed
    and position values (own_half_share included) are NaN and the counts 0, before smoothing.

        n_players               players present on the frame (count, not smoothed)
        speed_mean              mean player speed on the frame (m/s), over players with a speed
        speed_mean_1s/3s/8s     speed_mean averaged over +/- 0.5 s, +/- 1.5 s, +/- 4 s (m/s)
        speed_past_3s           mean of speed_mean over the 3 s before the frame (m/s)
        speed_future_3s         mean of speed_mean over the 3 s after the frame (m/s)
        speed_trend_3s          future minus past (m/s); > 0 when the players speed up
        walk_share_3s           share of players slower than WALK_SPEED, +/- 1.5 s (0..1)
        run_share_3s            share of players faster than RUN_SPEED, +/- 1.5 s (0..1)
        speed_max_1s            speed of the fastest player, +/- 0.5 s (m/s)
        spread_x, spread_y      standard deviation of player x / y, +/- 0.5 s (m)
        centroid_from_halfway   |mean player x - halfway|, mean x over +/- 0.5 s (m)
        n_at_touchline          players within LINE_MARGIN of a touchline or past it, +/- 0.5 s
        n_outside               players more than OUT_MARGIN outside any line, +/- 0.5 s
        n_in_boxes              players inside either penalty box, +/- 0.5 s
        own_half_share          share of players in their own half, +/- 0.5 s (0..1); 1.0 at a
                                kick-off; NaN on frames with no tracked players and
                                outside the two halves

    Toy frame for own_half_share, length 105 so halfway is x = 52.5, home attacks right:

        home players at x = 20, 60      # own half is x < 52.5: one of two
        away players at x = 70, 30      # own half is x > 52.5: one of two
        own_half_share                  # -> 2 / 4 = 0.5 (before smoothing)

    Toy frame for the speeds: three players at 1.0, 3.0 and 5.0 m/s

        speed_mean -> 3.0,  walking share -> 1/3,  running share -> 1/3,  speed_max -> 5.0
    """
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
        counted = present[frames].sum(axis=1)
        own_half[frames] = np.where(counted > 0, own.sum(axis=1) / np.maximum(counted, 1), np.nan)

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
    """The BALL columns: the tracked ball, and the ball gap the frame is in, if any.

    Takes a Match and returns a dict {column name: float array of n_frames}, in BALL order. A
    "gap" is a stretch of frames where the ball is not tracked; the gap columns are 0 (counts)
    or NaN (line distances) on frames where it is tracked.

        ball_present            1 if the ball is tracked on the frame, else 0
        ball_present_3s         share of frames with the ball tracked, +/- 1.5 s (0..1)
        ball_speed_1s           ball speed from its displacement across ~1 s, centred (m/s); NaN
                                if either end of that window has no ball, and at the match ends
        ball_line_dist          line_distance of the ball (m, negative outside; NaN in a gap)
        ball_out                1 if the ball is tracked more than OUT_MARGIN outside the pitch
        gap_length              length of the current gap (frames)
        since_seen              frames since the ball was last tracked (1 on the gap's first)
        until_seen              frames until the ball is tracked again (1 on the gap's last)
        last_seen_line_dist     ball_line_dist on the last tracked frame before the gap (m)
        next_seen_line_dist     ball_line_dist on the first tracked frame after the gap (m)

    Toy run, frames 0..5, the ball lost on frames 2..4 and seen 3 m and 20 m inside the pitch
    either side of the gap:

        ball_present         = [1, 1, 0, 0, 0, 1]
        gap_length           = [0, 0, 3, 3, 3, 0]
        since_seen           = [0, 0, 1, 2, 3, 0]
        until_seen           = [0, 0, 3, 2, 1, 0]
        last_seen_line_dist  = [nan, nan, 3, 3, 3, nan]
        next_seen_line_dist  = [nan, nan, 20, 20, 20, nan]
    """
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
    """The TAGGING columns: 0 / 1 flags from the Smart Tagging intervals, per frame.

    Takes a Match and returns a dict {column name: float array of n_frames}; all zeros when the
    match has no Smart Tagging. Tag codes are matched case-insensitively, and both teams' tags
    count.

        set_piece_window    1 from SET_PIECE_BEFORE (20 s) before to SET_PIECE_AFTER (5 s) after
                            the start of a SET PIECES tag (any type, kick-offs included)
        goal_tag            1 inside a GOALS tag, start_frame to end_frame (half-open)
        in_possession_tag   1 inside a tag with one of IN_POSSESSION_CODES (BUILD UP, ...)

    Toy match at 25 fps with one SET PIECES tag starting at frame 1000:

        set_piece_window[500:1125] == 1      # frames 500..1124, everything else 0
    """
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
    """All features as a float32 DataFrame indexed by video frame.

    The PLAYER, BALL and TAGGING columns of one Match side by side, in that order, row i being
    video frame i (index 0..n_frames-1). This is the `table` the dead-ball detector takes.

        table = compute(match)
        table.shape                          # -> (match.n_frames, 31)
        table.loc[1000, "ball_present"]      # -> 1.0 if the ball is tracked on frame 1000
    """
    columns = {}
    columns.update(player_features(match))
    columns.update(ball_features(match))
    columns.update(tagging_features(match))
    return pd.DataFrame({name: np.asarray(values, dtype=np.float32)
                         for name, values in columns.items()})[PLAYER + BALL + TAGGING]
