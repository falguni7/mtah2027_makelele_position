"""Dead-ball detector: which frames are stoppages (label D), without any human labels.

    from mtah.dead_ball import detect
    dead, p_dead = detect(match, features)      # bool and probability per video frame

Three stages:

  1. confident_labels(): frames whose state is clear from a few strong rules, built from the
     ball and the Smart Tagging (BALL and TAGGING features).
       dead   a goal until its kick-off (GOALS tags), the second half before its kick-off,
              the ball tracked out of the pitch, and the first seconds of a tagged set piece
              (the restart comes ~5 s after the tag start)
       alive  the ball tracked for 3 s and moving inside the pitch, away from any tag, and
              sustained running by the players, ball tracked or not
  2. a classifier trained on those frames using PLAYER features and the ball-gap context, not
     the ball's speed or position, so it cannot just learn the rules back; alive examples inside
     short open-play gaps stop it from learning "ball missing = dead";
  3. decode(): confident frames keep their label, the rest take the classifier's probability,
     smoothed, with hysteresis and minimum durations so stoppages do not flicker.

The step-3 temporal model will replace stage 3; this is the first version.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah import features as F
from mtah.utils import centred, runs_of

# Params
FPS = 25
DEAD, ALIVE, UNKNOWN = 1, 0, -1
OUT_MIN_FRAMES = FPS // 2           # ball out for at least 0.5 s
SET_PIECE_DEAD = 3 * FPS            # dead from the tag start for this long
SET_PIECE_CLEAR = (5 * FPS, 10 * FPS)   # no alive labels this close before / after a tag start
GOAL_END_TRIM = 2 * FPS             # a GOALS tag ends ~1 s after the kick-off
ALIVE_BALL_SPEED = 2.5              # m/s; a dead ball is held, placed or rolled back slowly
ALIVE_LINE_DIST = 2.0               # metres inside the pitch
# A ball gap inside open play: the tracker lost the ball in a crowd, play went on. Alive examples
# inside gaps, so the classifier cannot learn "ball missing = dead".
GAP_ALIVE_MAX = 15 * FPS            # gap no longer than this
GAP_ALIVE_LINE_DIST = 5.0           # ball last seen and reappearing this far inside the pitch
GAP_ALIVE_TEAM_SPEED = 1.5          # m/s, median over the gap
# Sustained running means play is on, ball tracked or not (a stoppage is walked through, and goal
# celebrations are covered by the GOALS tags, which win). Alive examples inside long gaps too.
RUN_ALIVE_SPEED = 2.0               # m/s, team mean over 3 s
RUN_ALIVE_SHARE = 0.12              # share of players above 4 m/s, over 3 s
TRAIN_STRIDE = 5                    # every 5th confident frame: neighbours are near-duplicates
SMOOTH = FPS                        # +/- 1 s mean of the probability
ENTER, LEAVE = 0.6, 0.4             # hysteresis thresholds on the smoothed probability
MIN_DEAD = 2 * FPS
MIN_ALIVE = FPS
# Second-half kick-off, video frame (DATA.md, FAQ.md); the graded frames before it are all D.
KICKOFF_2ND = 113742
MODEL = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             random_state=0)


def confident_labels(match, table, use_set_pieces=True, running_alive=True):
    """int8 per frame: DEAD, ALIVE or UNKNOWN. Only graded frames get a label."""
    n = match.n_frames
    labels = np.full(n, UNKNOWN, dtype=np.int8)
    graded = np.asarray(match.graded)

    alive = (graded
             & (table["ball_present"].to_numpy() > 0)
             & (table["ball_speed_1s"].to_numpy() > ALIVE_BALL_SPEED)
             & (table["ball_present_3s"].to_numpy() == 1)
             & (table["ball_line_dist"].to_numpy() > ALIVE_LINE_DIST))
    dead = np.zeros(n, dtype=bool)
    near_tag = np.zeros(n, dtype=bool)

    tags = match.smart_tagging
    code = tags["code"].str.upper()
    for row in tags[code == "GOALS"].itertuples():
        dead[row.start_frame:max(row.start_frame, row.end_frame - GOAL_END_TRIM)] = True
        near_tag[row.start_frame - SET_PIECE_CLEAR[0]:row.end_frame + SET_PIECE_CLEAR[1]] = True
    for row in tags[code == "SET PIECES"].itertuples():
        near_tag[max(0, row.start_frame - SET_PIECE_CLEAR[0]):
                 row.start_frame + SET_PIECE_CLEAR[1]] = True
        if use_set_pieces and not str(row.type).lower().startswith("kick off"):
            dead[row.start_frame:row.start_frame + SET_PIECE_DEAD] = True

    # Second half before its kick-off: published as all D (DATA.md).
    second_half_start = [first for number, first, _ in match.meta["half_blocks"] if number == 2]
    if second_half_start:
        dead[second_half_start[0]:KICKOFF_2ND] = True

    out = table["ball_out"].to_numpy() > 0
    for start, stop in runs_of(out):
        if stop - start >= OUT_MIN_FRAMES:
            dead[start:stop] = True

    ball_speed = table["ball_speed_1s"].to_numpy()
    team_speed = table["speed_mean"].to_numpy()
    line = table["ball_line_dist"].to_numpy()
    half = FPS // 2 + 1
    for start, stop in runs_of(~np.asarray(match.ball_present)):
        if (stop - start > GAP_ALIVE_MAX or start - half < 0 or stop + half >= n
                or not graded[start:stop].all()):
            continue
        ends_inside = (line[start - 1] > GAP_ALIVE_LINE_DIST
                       and line[stop] > GAP_ALIVE_LINE_DIST)
        ends_moving = (ball_speed[start - half] > ALIVE_BALL_SPEED
                       and ball_speed[stop + half] > ALIVE_BALL_SPEED)
        if ends_inside and ends_moving and np.nanmedian(team_speed[start:stop]) > \
                GAP_ALIVE_TEAM_SPEED:
            alive[start:stop] = True

    if running_alive:
        alive |= ((table["speed_mean_3s"].to_numpy() > RUN_ALIVE_SPEED)
                  & (table["run_share_3s"].to_numpy() > RUN_ALIVE_SHARE))

    labels[graded & alive & ~near_tag] = ALIVE
    labels[graded & dead] = DEAD
    return labels


# Classifier inputs: how the players move, plus where the current ball gap starts and ends. Not
# the ball's speed or position, which define the labels.
GAP_TIMING = ["gap_length", "since_seen", "until_seen"]
GAP_ENDS = ["last_seen_line_dist", "next_seen_line_dist"]
COLUMNS = F.PLAYER + GAP_TIMING + GAP_ENDS


def train(table, labels, columns=COLUMNS):
    """Classifier for P(dead) on confident frames, every TRAIN_STRIDE-th one."""
    rows = np.flatnonzero(labels != UNKNOWN)[::TRAIN_STRIDE]
    model = HistGradientBoostingClassifier(**MODEL)
    model.fit(table.iloc[rows][columns].to_numpy(), labels[rows] == DEAD)
    return model


def predict(model, table, columns=COLUMNS):
    return model.predict_proba(table[columns].to_numpy())[:, 1].astype(np.float32)


def drop_short(mask, value, min_length):
    """Flip runs of `value` shorter than min_length to the other value."""
    out = mask.copy()
    for start, stop in runs_of(out == value):
        if stop - start < min_length:
            out[start:stop] = not value
    return out


def decode(p_dead, labels, graded):
    """Dead mask from probabilities: confident frames pinned, smoothed, hysteresis, durations."""
    p = p_dead.astype(np.float64).copy()
    p[labels == DEAD] = 1.0
    p[labels == ALIVE] = 0.0
    p = centred(p, SMOOTH)

    dead = np.zeros(len(p), dtype=bool)
    state = False
    for i in np.flatnonzero(graded):
        if state and p[i] < LEAVE:
            state = False
        elif not state and p[i] > ENTER:
            state = True
        dead[i] = state
    dead = drop_short(dead, True, MIN_DEAD)
    dead = drop_short(dead, False, MIN_ALIVE)
    dead[labels == DEAD] = True
    return dead & graded


def detect(match, table, use_set_pieces=True, columns=COLUMNS, running_alive=True):
    """(dead mask, raw P(dead), confident labels) for every video frame."""
    labels = confident_labels(match, table, use_set_pieces, running_alive)
    model = train(table, labels, columns)
    p_dead = predict(model, table, columns)
    return decode(p_dead, labels, np.asarray(match.graded)), p_dead, labels
