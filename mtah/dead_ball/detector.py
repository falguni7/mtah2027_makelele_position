"""Dead-ball detector: which frames are stoppages (label D), without any human labels.

    from mtah import features
    from mtah.data import load
    from mtah.dead_ball import detect
    match = load()
    table = features.compute(match)
    dead, p_dead, labels = detect(match, table)   # bool, probability and label per video frame

dead is what control_dead and baseline50_dead in mtah.methods treat as D; the temporal model
(mtah.temporal.hmm) takes p_dead and labels instead, as the dead-ball evidence.

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
     short open-play gaps stop it from learning "ball missing = dead". Optionally also the
     referee's whistles (WHISTLE: seconds since the last whistle and until the next one, from
     mtah/audio/whistle.py) and how fast the tactical camera turns (CAMERA, from
     mtah/features/camera.py), which both exist inside ball gaps just as well;
  3. decode(): confident DEAD frames are pinned dead; confident ALIVE frames only enter as
     probability 0 before smoothing, so hysteresis and minimum durations can still turn them
     dead. The rest take the classifier's probability, smoothed, with hysteresis and minimum
     durations so stoppages do not flicker.

The temporal model (mtah/temporal/hmm.py) uses p_dead and the confident labels directly and does
not use decode(); decode() is what control_dead and baseline50_dead in mtah.methods use.

In this file: confident_labels (stage 1), train and predict (stage 2), drop_short and decode
(stage 3) and detect, which runs all three. Labels are the ints DEAD = 1, ALIVE = 0 and
UNKNOWN = -1; only graded frames (match.graded) are ever labelled or marked dead.
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
    """int8 per frame: DEAD, ALIVE or UNKNOWN. Only graded frames get a label.

    Stage 1. match is the cached Match, which must have match.smart_tagging (it is required; a
    match without tagging raises), table the feature table of mtah.features.compute. A
    frame is DEAD if any dead rule fires, else ALIVE if any alive rule fires and it is not near
    a tag, else UNKNOWN; DEAD wins over ALIVE.

      dead    inside a GOALS tag, up to GOAL_END_TRIM (2 s) before its end (~ the kick-off);
              the first SET_PIECE_DEAD (3 s) from a SET PIECES tag start, kick-offs excepted
              (only if use_set_pieces); the second half from its first video frame up to
              KICKOFF_2ND; the ball tracked out (ball_out) for at least OUT_MIN_FRAMES (0.5 s)
      alive   the ball tracked, tracked for all of +/- 1.5 s, faster than ALIVE_BALL_SPEED and
              more than ALIVE_LINE_DIST inside the pitch; a whole open-play ball gap (at most
              GAP_ALIVE_MAX, every frame of the gap graded, at least 13 frames from both ends
              of the match, both ends more than GAP_ALIVE_LINE_DIST inside and moving fast,
              i.e. ball_speed_1s above ALIVE_BALL_SPEED 13 frames before the gap and 13 frames
              after it, players' median speed over GAP_ALIVE_TEAM_SPEED); sustained running
              (speed_mean_3s above RUN_ALIVE_SPEED and run_share_3s above RUN_ALIVE_SHARE, if
              running_alive)
      near a tag   from 5 s before to 10 s after a SET PIECES tag start, or from 5 s before
              the start to 10 s after the end of a GOALS tag; never ALIVE

    Toy frames, all graded and far from any tag:

        ball tracked, 20 m inside, moving 8 m/s for 3 s          -> ALIVE
        ball tracked 3 m behind the goal line for 1 s            -> DEAD
        ball lost 2 s at midfield, seen moving fast either side,
            players jogging at 3 m/s                             -> ALIVE (whole gap)
        ball lost 30 s, players walking                          -> UNKNOWN
        1 s after the start of a corner (SET PIECES) tag         -> DEAD
        8 s after it, ball moving fast inside the pitch          -> UNKNOWN (near the tag)
    """
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
        near_tag[max(0, row.start_frame - SET_PIECE_CLEAR[0]):
                 row.end_frame + SET_PIECE_CLEAR[1]] = True
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
# Optional: when the referee last whistled and next whistles (mtah.audio.whistle.timing_features).
WHISTLE = ["since_whistle", "until_whistle"]
# Optional: how fast the tactical camera turns, over 1 s and 3 s (mtah.features.camera). The
# camera follows the ball, so it rests in a stoppage and turns in play, ball tracked or not.
CAMERA = ["cam_turn", "cam_turn_3s"]


def train(table, labels, columns=COLUMNS):
    """Classifier for P(dead) on confident frames, every TRAIN_STRIDE-th one.

    Stage 2. Fits a gradient-boosted tree classifier (settings in MODEL) on the given feature
    columns of the frames labelled DEAD or ALIVE by confident_labels, target "is DEAD". Only
    every TRAIN_STRIDE-th (5th) confident frame is used, as neighbouring frames are nearly
    identical. Returns the fitted model for predict.

        labels = [U, D, D, D, D, D, D, A, A, A, A, A]     # U = UNKNOWN
        # confident frames 1..11, training rows 1, 6, 11 -> targets True, True, False
    """
    rows = np.flatnonzero(labels != UNKNOWN)[::TRAIN_STRIDE]
    model = HistGradientBoostingClassifier(**MODEL)
    model.fit(table.iloc[rows][columns].to_numpy(), labels[rows] == DEAD)
    return model


def predict(model, table, columns=COLUMNS):
    """P(dead) per frame from the trained classifier, as float32.

    Every row of table is scored, confident frames included, with the same columns the model
    was trained on. Values are 0..1; decode turns them into the dead mask.

        p_dead = predict(model, table)       # e.g. [0.02, 0.05, 0.91, ...], one per frame
    """
    return model.predict_proba(table[columns].to_numpy())[:, 1].astype(np.float32)


def drop_short(mask, value, min_length):
    """Flip runs of `value` shorter than min_length to the other value.

    mask is a boolean array per frame; every stretch of consecutive `value` frames lasting fewer
    than min_length frames is set to `not value`, longer ones are kept. Returns a new array.
    decode uses it to drop stoppages under MIN_DEAD and play bursts under MIN_ALIVE.

        mask = [T, T, F, T, T, T, F, F]
        drop_short(mask, True, 3)            # -> [F, F, F, T, T, T, F, F]
        drop_short(mask, False, 2)           # -> [T, T, T, T, T, T, F, F]
    """
    out = mask.copy()
    for start, stop in runs_of(out == value):
        if stop - start < min_length:
            out[start:stop] = not value
    return out


def decode(p_dead, labels, graded):
    """Dead mask from probabilities: confident frames pinned, smoothed, hysteresis, durations.

    Stage 3. p_dead is the classifier's probability per frame, labels the confident labels,
    graded the graded-frame mask. Steps: set confident DEAD frames to 1 and ALIVE to 0; average
    over +/- SMOOTH (1 s); walk each graded half, starting alive, and switch to dead when the
    smoothed value goes above ENTER (0.6), back to alive when it goes below LEAVE (0.4); drop
    dead stretches shorter than MIN_DEAD (2 s), then alive stretches shorter than MIN_ALIVE
    (1 s), measured inside the half; finally every confident DEAD frame is dead again. Nothing
    carries across half-time. Returns a bool mask, False off the graded frames.

        smoothed p = [0.5, 0.7, 0.5, 0.45, 0.3]          # ignoring the duration rules
        dead       = [F,   T,   T,   T,    F]            # stays dead until p < 0.4
    """
    p = p_dead.astype(np.float64).copy()
    p[labels == DEAD] = 1.0
    p[labels == ALIVE] = 0.0
    p = centred(p, SMOOTH)

    # Each graded half on its own: the state starts alive at a half's first frame, and runs are
    # measured inside the half, so nothing leaks across half-time.
    dead = np.zeros(len(p), dtype=bool)
    for start, stop in runs_of(np.asarray(graded)):
        half = np.zeros(stop - start, dtype=bool)
        state = False
        for i in range(stop - start):
            if state and p[start + i] < LEAVE:
                state = False
            elif not state and p[start + i] > ENTER:
                state = True
            half[i] = state
        half = drop_short(half, True, MIN_DEAD)
        half = drop_short(half, False, MIN_ALIVE)
        dead[start:stop] = half
    dead[labels == DEAD] = True
    return dead & graded


def detect(match, table, use_set_pieces=True, columns=COLUMNS, running_alive=True,
           whistles=None, camera=None):
    """(dead mask, raw P(dead), confident labels) for every video frame.

    The whole detector: confident_labels, train, predict, decode. match is the cached Match
    (match.smart_tagging must be present; a match without tagging raises), table the feature
    table of mtah.features.compute; use_set_pieces and running_alive switch those
    confident-label rules, columns picks the classifier inputs. whistles, if given, are whistle
    events (mtah.audio.whistle.match_whistles()); their WHISTLE timing columns are then added to
    the table (a copy) and to the classifier inputs. camera, if given, is the camera table of
    mtah.features.camera.compute(match); its CAMERA columns are added the same way. Neither
    changes the confident labels, only what the classifier sees. Returns a bool array
    (dead = D), a float32 array of the classifier's P(dead) before decoding and the int8
    labels, all of length n_frames.

        dead, p_dead, labels = detect(match, table)
        dead.mean()                          # share of frames called dead
        detect(match, table, use_set_pieces=False)   # without the set-piece rule
        detect(match, table, whistles=match_whistles())   # with the whistle features
        detect(match, table, camera=camera.compute(match))  # with the camera's turning speed
    """
    if whistles is not None:
        from mtah.audio.whistle import timing_features
        table = table.assign(**timing_features(match.n_frames, whistles))
        columns = list(columns) + [c for c in WHISTLE if c not in columns]
    if camera is not None:
        table = table.assign(**{c: camera[c].to_numpy() for c in CAMERA})
        columns = list(columns) + [c for c in CAMERA if c not in columns]
    labels = confident_labels(match, table, use_set_pieces, running_alive)
    model = train(table, labels, columns)
    p_dead = predict(model, table, columns)
    return decode(p_dead, labels, np.asarray(match.graded)), p_dead, labels
