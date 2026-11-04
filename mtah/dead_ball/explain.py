"""Which long ball gaps have a reason to be a stoppage, and which have none.

    from mtah import features
    from mtah.audio.whistle import match_whistles
    from mtah.dead_ball.explain import explain_gaps, unexplained_frames
    gaps = explain_gaps(match, features.compute(match), match_whistles())
    mask = unexplained_frames(match.n_frames, gaps)     # frames of gaps with no reason

The dead-ball detector calls most frames of long ball gaps dead (about 85%), because a long gap
looks like a stoppage from the players' movement and the gap's length alone. But the tracker
also loses the ball in open play, in a crowd or high in the air. A real stoppage almost always
leaves a trace of WHY play stopped:

  whistle    the referee whistled shortly before or during the gap, or just before it ended
             (a foul, offside, an injury, a restart)                -- mtah/audio/whistle.py
  line       the ball was last seen, or seen again, near a line of the pitch (it went out for a
             throw-in, corner or goal kick, or was placed there for the restart)
  tag        Smart Tagging has a set piece or a goal starting during the gap or around it

A ball gap is a run of graded frames without a tracked ball (inside one half). This module
looks at every gap of at least MIN_GAP seconds and says which of the three reasons it has; a
gap with none is "unexplained", and a long dead stretch there is suspicious: more likely the
ball lost in open play than a stoppage. The reasons are rules from how football works, with
windows chosen from those arguments, not tuned on labels.

In this file: ball_gaps() finds the gaps, explain_gaps() gives each its reasons,
unexplained_frames() marks the frames of the gaps that have none, and soften() lowers P(dead)
there (the odds of dead divided by UNEXPLAINED_ODDS, 4), which is how method hmm_v6 uses it:
a stoppage in an unexplained gap needs stronger evidence, but is never ruled out.
"""

import numpy as np

from mtah.utils import runs_of

# Params
FPS = 25
MIN_GAP = 3 * FPS               # frames; shorter gaps are left alone (mostly called alive already)
LINE_NEAR = 5.0                 # metres; ball last / next seen this close to a line, or outside
WHISTLE_WINDOW = (-2.0, 3.0)    # seconds: from 2 s before the gap starts to 3 s after it ends
TAG_WINDOW = (-5.0, 5.0)        # seconds: the same for set-piece and goal tag starts
TAG_CODES = ("SET PIECES", "GOALS")


def ball_gaps(match, min_length=MIN_GAP):
    """[(start, stop)] runs of graded frames without a tracked ball, at least min_length long.

    Frames are half-open. A gap never crosses half-time, because only graded frames count.

        ball_present = [1, 0, 0, 0, 1, 0, 1]          # all graded
        ball_gaps(match, min_length=2)                # -> [(1, 4)]   the 1-frame gap is too short
    """
    missing = np.asarray(match.graded) & ~np.asarray(match.ball_present)
    return [(a, b) for a, b in runs_of(missing) if b - a >= min_length]


def explain_gaps(match, table, whistles, min_length=MIN_GAP):
    """For every ball gap of at least min_length frames: its frames and which reasons it has.

    match is the cached Match (with match.smart_tagging), table the feature table of
    mtah.features.compute (for last_seen_line_dist and next_seen_line_dist), whistles the
    whistle events (start_s, stop_s, ...) of mtah.audio.whistle.match_whistles(). Returns a list
    of dicts {"start", "stop", "whistle", "line", "tag"}, one per gap in time order, where

      whistle   a whistle starts between WHISTLE_WINDOW[0] (2 s) before the gap's first frame
                and WHISTLE_WINDOW[1] (3 s) after its last
      line      the ball was last seen before the gap, or first seen after it, at most
                LINE_NEAR (5 m) inside a line or outside the pitch
      tag       a SET PIECES or GOALS tag (any type, kick-offs included) starts between
                TAG_WINDOW[0] (5 s) before the gap and TAG_WINDOW[1] (5 s) after it

    A gap at the very start or end of the tracking (no ball seen on one side) uses only the
    side that exists for "line".

        # a 10 s gap from 100 s to 110 s, the ball last seen 20 m and next seen 25 m inside,
        # one whistle at 101.5 s, no tags nearby
        explain_gaps(match, table, [(101.5, 101.9, 3800.0, 22.0)])
        # -> [{"start": 2500, "stop": 2750, "whistle": True, "line": False, "tag": False}]
        # the same gap without the whistle -> all three False: unexplained
    """
    whistle_starts = np.sort(np.array([e[0] for e in whistles], dtype=np.float64))
    tags = match.smart_tagging
    tag_starts = np.sort(tags.loc[tags["code"].str.upper().isin(TAG_CODES),
                                  "start_frame"].to_numpy(dtype=np.float64)) / FPS
    last_line = table["last_seen_line_dist"].to_numpy()
    next_line = table["next_seen_line_dist"].to_numpy()

    def any_in(starts, lo, hi):
        """True if at least one value of the sorted array starts lies in [lo, hi] (both ends
        included). Used for "is there a whistle (or a tag) in this window around the gap?".

            any_in(np.array([3.0, 10.0]), 9.0, 12.0)     # -> True    10 s is inside
            any_in(np.array([3.0, 10.0]), 4.0, 9.5)      # -> False   nothing in 4..9.5 s
        """
        return bool(np.searchsorted(starts, hi, side="right") > np.searchsorted(starts, lo))

    out = []
    for start, stop in ball_gaps(match, min_length):
        first_s, last_s = start / FPS, (stop - 1) / FPS
        ends = np.array([last_line[start], next_line[start]])
        out.append({
            "start": start, "stop": stop,
            "whistle": any_in(whistle_starts, first_s + WHISTLE_WINDOW[0],
                              last_s + WHISTLE_WINDOW[1]),
            "line": bool(np.any(ends[np.isfinite(ends)] <= LINE_NEAR)),
            "tag": any_in(tag_starts, first_s + TAG_WINDOW[0], last_s + TAG_WINDOW[1]),
        })
    return out


def unexplained_frames(n_frames, gaps):
    """Bool mask of the frames of the gaps (from explain_gaps()) that have none of the reasons.

        gaps = [{"start": 2, "stop": 5, "whistle": False, "line": False, "tag": False},
                {"start": 7, "stop": 9, "whistle": True, "line": False, "tag": False}]
        unexplained_frames(10, gaps)
        # -> [F, F, T, T, T, F, F, F, F, F]   the second gap has a whistle
    """
    mask = np.zeros(n_frames, dtype=bool)
    for g in gaps:
        if not (g["whistle"] or g["line"] or g["tag"]):
            mask[g["start"]:g["stop"]] = True
    return mask


# How much less likely a dead ball is in an unexplained gap: its odds are divided by this.
UNEXPLAINED_ODDS = 4.0


def soften(p_dead, mask, factor=UNEXPLAINED_ODDS):
    """P(dead) with the odds of dead divided by `factor` on the frames of mask, as float32.

    p_dead is the dead-ball classifier's probability per frame, mask a bool mask such as
    unexplained_frames(). On those frames p becomes p / (p + factor * (1 - p)): strong evidence
    stays strong but needs more of it, weak evidence turns into evidence for play. The other
    frames keep their value. Confident labels are not touched here; the temporal model pins
    them on its own.

        p_dead = [0.5, 0.8, 0.8, 0.98]
        mask   = [T,   T,   F,   T]
        soften(p_dead, mask)        # -> [0.2, 0.5, 0.8, 0.925]
    """
    p = np.asarray(p_dead, dtype=np.float64).copy()
    q = p[mask]
    p[mask] = q / (q + factor * (1.0 - q))
    return p.astype(np.float32)
