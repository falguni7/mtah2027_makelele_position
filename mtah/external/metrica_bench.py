"""The whole pipeline on the 3 Metrica sample matches, scored against labels from their events.

    from mtah.external import benchmark as bm, metrica_bench as mb
    ctx = bm.context()                                      # our player and ball gap statistics
    shares = mb.in_play_shares(ctx)                         # {"random": 0.409, "dead": 0.336}
    match, truth = mb.bench_match(1, ctx, "random", shares["random"])
    ev = bm.evidence(match, truth, "mix", mb.team_table(match), mb.model_probs(match, 1, "random", ctx))
    pred = bm.decode(match, ev, bm.DEFAULT)
    mb.score(truth, pred)                                   # -> {"spellF1": ..., "macroF1": ...}

A second benchmark next to the DFL one (mtah/external/benchmark.py), with labels from another
source: Metrica's human event data (mtah/external/metrica.py, the OR rule; the strict "and"
labels as a sensitivity check). It compares our methods at today's fixed weights. It never
chooses weights or trains anything that goes into the submission.

How a Metrica match is made to look like ours (bench_match):

  players     Metrica's sample tracking is cleaned: all 22 players on every graded frame, never
              a drop-out (measured 2026-11-07). They are degraded as the DFL players are
              (adapter.degrade_players: our drop-out lengths, 20.5 of 22 visible on average,
              0.15 m jitter), with seed DEGRADE_SEED + game, the same for both placements.
  ball        Metrica's ball is missing almost exactly on the dead-ball frames (tracked inside
              the pitch on 0.2-7 % of D frames, ~97 % of alive ones), so as it is it would give
              D away. It is also hidden IN PLAY (hide_in_play), in gaps of our ball gap lengths
              starting on alive frames, until the same share of alive frames has no ball as in
              the DFL benchmark: in_play_shares() measures it there, 0.409 for the "random"
              placement and 0.336 for "dead" (2026-11-07). The two placements keep the DFL names
              so the two benchmarks line up. The dead frames stay as Metrica has them (mostly
              without a ball), so the total share missing is above our 41 %.
  speeds      Metrica has no speeds: they are the positions' central difference over
              +/- SPEED_HALF frames, in m/s (speeds()), NaN where a player is hidden.
  halves      graded from each half's kick-off to its last event (metrica.graded); the HMM
              decodes each half on its own as on our match.
  missing     as on DFL: no whistles, no camera, no Smart Tagging (benchmark.NO_TAGS), so the
              core of the method runs, in the same variants (benchmark.VARIANTS).

The DFL models (boosting shape model and the TCN + LSTM sequence model) never saw Metrica, so
their predictions are out of sample without any leave-one-out (model_probs). The sequence model
is also fine-tuned on each Metrica match's own confident frames, out of fold, exactly as on
ours (shape_sequence.for_match); those frames come from the ball, so this is done per placement.
Both are cached in cache/external/metrica/.

Dead ball. Metrica's ball vanishes at stoppages instead of being tracked out of the pitch, so
the dead-ball detector, without Smart Tagging, has almost no confident DEAD frames to learn from
(first run, 2026-11-07: D recall 0.07). The benchmark therefore gives the decoder the true dead
ball (true_dead()) and tests the team side: which team the methods name while the ball is alive.
That is where hmm_v9_deadA, hmm_v10A and hmm_v11_mix differ most; the DFL benchmark covers the
dead ball.

In this file: speeds(), goalkeeper_slots(), as_match() (the Match-like object), hide_in_play(),
in_play_shares(), bench_match(), team_table(), model_probs(), true_dead(), team_accuracy() and
score().
"""

import os

import numpy as np

from mtah.external import adapter, metrica
from mtah.external import benchmark as bm
from mtah.utils import runs_of

# Params
FPS = metrica.FPS
PLACEMENTS = bm.PLACEMENTS              # ("random", "dead"), named after the DFL settings
DEGRADE_SEED = 500                      # players of game k degraded with seed DEGRADE_SEED + k
BALL_SEED = 2000                        # ball of game k: BALL_SEED + 10 k (+ 1 for "dead")
SPEED_HALF = 2                          # frames either side of the speed's central difference
CACHE = metrica.CACHE


def speeds(xy):
    """(n, slots) speed in m/s from (n, slots, 2) positions in metres, NaN where unknown.

    The central difference over +/- SPEED_HALF frames divided by its time (2 * SPEED_HALF / 25
    s); the first and last SPEED_HALF frames and every frame whose neighbours are NaN are NaN.

        a player moving 0.2 m per frame along x (5 m/s at 25 fps), 10 frames
        speeds(xy)[:, 0]               # -> [nan, nan, 5, 5, 5, 5, 5, 5, nan, nan]
    """
    xy = np.asarray(xy, dtype=np.float64)
    out = np.full(xy.shape[:2], np.nan)
    k = SPEED_HALF
    d = xy[2 * k:] - xy[:-2 * k]
    out[k:-k] = np.sqrt((d ** 2).sum(axis=2)) / (2 * k / FPS)
    return out.astype(np.float32)


def goalkeeper_slots(m):
    """(home goalkeeper slot, away goalkeeper slot): per team, the slot deepest on average.

    Among each team's slots tracked on at least half of the first half's graded frames, the
    one whose mean x is farthest from the halfway line over those frames: the starting
    goalkeeper stays near his goal. The sample files have no goalkeeper change.

        home slots at mean x 0.05 (keeper), 0.30, 0.45; away at 0.95 (keeper), 0.60, 0.70
        goalkeeper_slots(m)            # -> (0, 3)
    """
    first = metrica.graded(m) & (m.half == 1)
    x = m.players_xy[first, :, 0]
    tracked = np.isfinite(x).mean(axis=0) >= 0.5
    with np.errstate(invalid="ignore"):
        depth = np.abs(np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0) - 0.5)
    out = []
    for team in (0, 1):
        slots = np.flatnonzero(tracked & (m.slot_team == team))
        out.append(int(slots[np.argmax(depth[slots])]))
    return tuple(out)


class MetricaAsMatch:
    """The attributes of our Match that the pipeline reads, filled from a MetricaMatch.

    players_xy (n, slots, 2) metres, players_speed (n, slots) m/s, player_present, team_of_slot,
    goalkeeper_slots, half, graded, home_attacks_right {half: bool}, pitch (105, 68), fps 25,
    n_frames, ball_xy (metres), ball_present, homography (NaN), homography_valid (all False),
    smart_tagging (empty) and meta (no half_blocks). Built by as_match().
    """


def as_match(m, rng=None, gaps=None):
    """A MetricaAsMatch of the MetricaMatch m: players degraded when rng and gaps are given.

    Positions go from Metrica's 0-1 to metres (x 105, y 68; (0, 0) is the top-left corner, as
    for our ATD). Speeds come from the clean positions (speeds()) and are NaN where a player is
    hidden. home_attacks_right: in a half where the home keeper's mean x is in the left half of
    the pitch, home attacks right. The ball is present where Metrica tracks it.

        match = as_match(metrica.load(1))
        match.home_attacks_right              # -> e.g. {1: False, 2: True}
        match.player_present[match.graded].sum(axis=1).mean()    # -> 22.0 (clean)
    """
    out = MetricaAsMatch()
    scale = np.array(metrica.PITCH)
    clean = np.asarray(m.players_xy, dtype=np.float64) * scale
    xy = adapter.degrade_players(clean, rng, gaps) if rng is not None and gaps is not None \
        else clean
    present = np.isfinite(xy).all(axis=2)
    out.players_xy = xy.astype(np.float32)
    out.player_present = present
    out.players_speed = np.where(present, speeds(clean), np.nan).astype(np.float32)
    out.team_of_slot = np.asarray(m.slot_team)
    out.goalkeeper_slots = goalkeeper_slots(m)
    out.half = np.asarray(m.half)
    out.n_frames = m.n_frames
    out.graded = metrica.graded(m)
    out.pitch = metrica.PITCH
    out.fps = FPS
    keeper = clean[:, out.goalkeeper_slots[0], 0]
    out.home_attacks_right = {h: bool(np.nanmean(keeper[out.graded & (out.half == h)])
                                      < metrica.PITCH[0] / 2) for h in (1, 2)}
    out.ball_xy = (np.asarray(m.ball_xy, dtype=np.float64) * scale).astype(np.float32)
    out.ball_present = np.isfinite(out.ball_xy).all(axis=1)
    out.homography = np.full((m.n_frames, 3, 3), np.nan)
    out.homography_valid = np.zeros(m.n_frames, dtype=bool)
    out.smart_tagging = bm.NO_TAGS
    out.meta = {"half_blocks": []}
    return out


def hide_in_play(alive, half, lengths, share, rng):
    """Bool mask: gaps of our lengths starting on alive frames until `share` of them is hidden.

    Gaps are added one at a time, each with a length drawn from `lengths` (frames) and a start
    drawn uniformly among the alive frames, cut at the end of its half; overlapping gaps merge.
    Only alive frames count towards the share (a gap may run on into a stoppage, where the ball
    is mostly missing anyway). rng is a numpy Generator.

        alive: 1000 frames, all alive, one half; lengths [10]; share 0.2
        hide_in_play(alive, half, np.array([10]), 0.2, np.random.default_rng(0)).mean()
        # -> 0.2 or a little above (stops at the first gap that reaches 200 alive frames)
    """
    alive = np.asarray(alive, dtype=bool)
    half = np.asarray(half)
    n = len(alive)
    stops = np.r_[np.flatnonzero(np.diff(half) != 0) + 1, n]
    half_end = stops[np.searchsorted(stops, np.arange(n), side="right")]
    candidates = np.flatnonzero(alive)
    target = int(np.ceil(share * len(candidates)))
    mask = np.zeros(n, dtype=bool)
    hidden = 0
    while hidden < target:
        for start, length in zip(rng.choice(candidates, 256), rng.choice(lengths, 256)):
            stop = min(start + int(length), half_end[start])
            hidden += int((alive[start:stop] & ~mask[start:stop]).sum())
            mask[start:stop] = True
            if hidden >= target:
                break
    return mask


def in_play_shares(ctx):
    """{placement: mean share of alive graded frames with the ball hidden} in the DFL benchmark.

    Runs benchmark.bench_match on the 7 DFL matches for each placement (cached features, a few
    seconds each) and averages the share of alive graded frames without a ball.

        in_play_shares(bm.context())  # -> {"random": 0.409, "dead": 0.336}   (2026-11-07)
    """
    from mtah.external import dfl
    out = {}
    for placement in PLACEMENTS:
        shares = []
        for match_id in dfl.MATCHES:
            match, truth = bm.bench_match(match_id, ctx, placement)
            alive = match.graded & (truth.labels != "D")
            shares.append(float((~match.ball_present[alive]).mean()))
        out[placement] = float(np.mean(shares))
    return out


def bench_match(game, ctx, placement, share, rule="or"):
    """(MetricaAsMatch made to look like ours, benchmark.Truth) for one game and placement.

    Players degraded with seed DEGRADE_SEED + game and our drop-out lengths (ctx, from
    benchmark.context()); the ball hidden in play on `share` of the alive frames (hide_in_play,
    seed BALL_SEED + 10 game, + 1 for "dead"), alive meaning not D in the labels. Truth holds
    metrica.labels(m, rule) (X outside the graded part), the half per row, k = game and
    match_id "metrica_game<k>".

        match, truth = bench_match(1, ctx, "random", 0.409)
        (~match.ball_present[match.graded & (truth.labels != "D")]).mean()   # -> about 0.43
        # 0.409 hidden by us plus Metrica's own ~3 % in play
    """
    if placement not in PLACEMENTS:
        raise ValueError("placement must be one of {}".format(PLACEMENTS))
    m = metrica.load(game)
    labels = metrica.labels(m, rule)
    match = as_match(m, np.random.default_rng(DEGRADE_SEED + game), ctx["player_gaps"])
    alive = match.graded & np.isin(labels, ["H", "A"])
    rng = np.random.default_rng(BALL_SEED + 10 * game + (placement == "dead"))
    hidden = hide_in_play(alive, m.half, ctx["ball_lengths"], share, rng)
    match.ball_present = match.ball_present & ~hidden
    match.ball_xy = np.where(match.ball_present[:, None], match.ball_xy, np.nan).astype(np.float32)
    return match, bm.Truth(labels=labels, half=np.asarray(m.half), k=game,
                           match_id="metrica_game{}".format(game))


def team_table(match):
    """Our shape model's feature table (shape_transfer.COLUMNS) for a benchmark match.

    The same features as on DFL (no camera columns): mtah/features/team_shape.py on the
    degraded players. Not cached (~4 minutes a match).

        team_table(match).shape        # -> (n_frames, 36)
    """
    from mtah.external import shape_transfer as st
    from mtah.features import team_shape
    return team_shape.compute(match)[st.COLUMNS]


def model_probs(match, game, placement, ctx, pool=False):
    """{"boosting": (n, 3), "sequence": (n, 3)} DFL-model P(H), P(A), P(D) on a benchmark match.

    boosting: shape_transfer.for_match, trained on all 7 DFL matches (players degraded with our
    drop-out lengths), cached per game (it reads no ball, so both placements share it).
    sequence: shape_sequence.for_match, pretrained on DFL and fine-tuned on this match's own
    confident frames out of fold (they come from the ball, so cached per game and placement;
    ~7 minutes on a GPU the first time). Frames off the graded halves are NaN.

    pool=True: the same models with the OTHER two Metrica games added to their training data
    (mtah/external/pool.py, exclude=game, so the tested game is never trained on with its
    labels; hmm_v12), cached under "pool" names.

        p = model_probs(match, 1, "random", ctx)
        p["sequence"][match.graded].sum(axis=1)[:2]    # -> [1., 1.]
        model_probs(match, 1, "random", ctx, pool=True)   # trained on 7 DFL + games 2 and 3
    """
    from mtah.external import pool as training_pool
    from mtah.external import shape_sequence, shape_transfer
    base = os.path.join(CACHE, "game{}_".format(game) + ("pool_" if pool else ""))
    extra = training_pool.metrica_entries(ctx["player_gaps"], exclude=game) if pool else ()
    return {"boosting": shape_transfer.for_match(match, path=base + "shape_v{}.npy".format(
                shape_transfer.FEATURE_VERSION), gaps=ctx["player_gaps"], extra=extra),
            "sequence": shape_sequence.for_match(match, path=base + "{}_sequence_v{}.npy".format(
                placement, shape_sequence.VERSION), gaps=ctx["player_gaps"], extra=extra)}


def true_dead(ev, truth):
    """The evidence dict ev with the dead ball taken from the labels: D pinned where truth says D.

    Metrica's ball vanishes at stoppages instead of being tracked out of the pitch, so without
    Smart Tagging the dead-ball detector gets almost no confident DEAD frames (607 against
    46,411 ALIVE in game 1) and calls D on 4 % of frames, against 40 % in the labels. The
    benchmark therefore tests the TEAM side on Metrica (agreed 2026-11-07): every graded frame
    is pinned, DEAD where truth.labels is D and ALIVE where it is H or A, with p_dead 1 / 0
    (clipped by the HMM), so the decoder picks only between H and A, from the control events,
    the carry between them and the team evidence. The DFL benchmark covers the dead ball.

        truth.labels = X H D D A X
        out = true_dead(ev, truth)
        out["dead_labels"]             # -> [UNKNOWN, ALIVE, DEAD, DEAD, ALIVE, UNKNOWN]
        out["p_dead"]                  # -> [nan, 0, 1, 1, 0, nan]
    """
    from mtah.dead_ball.detector import ALIVE, DEAD, UNKNOWN
    labels = np.asarray(truth.labels)
    out = dict(ev)
    out["dead_labels"] = np.where(labels == "D", DEAD,
                                  np.where(np.isin(labels, ["H", "A"]), ALIVE, UNKNOWN)
                                  ).astype(np.int8)
    out["p_dead"] = np.where(labels == "D", 1.0,
                             np.where(np.isin(labels, ["H", "A"]), 0.0, np.nan)
                             ).astype(np.float32)
    return out


def team_accuracy(truth, pred):
    """Share of the truly alive graded frames (H or A in truth) where pred names the same team.

        truth.labels = X H H A D, pred = D H A A D
        team_accuracy(truth, pred)     # -> 0.667 (2 of the 3 alive frames)
    """
    alive = np.isin(truth.labels, ["H", "A"])
    return float((np.asarray(pred)[alive] == truth.labels[alive]).mean())


def score(truth, pred):
    """benchmark.score on the graded frames only (truth.labels not X).

        truth.labels = X H H A A X, pred = D H H A A D, halves 1
        score(truth, pred)             # -> spellF1 1.0 (the X frames are left out), macroF1
                                       #    0.667: no graded D frame, so D's F1 counts as 0
    """
    graded = np.isin(truth.labels, ["H", "A", "D"])
    return bm.score(truth.labels[graded], np.asarray(pred)[graded], truth.half[graded])
