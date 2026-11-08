"""Possession labels for the 3 Metrica sample matches, from their events, and how they look.

    sbatch --time=00:30:00 scripts/slurm/run.sbatch scripts/external/metrica_labels.py

mtah/external/metrica.py turns each match's human event data into H / A / D per frame (dead ball
from stoppages to restarts, the team of the latest on-ball event, and the controlled-possession
rule of our DFL labels). Metrica's data is public, so per-match numbers may be printed; the
labels themselves are not written anywhere (metrica.labels() remakes them in a second).

Printed per game:

  frames      tracked, graded (kick-off to the half's last event), share D
  stoppages   kept, and dropped because play went on before a restart
  per rule    raw / or / and (controlled.RULES): possessions, mean and median length, the share
              of possessions under 3 s; for scale, DFL's OR labels have ~252 possessions per
              match of mean 12.7 s and our hmm_v11_mix 291 on BAR-ATH
  checks against the tracking, which the labels are not made from:
    ball on D / alive   share of frames with the ball tracked inside the pitch, on D frames and
                        on alive frames: D should have the ball missing or out far more often
    nearest team        on alive frames with the ball tracked, the share where the nearest
                        player to the ball is of the labelled team; also only where that player
                        is within 1 m (the ball at his feet). Loose balls and passes in flight
                        keep the last team, so 1.0 is not expected even for perfect labels
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.external import metrica  # noqa: E402
from mtah.utils import runs_of  # noqa: E402

# Params
SHORT = 3.0                 # seconds: "short" possession
NEAR = 1.0                  # metres: the nearest player has the ball at his feet


def possessions(labels, graded):
    """Lengths in seconds of the possessions (maximal runs of H or A) inside graded runs.

        labels = H H D A A A X, graded = all but the last, 25 fps
        possessions(labels, graded)    # -> [0.08, 0.12]
    """
    out = []
    for a, b in runs_of(graded):
        lab = labels[a:b]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        for s, e in zip(np.r_[0, change], np.r_[change, len(lab)]):
            if lab[s] in ("H", "A"):
                out.append((e - s) / metrica.FPS)
    return np.array(out)


def nearest_team(m):
    """(team 0 / 1 of the player nearest to the ball, his distance in metres) per frame.

    NaN distance (team -1) where the ball or every player is untracked.

        ball at (0.5, 0.5), home player at (0.51, 0.5), away at (0.6, 0.5)
        -> team 0, distance 1.05 m (0.01 of 105 m)
    """
    d = (m.players_xy - m.ball_xy[:, None, :]) * np.array(metrica.PITCH, dtype=np.float32)
    dist = np.sqrt((d ** 2).sum(axis=2))
    has = np.isfinite(dist).any(axis=1)
    best = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    team = np.where(has, m.slot_team[best], -1)
    return team, np.where(has, dist[np.arange(len(best)), best], np.nan)


if __name__ == '__main__':
    for game in metrica.GAMES:
        m = metrica.load(game)
        g = metrica.graded(m)
        is_dead, kept, dropped = metrica.dead(m)
        print("\n== game {}: {} frames, {} events, graded {} ({:.1f} min)".format(
            game, m.n_frames, len(m.events), int(g.sum()), g.sum() / metrica.FPS / 60))
        print("   stoppages kept {}, dropped {} (play went on); D share of graded {:.3f}".format(
            len(kept), dropped, (is_dead & g).sum() / g.sum()))
        print("   {:<5}{:>8}{:>9}{:>9}{:>10}".format("rule", "poss", "mean s", "median",
                                                    "< 3 s"))
        lab = {}
        for rule in ("raw", "or", "and"):
            lab[rule] = metrica.labels(m, rule)
            p = possessions(lab[rule], g)
            print("   {:<5}{:>8d}{:>9.1f}{:>9.1f}{:>10.3f}".format(
                rule, len(p), p.mean(), np.median(p), (p < SHORT).mean()))
        main = lab["or"]
        ball_in = np.isfinite(m.ball_xy).all(axis=1) & (m.ball_xy >= 0).all(axis=1) \
            & (m.ball_xy <= 1).all(axis=1)
        d, alive = g & (main == "D"), g & np.isin(main, ["H", "A"])
        print("   ball tracked inside the pitch: on D {:.3f}, on alive {:.3f}".format(
            ball_in[d].mean(), ball_in[alive].mean()))
        team, dist = nearest_team(m)
        code = np.where(main == "H", 0, 1)
        for name, rule_lab in (("or", main), ("raw", lab["raw"])):
            code = np.where(rule_lab == "H", 0, 1)
            use = alive & ball_in & (team >= 0)
            near = use & (dist <= NEAR)
            print("   nearest team = label ({}): all {:.3f} (n {}), within {:g} m {:.3f} "
                  "(n {})".format(name, (team[use] == code[use]).mean(), int(use.sum()), NEAR,
                                  (team[near] == code[near]).mean(), int(near.sum())))
        print("   frames changed by the rule (raw -> or): {:.3f} of alive".format(
            (lab["raw"][alive] != main[alive]).mean()))
