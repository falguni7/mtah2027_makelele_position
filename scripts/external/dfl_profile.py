"""What do the DFL labels look like? Per-match statistics of possession and dead balls.

    python scripts/external/dfl_profile.py
    sbatch scripts/slurm/run.sbatch scripts/external/dfl_profile.py

Parses (first run, ~1 min per match) and profiles the 7 DFL Bundesliga matches of
mtah/external/dfl.py. These are other matches with public labels, so their numbers can be
looked at freely; they are compared with what the challenge says about our ground truth and
with the public 5 minutes of BAR-ATH (17 possessions of mean 14.2 s, ~20 % D; DATA.md).

Per match, labelled H / A / D the way our ground truth is built (D where BallStatus is dead,
else the team BallPossession names):

  frames, minutes      both halves together
  H / A / D            shares of the frames
  poss, mean s         possessions as the official scorer counts them (maximal H or A runs inside a
                       half), and their mean length; also the median and the share under 2 s
  D n, median s        dead stretches, their median length, the share under 2 s and over 30 s
  D out                share of dead stretches whose start has the ball outside the pitch (more
                       than 0.5 m past a line): the ball went out; the rest are fouls, offside,
                       goals, injuries ... called with the ball on the pitch
  restart other        share of dead stretches after which a different team has the ball than
                       before (BallPossession just before and just after the stretch)
  owner flips in D     share of dead stretches during which BallPossession changes at all: the
                       tracking re-assigns the dead ball to the team that will restart
  changes in play      possession changes H <-> A with no D in between, per 90 minutes
  restarts (events)    number of restart events (kick-off, throw-in, free kick, corner, goal kick,
                       penalty) in the event file, next to the number of D stretches

Then the same over the first 5 minutes of play of each match (like our public window), so the
two can be compared directly. Nothing is written except the parse cache.
"""

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.external import dfl  # noqa: E402
from mtah.features.frame import line_distance  # noqa: E402
from mtah.utils import runs_of  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
OUT_MARGIN = 0.5            # metres past a line for "ball out"
RESTARTS = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")
WINDOW = 5 * 60 * FPS       # first 5 minutes of play


def half_runs(half):
    """[(start, stop)] of each half's rows (frames are in a row, halves one after the other).

        half_runs([1, 1, 1, 2, 2])           # -> [(0, 3), (3, 5)]
    """
    half = np.asarray(half)
    return [r for h in (1, 2) for r in runs_of(half == h)]


def profile(labels, possession, alive, ball_xy, runs):
    """Dict of the per-match numbers listed in the module docstring, for the given rows.

    labels are H / A / D per row, possession 1 / 2 per row, alive bool, ball_xy (n, 2) in our
    convention, runs the [(start, stop)] stretches to count inside (halves).

        labels = H H D D A A (25 fps), possession = 1 1 1 2 2 2, ball out at row 2
        profile(...)["restart_other"]        # -> 1.0   home before the stoppage, away after
    """
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, runs)
    lengths = np.array([b - a for a, b, *_ in spells]) / FPS
    dead = [(a + s, b + s) for s, e in runs for a, b in runs_of(~alive[s:e])]
    d_len = np.array([b - a for a, b in dead]) / FPS
    line = line_distance(ball_xy[:, 0], ball_xy[:, 1], *dfl.PITCH)
    n = sum(e - s for s, e in runs)
    inner = [(a, b) for a, b in dead if a > 0 and b < len(alive)]
    changes = 0
    for s, e in runs:
        lab = labels[s:e]
        team = lab != "D"
        idx = np.flatnonzero(team)
        changes += int(((lab[idx][1:] != lab[idx][:-1]) & (np.diff(idx) == 1)).sum())
    return {
        "frames": n, "minutes": n / FPS / 60,
        "H": np.mean([(labels[s:e] == "H").mean() for s, e in runs]),
        "A": np.mean([(labels[s:e] == "A").mean() for s, e in runs]),
        "D": sum((labels[s:e] == "D").sum() for s, e in runs) / n,
        "poss": len(spells), "mean_s": lengths.mean(), "median_s": np.median(lengths),
        "poss_under_2s": (lengths < 2).mean(),
        "D_n": len(dead), "D_median_s": np.median(d_len) if len(d_len) else np.nan,
        "D_under_2s": (d_len < 2).mean() if len(d_len) else np.nan,
        "D_over_30s": (d_len > 30).mean() if len(d_len) else np.nan,
        "D_out": np.mean([line[a] < -OUT_MARGIN for a, _ in dead]) if dead else np.nan,
        "restart_other": np.mean([possession[a - 1] != possession[b] for a, b in inner])
        if inner else np.nan,
        "owner_flips_in_D": np.mean([len(set(possession[a:b])) > 1 for a, b in dead])
        if dead else np.nan,
        "changes_in_play_per90": changes / (n / FPS / 60) * 90,
    }


def restart_events(match_id):
    """Number of restart events (RESTARTS) in the match's event file.

        restart_events("DFL-MAT-J03WMX")     # -> e.g. 101
    """
    text = open(dfl.paths(match_id)[1], encoding="utf-8").read()
    return sum(len(re.findall("<{}[ >/]".format(kind), text)) for kind in RESTARTS)


if __name__ == '__main__':
    rows = []
    for match_id in dfl.MATCHES:
        m = dfl.load(match_id)
        labels = m.labels
        runs = half_runs(m.half)
        full = profile(labels, m.possession, m.alive, m.ball_xyz[:, :2], runs)
        full["restart_events"] = restart_events(match_id)
        first = [(runs[0][0], runs[0][0] + WINDOW)]
        five = profile(labels, m.possession, m.alive, m.ball_xyz[:, :2], first)
        rows.append((match_id, full, five, m.players_xy.shape[1]))
        print("{}: {} frames, {} player slots".format(match_id, m.n_frames, m.players_xy.shape[1]),
              flush=True)

    keys = ["minutes", "H", "A", "D", "poss", "mean_s", "median_s", "poss_under_2s", "D_n",
            "D_median_s", "D_under_2s", "D_over_30s", "D_out", "restart_other",
            "owner_flips_in_D", "changes_in_play_per90"]
    for title, k in (("whole match", 1), ("first 5 minutes of play", 2)):
        print("\n" + title)
        print("  {:<16}".format("match") + "".join("{:>9}".format(c[:9]) for c in keys)
              + ("{:>9}".format("restarts") if k == 1 else ""))
        for row in rows:
            d = row[k]
            print("  {:<16}".format(row[0]) + "".join(
                "{:>9.0f}".format(d[c]) if c in ("poss", "D_n") else "{:>9.2f}".format(d[c])
                for c in keys) + ("{:>9d}".format(d["restart_events"]) if k == 1 else ""))
