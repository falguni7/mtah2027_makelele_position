"""Controlled possession for DFL: H / A / D labels that, like our ground truth, ignore mere touches.

    from mtah.external import dfl, dfl_events, controlled
    m = dfl.load("DFL-MAT-J03WMX")
    events = dfl_events.load(m)
    labels = controlled.labels(m, events)      # H / A / D per frame, OR rule, T = 3.75 s

DFL's tracking names an owner on every frame (BallPossession) and switches it at every touch,
duels included: a third of its possessions are under 2 s. CHALLENGE.md asks for controlled
possession instead: the possession changes only when the other team gains control (a touch, a
deflection or a lost duel does not count, and a loose ball stays with the last team). This
module turns DFL's owner into that.

The tracking owner is cut into runs: maximal stretches of alive frames with the same owner,
inside one half (a dead ball ends a run). Walking through a half in order, a run of the team
that already has the ball just continues it. A run of the OTHER team is a real change if it
passes the rule; otherwise it is given to the team that had the ball before (as a touch that
did not win the ball). The first run of a half is always kept. Dead frames stay D, exactly as
BallStatus says, and the team "having the ball" carries over a stoppage, so a restart by the
other team is a change only when it passes the rule too (a throw-in is a Play, so it does).

The rules, for a run of the other team:

  raw        always a change: DFL's owner as it is (dfl.DFLMatch.labels)
  duration   the run lasts at least min_seconds
  events     the team makes a controlled action in it (mtah/external/dfl_events.py: a pass,
             cross, shot, claimed ball or won duel), within TOLERANCE frames of the run
  or         duration or events: the main label (agreed 2026-11-05)
  and        duration and events: stricter, a sensitivity check

min_seconds (T) is chosen on DFL itself, by how well the duration rule alone agrees with the
events rule (scripts/external/dfl_labels.py), never by matching our public 5 minutes: 3.75 s
(MIN_SECONDS). With it the OR label has ~250 possessions per match, mean 12.7 s.

In this file: runs() (the owner runs of one half), with_action() (which runs hold a controlled
action of their team), labels() (the relabelled H / A / D) and spell_f1() (agreement of two
labellings with the official scorer).
"""

import numpy as np

import possession_metrics as pm  # evaluation/, on the path via mtah/__init__.py
from mtah.utils import runs_of

# Params
FPS = 25
TOLERANCE = FPS // 2        # an event this many frames outside a run still counts for it
MIN_SECONDS = 3.75          # T, chosen by scripts/external/dfl_labels.py (best duration-vs-events
                            # agreement on the 7 matches; 3.0-3.75 s are within 0.004)
RULES = ("raw", "duration", "events", "or", "and")


def runs(owner, alive, start, stop):
    """[(first row, row after the last, team)] of the alive owner runs in rows start..stop-1.

    owner is 0 / 1 per row (home / guest), alive a bool per row. A run is a maximal stretch of
    consecutive alive rows with the same owner; a dead row ends it, and so does a change of
    owner. Rows outside [start, stop) are ignored, so a run never crosses a half.

        owner = [0, 0, 1, 1, 1, 0, 0], alive = [T, T, T, F, T, T, T]
        runs(owner, alive, 0, 7)       # -> [(0, 2, 0), (2, 3, 1), (4, 5, 1), (5, 7, 0)]
    """
    out = []
    owner = np.asarray(owner)
    for a, b in runs_of(np.asarray(alive)[start:stop]):
        a, b = a + start, b + start
        cuts = np.flatnonzero(np.diff(owner[a:b])) + 1 + a
        edges = [a] + cuts.tolist() + [b]
        out += [(s, e, int(owner[s])) for s, e in zip(edges[:-1], edges[1:])]
    return out


def with_action(run_list, events, tolerance=TOLERANCE):
    """bool per run: does the run's team make a controlled action in it (or within tolerance)?

    events is a dfl_events.load() table. A run (a, b, team) has an action when some controlled
    event of that same team has its row in [a - tolerance, b + tolerance): the event clock is
    synced to the tracking to a few frames, not exactly, so an interception logged a moment
    before the tracking switches owner still belongs to the run it starts.

        run_list = [(0, 50, 0), (50, 60, 1)], one controlled guest event at row 48
        with_action(run_list, events)    # -> [False, True]   48 is within 12 frames of 50
    """
    use = events[events.controlled & (events.row >= 0)]
    rows_of = {team: np.sort(use.loc[use.team == team, "row"].to_numpy()) for team in (0, 1)}
    out = []
    for a, b, team in run_list:
        r = rows_of[team]
        i = np.searchsorted(r, a - tolerance)
        out.append(bool(i < len(r) and r[i] < b + tolerance))
    return np.array(out, dtype=bool)


def labels(match, events, rule="or", min_seconds=MIN_SECONDS, tolerance=TOLERANCE):
    """H / A / D per row of a DFLMatch, with possession changes only where `rule` accepts them.

    Every half is walked run by run (runs()). The team in possession starts as the owner of the
    half's first run. A run of that team keeps it. A run of the other team is accepted as a
    change when the rule (module docstring) says so, and then that team is in possession;
    otherwise the run is relabelled to the team in possession. Dead rows are D. With
    rule="raw" the result equals match.labels.

    Toy half, 25 fps, home owns rows 0-99, guest 100-109 (0.4 s, no guest event), home 110-199:
        labels(m, events, rule="or", min_seconds=2)    # -> H on all 200 rows: a touch
    and with a controlled guest event (say a pass) at row 105 instead:
        labels(m, events, rule="or", min_seconds=2)    # -> H 0-99, A 100-109, H 110-199
    """
    if rule not in RULES:
        raise ValueError("rule must be one of {}".format(RULES))
    owner = np.asarray(match.possession).astype(np.int64) - 1
    alive = np.asarray(match.alive)
    half = np.asarray(match.half)
    team = owner.copy()
    min_frames = min_seconds * FPS
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if len(rows) == 0:
            continue
        run_list = runs(owner, alive, rows[0], rows[-1] + 1)
        action = with_action(run_list, events, tolerance)
        current = None
        for (a, b, who), act in zip(run_list, action):
            if current is None or who == current or rule == "raw":
                current = who
                continue
            long = (b - a) >= min_frames
            accept = {"duration": long, "events": act, "or": long or act,
                      "and": long and act}[rule]
            if accept:
                current = who
            else:
                team[a:b] = current
    out = np.where(team == 0, "H", "A").astype("<U1")
    out[~alive] = "D"
    return out


def spell_f1(reference, other, half):
    """Official spellF1 (mean over IoU 0.3-0.7) of `other` scored against `reference`.

    Both are H / A / D label arrays of one DFLMatch, half its half per row; every half is a
    scored run, as the two halves are for BAR-ATH. This is how two DFL labellings are compared,
    e.g. the duration rule against the events rule when choosing T. 1.0 means the two have the
    same possession spells.

        reference = H H H A A A, other = H H H A A A    (one half)
        spell_f1(reference, other, half)                # -> 1.0
    """
    runs_ = [r for h in (1, 2) for r in runs_of(np.asarray(half) == h)]
    return pm.spell_scores(pm.label_codes(reference), pm.label_codes(other), runs_)["mean_f1"]
