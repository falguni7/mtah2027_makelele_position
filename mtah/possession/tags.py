"""Smart Tagging's in-possession phases as team evidence for the temporal model (hmm_v15_tag).

    from mtah.possession import tags
    q = tags.phase_reliability(match, events)              # e.g. 0.823 on BAR-ATH
    p_home = tags.add_phase_evidence(p_home, match, events)

Smart Tagging (automatic tactical tagging, part of the provided data) marks in-possession phases
(BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION; mtah.features.frame.
IN_POSSESSION_CODES) with the team in possession. They cover ~40 % of the graded frames of
BAR-ATH, ball gaps included, where our own team evidence is weakest.

How much to trust a phase is measured on the match itself, without labels: on the frames of the
step-2 control events (mtah/possession/control.py, our surest team frames) that lie inside a
phase, q is the share where the event's team is the phase's team (0.823 on BAR-ATH). A phase
then counts as one more observation of the team, added in log-odds to the shape model's P(home):

    logit P'(home) = logit P(home) + logit(q)    inside a home phase
                   = logit P(home) - logit(q)    inside an away phase

which is the Bayes update for an independent observation that is right with probability q. The
temporal model uses P(home) only on frames far from any control event (shape_model.gap_frames,
the same frames as the shape evidence) and weighs it with TEAM_WEIGHT, so control events and the
change points they set are not touched, and a phase can never outvote a control event.

The "DEFENDING IN ..." / "DEFENSIVE TRANSITION" codes are the same intervals with the other
team (checked: identical counts and lengths), so they add nothing and are not used.
"""

import numpy as np

from mtah.validation.tagging import phase_team

# Params
Q_CLIP = (0.55, 0.95)          # a phase is never treated as certain, nor as worthless


def phase_reliability(match, events):
    """q: share of control-event frames inside an in-possession phase whose team is the phase's.

    events are (start, stop, team) control events, team 0 home / 1 away. Only graded frames
    count (phase_team is "" elsewhere). Clipped to Q_CLIP; 0.5 (no information) when no event
    frame lies in a phase.

        # phases: home on frames 0-9; events: home 2-5 (4 frames), away 6-7 (2 frames)
        phase_reliability(match, [(2, 6, 0), (6, 8, 1)])        # -> 4 / 6 = 0.667
    """
    team = phase_team(match)
    agree = total = 0
    for start, stop, code in events:
        t = team[start:stop]
        t = t[t != ""]
        total += len(t)
        agree += int((t == ("H" if code == 0 else "A")).sum())
    if total == 0:
        return 0.5
    return float(np.clip(agree / total, *Q_CLIP))


def add_phase_evidence(p_home, match, events, q=None):
    """P(home) with every in-possession phase added as an observation right with probability q.

    p_home is the per-frame P(home) of the shape model(s), NaN where it has no opinion; inside a
    phase a NaN becomes the phase's opinion alone (q or 1 - q). q defaults to
    phase_reliability(match, events). Frames outside phases are returned unchanged.

        # q = 0.8 (logit 1.386); a home phase on frame 1, an away phase on frame 2
        add_phase_evidence([0.5, 0.5, 0.5, nan], match, events, q=0.8)
        # -> [0.5, 0.8, 0.2, nan]
        add_phase_evidence([0.8], ...)   # inside a home phase -> 0.941 (logit 1.386 + 1.386)
    """
    if q is None:
        q = phase_reliability(match, events)
    team = phase_team(match)
    p = np.asarray(p_home, np.float64).copy()
    shift = np.where(team == "H", 1.0, np.where(team == "A", -1.0, 0.0)) * np.log(q / (1 - q))
    inside = team != ""
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.where(np.isnan(p), 0.0, np.log(p / (1 - p)))
    p[inside] = 1 / (1 + np.exp(-(base[inside] + shift[inside])))
    return p.astype(np.float32)
