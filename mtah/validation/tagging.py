"""Agreement with Smart Tagging: an independent, automatic opinion on which team has the ball.

Smart Tagging's in-possession phases (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING
TRANSITION) name the team in possession. They are inferred automatically by Metrica, they are
phases rather than possessions, and they contain errors -- so this is a consistency check, not
accuracy. It stops being independent the moment a method uses those tags as input.
"""

import numpy as np

from mtah.features.frame import IN_POSSESSION_CODES


def phase_team(match):
    """Per frame: "H", "A" or "" -- the team of the in-possession phase covering it, graded only."""
    out = np.full(match.n_frames, "", dtype="<U1")
    tags = match.smart_tagging
    phases = tags[tags["code"].str.upper().isin(IN_POSSESSION_CODES) & tags["team"].notna()]
    for row in phases.itertuples():
        out[row.start_frame:row.end_frame] = row.team
    out[~np.asarray(match.graded)] = ""
    return out


def agreement(labels, tagged):
    """{agree, disagree, dead}: shares of tagged frames where labels say the same team, the other
    team, or D."""
    covered = tagged != ""
    labels = np.asarray(labels)[covered]
    team = tagged[covered]
    return {
        "frames": int(covered.sum()),
        "agree": float((labels == team).mean()),
        "disagree": float(((labels != team) & (labels != "D")).mean()),
        "dead": float((labels == "D").mean()),
    }
