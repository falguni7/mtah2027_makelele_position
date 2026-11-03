"""Agreement with Smart Tagging: an independent, automatic opinion on which team has the ball.

Smart Tagging's in-possession phases (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING
TRANSITION) name the team in possession. They are inferred automatically by Metrica, they are
phases rather than possessions, and they contain errors -- so this is a consistency check, not
accuracy. It stops being independent the moment a method uses those tags as input.

Two steps: phase_team() turns the tags into a per-frame team ("H", "A" or "" where no
in-possession phase covers the frame), and agreement() compares a method's labels with it.

    from mtah.validation import tagging
    tagged = tagging.phase_team(match)
    tagging.agreement(labels, tagged)
    # -> e.g. {"frames": 1000, "agree": 0.8, "disagree": 0.15, "dead": 0.05}
"""

import numpy as np

from mtah.features.frame import IN_POSSESSION_CODES


def phase_team(match):
    """Per frame: "H", "A" or "" -- the team of the in-possession phase covering it, graded only.

    Only the in-possession phase codes (IN_POSSESSION_CODES, matched case-insensitively) that
    name a team are used; each covers its half-open [start_frame, end_frame) on the video
    timeline. Where phases overlap, the one listed later in the tags wins. Frames outside the
    graded halves, and frames no phase covers, are "".

        tags: BUILD UP     team H   frames 2..4      (start_frame 2, end_frame 5)
              PROGRESSION  team A   frames 6..6      (start_frame 6, end_frame 7)
        graded = all True on frames 0..7
        phase_team(match)              # -> ["", "", H, H, H, "", A, ""]
    """
    out = np.full(match.n_frames, "", dtype="<U1")
    tags = match.smart_tagging
    phases = tags[tags["code"].str.upper().isin(IN_POSSESSION_CODES) & tags["team"].notna()]
    for row in phases.itertuples():
        out[row.start_frame:row.end_frame] = row.team
    out[~np.asarray(match.graded)] = ""
    return out


def agreement(labels, tagged):
    """{agree, disagree, dead}: shares of tagged frames where labels say the same team, the other
    team, or D.

    labels is a method's per-frame output ("H" / "A" / "D" / "X"), tagged the output of
    phase_team(). Only frames with a tagged team count; "frames" is how many there are. The three
    shares are of those frames; any label that is not the tagged team and not D (the other team,
    or X) counts as disagree. Shares are NaN when no frame is tagged.

        labels = [H, H, A, D, A, H]
        tagged = [H, H, H, H, "", ""]
        agreement(labels, tagged)
        # -> {"frames": 4, "agree": 0.5, "disagree": 0.25, "dead": 0.25}
    """
    covered = tagged != ""
    labels = np.asarray(labels)[covered]
    team = tagged[covered]
    return {
        "frames": int(covered.sum()),
        "agree": float((labels == team).mean()),
        "disagree": float(((labels != team) & (labels != "D")).mean()),
        "dead": float((labels == "D").mean()),
    }
