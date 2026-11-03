"""Temporal models over the whole match (roadmap step 3). See hmm.py.

hmm.py decodes each graded half in one pass over the states H / A / D, combining the dead-ball
probability (step 1), the control events (step 2) and optionally the team-shape model (step 4).

    from mtah.temporal.hmm import decode
    labels = decode(match, events, p_dead, dead_labels)    # "H" / "A" / "D" per frame
"""
