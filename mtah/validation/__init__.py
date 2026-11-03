"""Ways to compare methods without the ground truth. See masking.py.

    masking.py     the masking test: hide the ball where it is tracked and measure how much a
                   method's answer changes (robustness to ball gaps, not accuracy)
    tagging.py     agreement with Smart Tagging's in-possession phases (a consistency check)

    from mtah.validation import masking, tagging
    blocks = masking.sample_blocks(match, masking.real_gap_lengths(match), rng)
    tagging.agreement(labels, tagging.phase_team(match))     # -> {frames, agree, disagree, dead}
"""
