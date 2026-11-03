"""Per-frame features computed from the cache. See frame.py.

Re-exports compute (the per-frame feature table of a Match) and the column groups PLAYER, BALL
and TAGGING from frame.py. The ball-free possession features are in team_shape.py and are
imported from there.

    from mtah import features
    table = features.compute(match)          # DataFrame, one row per video frame
    table[features.PLAYER]                   # the player-movement columns only

    from mtah.features.team_shape import compute as team_shape
"""

from mtah.features.frame import BALL, PLAYER, TAGGING, compute  # noqa: F401
