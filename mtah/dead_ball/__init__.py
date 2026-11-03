"""Dead-ball detection (roadmap step 1). See detector.py.

Finds the stoppages (label D) without human labels. Re-exports detect, which takes a Match and
its feature table and returns (dead mask, P(dead), confident labels) per video frame.

    from mtah import dead_ball, features
    dead, p_dead, labels = dead_ball.detect(match, features.compute(match))
"""

from mtah.dead_ball.detector import detect  # noqa: F401
