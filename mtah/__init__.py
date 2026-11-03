"""Our possession pipeline for More than a Hack 2027.

    mtah.data          parse the provided files and build / load the numpy cache
    mtah.features      per-frame features from the cache
    mtah.dead_ball     dead-ball detection (roadmap step 1)
    mtah.possession    who has the ball (steps 2-4); the official baseline lives here
    mtah.temporal      decoding the whole match at once (step 3)
    mtah.validation    comparing methods without labels (masking test)
    mtah.methods       every method by name, for the scripts
    mtah.utils         small shared helpers on per-frame arrays

Importing mtah puts the repository root, starter/ and evaluation/ on sys.path, so games.py,
the official baseline and the official scorer import the same way everywhere.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.append(_path)
