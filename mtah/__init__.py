"""Our possession pipeline for More than a Hack 2027.

The package turns the provided tracking data of one match into a per-frame possession label
("H", "A", "D", "X") for every frame of the tactical video -- the submission. It is organised by
topic:

    mtah.data          parse the provided files and build / load the numpy cache
    mtah.features      per-frame features from the cache
    mtah.dead_ball     dead-ball detection (roadmap step 1)
    mtah.possession    who has the ball (steps 2-4); the official baseline lives here
    mtah.temporal      decoding the whole match at once (step 3)
    mtah.validation    comparing methods without labels (masking test)
    mtah.methods       every method by name, for the scripts
    mtah.utils         small shared helpers on per-frame arrays

Importing mtah puts the repository root, starter/ and evaluation/ on sys.path, so games.py,
the official baseline and the official scorer import the same way everywhere. ROOT is the
repository root, used to place cache/ and other top-level folders.

    import mtah                            # sets up the path; mtah.ROOT is the repository root
    from mtah.data import load
    from mtah.methods import METHODS
    labels = METHODS["hmm_v2"](load())     # (n_frames,) array of "H" / "A" / "D"
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.append(_path)
