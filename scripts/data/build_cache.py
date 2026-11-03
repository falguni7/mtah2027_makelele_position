"""Parse the match files once into the numpy cache (see mtah/data/cache.py).

Every other script reads the match through mtah.data.load(), which needs this cache, so this is
the first thing to run after the match files are in data/. Run it again whenever the data or the
cache layout changes.

    python scripts/data/build_cache.py                      # tactical, the submission timeline
    python scripts/data/build_cache.py --source broadcast   # optional, its own timeline
    sbatch scripts/slurm/run.sbatch scripts/data/build_cache.py

Writes one folder per source, cache/<game>_<source>/ (or under $MTAH_CACHE_DIR): one .npy per
array plus meta.json, plus smart_tagging.csv for the tactical source; about 131 MB for tactical
(gitignored, never leaves the cluster). Takes half a minute to a minute, ~1.2 GB peak memory.
Prints progress only: "parsing ATD ...", the number of frames and the seconds taken, "reading
homography ...", and finally "cache written to <folder> in <n> s".
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mtah.data import cache  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    args = parser.parse_args()
    cache.build(source=args.source)
