"""Parse the match files once into the numpy cache (see mtah/cache.py).

    python scripts/build_cache.py                      # tactical, the submission timeline
    python scripts/build_cache.py --source broadcast   # optional, its own timeline
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah import cache  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=("tactical", "broadcast"))
    args = parser.parse_args()
    cache.build(source=args.source)
