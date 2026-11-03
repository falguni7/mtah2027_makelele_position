"""Reading the match: parsing the provided files (io.py) and the numpy cache (cache.py).

io.py parses the raw provided files (slow, about a minute); cache.py writes the result once as
numpy arrays and loads it back in about a second. Everything else in the pipeline only ever sees
the cached Match, so the three names re-exported here are all most code needs:

    Match      one source of one match, loaded from the cache (arrays indexed by video frame)
    build      parse the source files and write the cache (scripts/data/build_cache.py)
    load       the cached Match, with checks that the cache is current

    from mtah.data import load
    match = load()                         # tactical source of BAR-ATH by default
    match.n_frames, match.fps              # -> e.g. (199319, 25)
"""

from mtah.data.cache import Match, build, load  # noqa: F401
