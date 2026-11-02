"""The numpy cache: the match parsed once, loaded in about a second.

    python scripts/build_cache.py            # once, ~1 min
    from mtah.cache import load
    match = load()                           # memory-mapped arrays, see Match below

One folder per source under cache/ (or $MTAH_CACHE_DIR), one .npy per array plus meta.json.
Everything is indexed by video frame. Coordinates are in METRES, origin at a pitch corner, x
along the length (0..105.3) and y across the width (0..68). The homography still takes normalised
coordinates: divide by match.pitch before projecting.

The cache is derived from the match data, so it is gitignored and never leaves the cluster.
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DEFAULT_SOURCE, asset_paths, half_video_blocks  # noqa: E402
from mtah import io  # noqa: E402

# Params
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "cache")
# Bump when the cache layout or its contents change, so stale caches are rebuilt, not misread.
CACHE_VERSION = 1
ARRAYS = ("players_xy", "players_speed", "player_present", "ball_xy", "ball_present",
          "half", "graded", "homography", "reprojection_error", "homography_valid",
          "team_of_slot")


def cache_dir(game=GAME, source=DEFAULT_SOURCE):
    return os.path.join(CACHE_DIR, "{}_{}".format(game, source))


def _source_stats(game, data_dir, source):
    """Size and mtime of every input file, to detect a cache built from different data."""
    stats = {}
    for name, path in asset_paths(game, data_dir, source).items():
        if name != "video" and os.path.isfile(path):
            info = os.stat(path)
            stats[name] = {"size": info.st_size, "mtime": int(info.st_mtime)}
    return stats


def build(game=GAME, data_dir=None, source=DEFAULT_SOURCE, verbose=True):
    """Parse the source files and write the cache. Returns the cache folder."""
    say = print if verbose else (lambda *args, **kwargs: None)
    start = time.time()
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)

    meta = io.read_atd_meta(game, data_dir, source)
    say("parsing ATD ...")
    players, ball = io.read_atd_raw(game, data_dir, source)
    n_frames = len(players)
    say("  {} frames in {:.0f} s".format(n_frames, time.time() - start))

    say("reading homography ...")
    matrices, reprojection_error, valid = io.read_homography(game, data_dir, source)

    # Half and graded masks exist only for the tactical timeline, which is the one the half
    # boundaries in games.py are measured on.
    half = np.zeros(n_frames, dtype=np.int8)
    if source == "tactical":
        for number, first, last in half_video_blocks(game):
            half[first:last + 1] = number
    graded = half > 0

    arrays = {
        "players_xy": players[:, :, :2] * scale,
        "players_speed": players[:, :, 2],
        "player_present": ~np.isnan(players[:, :, 0]),
        "ball_xy": ball * scale,
        "ball_present": ~np.isnan(ball[:, 0]),
        "half": half,
        "graded": graded,
        "homography": matrices,
        "reprojection_error": reprojection_error,
        "homography_valid": valid,
        "team_of_slot": meta["team_of_slot"],
    }

    # Playing direction per half, from where each team's goalkeeper stands. Teams swap ends at
    # halftime, so this is per half: True means home attacks towards x = length.
    home_attacks_right = {}
    home_keeper = meta["goalkeeper_slots"][0]
    for number in (1, 2) if source == "tactical" else ():
        keeper_x = arrays["players_xy"][half == number, home_keeper, 0]
        home_attacks_right[number] = bool(np.nanmedian(keeper_x) < length / 2)

    folder = cache_dir(game, source)
    os.makedirs(folder, exist_ok=True)
    for name in ARRAYS:
        np.save(os.path.join(folder, name + ".npy"), np.ascontiguousarray(arrays[name]))

    if source == "tactical":
        tags = io.read_smart_tagging(meta["team_names"], game, data_dir, GAMES[game]["fps"])
        tags["tags"] = tags["tags"].apply(json.dumps)
        tags.to_csv(os.path.join(folder, "smart_tagging.csv"), index=False)

    info = {
        "version": CACHE_VERSION,
        "game": game,
        "source": source,
        "n_frames": n_frames,
        "fps": GAMES[game]["fps"],
        "pitch": [length, width],
        "team_names": list(meta["team_names"]),
        "goalkeeper_slots": list(meta["goalkeeper_slots"]),
        "home_attacks_right": home_attacks_right,
        "half_blocks": half_video_blocks(game) if source == "tactical" else [],
        "sources": _source_stats(game, data_dir, source),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(folder, "meta.json"), "w") as handle:
        json.dump(info, handle, indent=2)
    say("cache written to {} in {:.0f} s".format(folder, time.time() - start))
    return folder


class Match:
    """One source of one match, loaded from the cache. Arrays are read-only memory maps.

    players_xy          (n, 46, 2) float32  metres, NaN where the slot is empty
    players_speed       (n, 46)    float32  m/s, NaN where unknown (often for goalkeepers)
    player_present      (n, 46)    bool
    ball_xy             (n, 2)     float32  metres, NaN where the ball is not tracked
    ball_present        (n,)       bool
    half                (n,)       int8     1, 2, or 0 outside the graded halves
    graded              (n,)       bool     the 158857 frames the evaluation scores
    homography          (n, 3, 3)  float64  normalised pitch -> pixels
    reprojection_error  (n,)       float32
    homography_valid    (n,)       bool
    team_of_slot        (46,)      int8     0 home, 1 away
    smart_tagging       DataFrame or None   see mtah.io.read_smart_tagging
    meta                dict               meta.json, plus shortcuts below
    """

    def __init__(self, folder, mmap=True):
        with open(os.path.join(folder, "meta.json")) as handle:
            self.meta = json.load(handle)
        for name in ARRAYS:
            setattr(self, name, np.load(os.path.join(folder, name + ".npy"),
                                        mmap_mode="r" if mmap else None))
        tags_path = os.path.join(folder, "smart_tagging.csv")
        self.smart_tagging = None
        if os.path.isfile(tags_path):
            import pandas as pd
            self.smart_tagging = pd.read_csv(tags_path)
            self.smart_tagging["tags"] = self.smart_tagging["tags"].apply(json.loads)

        self.n_frames = self.meta["n_frames"]
        self.fps = self.meta["fps"]
        self.pitch = tuple(self.meta["pitch"])
        self.team_names = tuple(self.meta["team_names"])
        self.goalkeeper_slots = tuple(self.meta["goalkeeper_slots"])
        self.home_attacks_right = {int(k): v for k, v in self.meta["home_attacks_right"].items()}


def load(game=GAME, source=DEFAULT_SOURCE, data_dir=None, mmap=True):
    """The cached match. Raises if the cache is missing, outdated, or built from other files."""
    folder = cache_dir(game, source)
    meta_path = os.path.join(folder, "meta.json")
    if not os.path.isfile(meta_path):
        raise IOError("no cache at {}\nBuild it first:  python scripts/build_cache.py".format(
            folder))
    with open(meta_path) as handle:
        meta = json.load(handle)
    if meta.get("version") != CACHE_VERSION:
        raise IOError("cache at {} is version {}, code expects {}. Rebuild it:  python "
                      "scripts/build_cache.py".format(folder, meta.get("version"), CACHE_VERSION))
    current = _source_stats(game, data_dir, source)
    if current and current != meta.get("sources"):
        raise IOError("cache at {} was built from different input files. Rebuild it:  python "
                      "scripts/build_cache.py".format(folder))
    return Match(folder, mmap=mmap)
