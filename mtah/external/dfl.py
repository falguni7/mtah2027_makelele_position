"""DFL Bundesliga open data: 7 matches with per-frame possession and ball status labels.

    from mtah.external import dfl
    dfl.MATCHES                           # the 7 match ids
    m = dfl.load("DFL-MAT-J03WMX")        # parsed on first use, then read from the cache
    m.labels[:3]                          # -> e.g. array(['A', 'A', 'A'])   H / A / D per frame

The dataset (Bassek et al., Scientific Data 12, 195, 2025; CC BY 4.0, figshare 28196177) has,
per match, three XML files in external/dfl/:

  matchinformation   teams (which is home, which guest), players, result, stadium
  events_raw         the official event data: passes, tackles, throw-ins, fouls, ... with times
  positions_raw_observed   TRACAB tracking at 25 Hz: every player, the referees and the ball,
                     each as a <FrameSet> per person and half, one <Frame> per frame with
                     N (frame number), X, Y (metres from the centre spot), S (speed, km/h), and
                     for the ball also Z (height), BallPossession (1 home, 2 guest) and
                     BallStatus (1 alive, 0 dead)

The two ball fields are what our ground truth is made of (CHALLENGE.md: "a frame is D where the
corrected tracking marks the ball dead, and otherwise takes the team the tracking names as owning
it"), so labels() turns them into H / A / D exactly that way.

load() returns a DFLMatch with the frames of both halves one after the other (the frame numbers
N, which jump at half-time, are kept in `frame_n`) and positions in OUR convention: metres from
the pitch corner, x along the length 0..105, y across 0..68. Players are slots: home players
first, then guest players, in the order they appear in the file; a slot is NaN on frames where
that player is not on the pitch (before coming on, after going off). Referees are left out.
Speeds are converted from DFL's km/h to m/s, our unit (players_speed in our cache is m/s).

Parsing a 400 MB XML file takes about a minute; the result is kept in
cache/external/dfl/<match>.npz and read from there afterwards (parsed again when the cache was
written by an older CACHE_VERSION of this code).
"""

import json
import os
import re
from dataclasses import dataclass

import numpy as np

from mtah.data.cache import CACHE_DIR

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DFL_DIR = os.path.join(ROOT, "external", "dfl")
CACHE = os.path.join(CACHE_DIR, "external", "dfl")
FPS = 25
PITCH = (105.0, 68.0)
KMH = 3.6                   # DFL speeds are km/h; divided by this they are m/s
CACHE_VERSION = 2           # 2: speeds in m/s (1 kept DFL's km/h)
MATCHES = ("DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
           "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9")

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" '
                      r'PersonId="([^"]*)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"(?: Z="([-\d.]+)")?'
                   r'[^>]*? S="([-\d.]+)"(?:[^>]*? BallPossession="(\d)" BallStatus="(\d)")?')


@dataclass
class DFLMatch:
    """One DFL match, all frames of both halves in a row.

    match_id     e.g. "DFL-MAT-J03WMX"
    home, guest  team ids (DFL-CLU-...)
    frame_n      (n,) DFL frame number of each row
    half         (n,) 1 or 2
    ball_xyz     (n, 3) metres, our convention for x, y; z height
    ball_speed   (n,) m/s
    possession   (n,) 1 home, 2 guest (as in the file)
    alive        (n,) bool, BallStatus == 1
    players_xy   (n, slots, 2) metres, NaN where the player is not on the pitch
    players_speed (n, slots) m/s
    slot_team    (slots,) 0 home, 1 guest
    """
    match_id: str
    home: str
    guest: str
    frame_n: np.ndarray
    half: np.ndarray
    ball_xyz: np.ndarray
    ball_speed: np.ndarray
    possession: np.ndarray
    alive: np.ndarray
    players_xy: np.ndarray
    players_speed: np.ndarray
    slot_team: np.ndarray

    @property
    def n_frames(self):
        """Number of frames (rows), both halves together."""
        return len(self.frame_n)

    @property
    def labels(self):
        """H / A / D per frame, the way our ground truth is built: D where the ball is dead,
        else H if the home team owns it and A if the guest team does.

            possession = [1, 1, 2, 2], alive = [T, F, T, T]
            labels                     # -> ["H", "D", "A", "A"]
        """
        out = np.where(self.possession == 1, "H", "A").astype("<U1")
        out[~self.alive] = "D"
        return out


def paths(match_id):
    """(matchinformation, events, positions) file paths of one match in external/dfl/.

        paths("DFL-MAT-J03WMX")[2]
        # -> ".../external/dfl/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml"
    """
    names = os.listdir(DFL_DIR)
    pick = [next(os.path.join(DFL_DIR, n) for n in names if n.startswith(prefix) and match_id in n)
            for prefix in ("DFL_02_01_matchinformation", "DFL_03_02_events_raw",
                           "DFL_04_03_positions_raw_observed")]
    return tuple(pick)


def available():
    """True when all three files of all 7 matches are in external/dfl/, else False.

    What a method using DFL needs before it can run: the match information, the events and
    the positions of every match in MATCHES (paths() finds each by its prefix and match id).
    It only checks that the files are there; scripts/data/download_external.sh checks their
    sizes when it downloads them.

        available()        # -> True after `bash scripts/data/download_external.sh --dfl-only`
    """
    for match_id in MATCHES:
        try:
            files = paths(match_id)
        except (OSError, StopIteration):        # no folder, or a file missing
            return False
        if not all(os.path.isfile(p) for p in files):
            return False
    return True


def teams(match_id):
    """(home team id, guest team id) from the match information file.

        teams("DFL-MAT-J03WMX")      # -> ("DFL-CLU-000008", "DFL-CLU-00000G")   Köln, Bayern
    """
    text = open(paths(match_id)[0], encoding="utf-8").read()
    return (re.search(r'HomeTeamId="([^"]+)"', text).group(1),
            re.search(r'GuestTeamId="([^"]+)"', text).group(1))


def parse(match_id):
    """Read one match's positions file into a DFLMatch (slow: ~1 min; load() caches it).

    Every <FrameSet> (one person, one half) is read line by line. The ball's frames fix the
    rows: every frame number at which the ball has a position, per half, in order. Players are
    placed on those rows by frame number. DFL's x, y are metres from the centre spot; they are
    shifted by half the pitch so that x runs 0..105 and y 0..68 from a corner, as in our cache,
    and speeds divided by KMH (km/h -> m/s).

        m = parse("DFL-MAT-J03WMX")
        m.players_xy.shape               # -> (n_frames, number of players used, 2)
    """
    home, guest = teams(match_id)
    persons = {}            # (half, team, person) -> list of (N, x, y, z, s, poss, status)
    current = None
    with open(paths(match_id)[2], encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                m = FRAMESET.search(line)
                current = None if m.group(2) == "referee" else (m.group(1), m.group(2), m.group(3))
                if current is not None:
                    persons[current] = []
            elif current is not None and "<Frame " in line:
                m = FRAME.search(line)
                persons[current].append(m.groups())
    halves = [("firstHalf", 1), ("secondHalf", 2)]
    rows = {}
    ball = []
    for section, number in halves:
        key = next(k for k in persons if k[0] == section and k[1] == "BALL")
        data = persons[key]
        n = np.array([int(r[0]) for r in data])
        rows[section] = {v: i for i, v in enumerate(n)}
        ball.append((number, n, np.array([[float(r[1]), float(r[2]), float(r[3] or 0), float(r[4]),
                                           int(r[5]), int(r[6])] for r in data])))
    offsets = np.cumsum([0] + [len(b[1]) for b in ball])
    n_total = offsets[-1]
    slot_ids = []
    for team in (home, guest):
        for (_, t, person) in persons:
            if t == team and person not in slot_ids:
                slot_ids.append(person)
    slot_of = {p: i for i, p in enumerate(slot_ids)}
    xy = np.full((n_total, len(slot_ids), 2), np.nan, dtype=np.float32)
    speed = np.full((n_total, len(slot_ids)), np.nan, dtype=np.float32)
    for (section, team, person), data in persons.items():
        if team == "BALL":
            continue
        k = [s for s, _ in halves].index(section)
        index = rows[section]
        frames = np.array([int(r[0]) for r in data])
        keep = np.array([f in index for f in frames])
        at = np.array([index[f] for f in frames[keep]], dtype=int) + offsets[k]
        values = np.array([[float(r[1]), float(r[2]), float(r[4]) / KMH] for r in data])[keep]
        xy[at, slot_of[person], 0] = values[:, 0] + PITCH[0] / 2
        xy[at, slot_of[person], 1] = values[:, 1] + PITCH[1] / 2
        speed[at, slot_of[person]] = values[:, 2]
    b = np.concatenate([v for _, _, v in ball])
    return DFLMatch(
        match_id=match_id, home=home, guest=guest,
        frame_n=np.concatenate([n for _, n, _ in ball]),
        half=np.concatenate([np.full(len(n), h) for h, n, _ in ball]).astype(np.int8),
        ball_xyz=np.column_stack([b[:, 0] + PITCH[0] / 2, b[:, 1] + PITCH[1] / 2,
                                  b[:, 2]]).astype(np.float32),
        ball_speed=(b[:, 3] / KMH).astype(np.float32),
        possession=b[:, 4].astype(np.int8), alive=b[:, 5] == 1,
        players_xy=xy, players_speed=speed,
        slot_team=np.array([0 if any(k[1] == home and k[2] == p for k in persons) else 1
                            for p in slot_ids], dtype=np.int8))


def slot_persons(match_id, home, guest):
    """Person id of every player slot, in the slot order parse() uses (home players first).

    parse() gives each player a slot: first every home player, then every guest player, each in
    the order their first <FrameSet> appears in the positions file. The DFLMatch does not keep
    the person ids, so this re-reads only the <FrameSet> header lines (a fast pass over the
    file; the result is kept in cache/external/dfl/<match>_slots.json) and rebuilds the same
    order. Used to find the goalkeepers' slots.

        slot_persons("DFL-MAT-J03WMX", m.home, m.guest)[:2]
        # -> ["DFL-OBJ-...", "DFL-OBJ-..."]   slots 0 and 1, both home players
    """
    path = os.path.join(CACHE, match_id + "_slots.json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    order = []
    with open(paths(match_id)[2], encoding="utf-8") as handle:
        for line in handle:
            if "<FrameSet" in line:
                m = FRAMESET.search(line)
                if (m.group(2), m.group(3)) not in order:
                    order.append((m.group(2), m.group(3)))
    out = [person for team in (home, guest) for t, person in order if t == team]
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out


def goalkeeper_slots(match):
    """(home goalkeeper slot, guest goalkeeper slot) of a DFLMatch: the starting goalkeepers.

    The match information lists each starting player's PlayingPosition; "TW" (Torwart) is the
    goalkeeper. Substitutes have no position there, so a goalkeeper who comes on later is
    treated as an outfield player (none of the 7 matches changes goalkeeper). The slots are
    those of slot_persons(). Our feature code leaves goalkeepers out of a team's shape, as for
    BAR-ATH (match.goalkeeper_slots there).

        goalkeeper_slots(dfl.load("DFL-MAT-J03WMX"))      # -> (3, 16)
    """
    text = open(paths(match.match_id)[0], encoding="utf-8").read()
    keepers = set(re.findall(r'<Player PersonId="([^"]+)"[^>]*PlayingPosition="TW"', text))
    persons = slot_persons(match.match_id, match.home, match.guest)
    slots = [s for s, p in enumerate(persons) if p in keepers]
    team = np.asarray(match.slot_team)
    return (next(s for s in slots if team[s] == 0), next(s for s in slots if team[s] == 1))


def load(match_id):
    """The DFLMatch of one match: from cache/external/dfl/<match>.npz, parsed first if needed.

    The cache is parsed again when it is missing or was written with another CACHE_VERSION
    (e.g. version 1, which kept speeds in km/h).

        m = load("DFL-MAT-J03WMX")       # ~1 min the first time, ~1 s afterwards
        m.labels                         # H / A / D per frame
    """
    path = os.path.join(CACHE, match_id + ".npz")
    fields = [f for f in DFLMatch.__dataclass_fields__]
    if os.path.exists(path):
        with np.load(path) as data:
            current = "version" in data and int(data["version"]) == CACHE_VERSION
    if not os.path.exists(path) or not current:
        m = parse(match_id)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, version=CACHE_VERSION, **{f: getattr(m, f) for f in fields})
    data = np.load(path)
    return DFLMatch(**{f: (str(data[f]) if data[f].ndim == 0 else data[f]) for f in fields})
