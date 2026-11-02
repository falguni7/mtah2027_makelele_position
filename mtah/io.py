"""Parse the provided match files into numpy arrays, indexed by video frame.

Everything returned here is on the video timeline of its source: index i is video frame i, which
for the tactical source is also submission row i. The one conversion this needs is the ATD's
frame_id, which counts from 1 (video_frame = frame_id - 1, see starter/load_atd.py).

Facts about the files that the code below relies on (measured, see the dataset notes):

  * ATD line = "<frame_id>:<46 slot groups, ';'-separated>:<ball>", every group "x,y,speed",
    missing values written as NaN. x, y are normalised 0-1; speed is m/s (the XML says kph).
  * The ball's speed field is a tracker artifact and is dropped. Derive speed from positions.
  * Slot teams come from the ATD metadata; the first team listed is home.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, DEFAULT_SOURCE, require, source_frames  # noqa: E402

# Params
N_SLOTS = 46
HOME, AWAY = 0, 1
HOMOGRAPHY_COLUMNS = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]


def read_atd_meta(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """{team_names: (home, away), team_of_slot: (46,) int8, goalkeeper_slots: (home, away)}.

    Slots are 0-based here (the XML's "Track 1" is slot 0).
    """
    meta_path = require(["atd_meta"], game, data_dir, source)[0]
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]

    players = list(root.iter("Player"))
    if len(players) != N_SLOTS:
        raise ValueError("{}: expected {} slots, found {}".format(
            meta_path, N_SLOTS, len(players)))
    team_of_slot = np.array([HOME if player.get("teamId") == home_id else AWAY
                             for player in players], dtype=np.int8)

    goalkeepers = {}
    for slot, player in enumerate(players):
        params = {param.findtext("Name"): param.findtext("Value")
                  for param in player.iter("ProviderParameter")}
        if params.get("position_type") == "Goalkeeper":
            goalkeepers[int(team_of_slot[slot])] = slot
    if sorted(goalkeepers) != [HOME, AWAY]:
        raise ValueError("{}: expected one goalkeeper slot per team, found {}".format(
            meta_path, goalkeepers))

    return {
        "team_names": (teams[0][1], teams[1][1]),
        "team_of_slot": team_of_slot,
        "goalkeeper_slots": (goalkeepers[HOME], goalkeepers[AWAY]),
    }


def read_atd_raw(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(players, ball) for every video frame, normalised coordinates, NaN where missing.

    players is (n, 46, 3) float32 with x, y, speed. ball is (n, 2) float32 with x, y.

    Every separator becomes a comma so pandas' C parser reads the file as one numeric table:
    1 frame id + 46 * 3 player values + 3 ball values = 142 columns. Takes ~20 s, versus minutes
    for a line-by-line Python loop.
    """
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    n_frames = source_frames(source, game)
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32,
                        engine="c").to_numpy()
    del text

    n_columns = 1 + N_SLOTS * 3 + 3
    if table.shape[1] != n_columns:
        raise ValueError("{}: expected {} values per line, found {}".format(
            raw_path, n_columns, table.shape[1]))
    frame_ids = table[:, 0].astype(np.int64)
    if len(frame_ids) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("{}: frame ids are not 1..{} without gaps".format(raw_path, n_frames))

    players = np.ascontiguousarray(table[:, 1:1 + N_SLOTS * 3].reshape(n_frames, N_SLOTS, 3))
    ball = np.ascontiguousarray(table[:, 1 + N_SLOTS * 3:1 + N_SLOTS * 3 + 2])
    return players, ball


def read_homography(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(matrices (n, 3, 3) float64, reprojection_error (n,) float32, valid (n,) bool).

    The matrices map normalised pitch coordinates (0-1) to image pixels.
    """
    path = require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)
    n_frames = source_frames(source, game)
    if len(table) != n_frames or not np.array_equal(table["frame"].values, np.arange(n_frames)):
        raise ValueError("{}: rows are not frames 0..{}".format(path, n_frames - 1))
    matrices = table[HOMOGRAPHY_COLUMNS].to_numpy(np.float64).reshape(-1, 3, 3)
    return (matrices, table["reprojection_error"].to_numpy(np.float32),
            table["valid"].to_numpy().astype(bool))


def read_smart_tagging(team_names, game=GAME, data_dir=None, fps=25):
    """Smart Tagging instances as a DataFrame, one row per instance, frames on the video timeline.

    Columns: code, start_s, end_s, start_frame, end_frame (half-open), team ("H", "A" or None),
    half, type, and tags (every label group, as a dict).

    The Team tag names a club, mapped to H / A through the ATD metadata's team names (the two
    files spell them identically). "N/A" (throw-ins, kick-offs) becomes None. An unknown club is
    an error rather than a silent guess.
    """
    path = require(["smart_tagging"], game, data_dir)[0]
    side_of_club = {team_names[0]: "H", team_names[1]: "A", "N/A": None, None: None}
    rows = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        club = tags.get("Team")
        if club not in side_of_club:
            raise ValueError("{}: unknown Team tag {!r}".format(path, club))
        team = side_of_club[club]
        rows.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "team": team,
            "half": tags.get("Half"),
            "type": tags.get("Type"),
            "tags": tags,
        })
    return pd.DataFrame(rows)
