"""Metrica Sports' 3 open sample matches, with H / A / D possession labels made from their events.

    from mtah.external import metrica
    m = metrica.load(1)                 # Sample Game 1: tracking + events, parsed and cached
    labels = metrica.labels(m)          # H / A / D per frame, X outside the two halves
    labels[m.half == 1][:3]             # -> e.g. array(['D', 'A', 'A'])

The data (external/metrica/, github.com/metrica-sports/sample-data; no licence stated, "please
be responsible", acknowledge the source if used publicly) is 3 anonymised matches at 25 fps,
tracked by Metrica Sports, the company whose automatic tracking and Smart Tagging we get for
BAR-ATH. Its ball is missing on ~40 % of frames, about as often as ours, and for the same reason
(one camera loses it), so its gaps are real, not imitated as in the DFL benchmark.

  game 1, 2   CSV: one tracking file per team (Period, Frame, Time, x / y per player, ball x /
              y) and one event file
  game 3      EPTS (FIFA format, the same as our ATD): a metadata XML and a text file with one
              line per frame, "frame:x,y;x,y;...:ball x,ball y", whose player order changes at
              every substitution (DataFormatSpecification per frame range); events in JSON,
              with CARRY events that games 1-2 do not have

Coordinates are 0-1 along the length and the width, (0, 0) top left, as our ATD; the pitch is
105 x 68 m. Metrica's frame numbers start at 1; row i of a match is frame first + i.

There are no per-frame possession labels, only human event data (documentation/
events-definitions.pdf): ~1,700-3,600 events per match, each with the team, a type, subtypes and
start / end frames synced to the tracking. labels() derives possession from them in three steps:

  1. dead ball (dead()). A stoppage starts where the event data says play stopped and ends at the
     next restart. Starts (stoppage_start()): a BALL OUT or a SHOT that goes OUT or in as a GOAL,
     at its end frame (the ball crosses the line); a FAULT RECEIVED, or a CHALLENGE with a FAULT
     and no ADVANTAGE, at its start frame (the whistle); a BALL LOST that is FORCED (the referee
     stops play), OFFSIDE or HAND BALL, at its end frame. The restart is the first later SET
     PIECE, or RECOVERY with no subtype ("brings the ball back into play via a dead-ball
     situation that is not a SET PIECE": goal kicks and drop balls in games 1-2). If a live event
     (a pass, carry, shot, or any other event of the run of play) starts in between, play went
     on (advantage, or a whistle logged early), and the stoppage is dropped. Before the kick-off
     and after the last event of a half the frames are not graded (X), as half-time in our data.
  2. raw owner (owner()). The team of the latest on-ball event: PASS, CARRY, SHOT, SET PIECE,
     RECOVERY, BALL LOST, BALL OUT and FAULT RECEIVED all say that team had the ball at their
     start frame. A pass in flight stays with the passer's team, and a lost ball stays with the
     loser until the other team's RECOVERY: "a loose ball stays with the last team" (CHALLENGE.md).
  3. controlled possession (mtah/external/controlled.py, the rule our DFL labels use). A short
     spell of the other team is a change only if it lasts MIN_SECONDS (3.75 s, chosen on DFL) or
     holds a controlled action of that team (controlled_events(): a PASS, which in Metrica is
     always completed, failed passes being BALL LOST; a SHOT, SET PIECE, FAULT RECEIVED, or a
     RECOVERY THEFT, stealing the ball from an opponent in possession). Otherwise it is a touch,
     a deflection or an interception that went straight back, and the frames go to the team that
     had the ball. A RECOVERY INTERCEPTION alone does not count: Metrica also logs blocks and
     one-touch headers as interceptions, and a real one is followed by a pass anyway. A CARRY
     does not count either: game 3 logs one for almost any movement with the ball, even right
     after a one-touch interception, so with carries game 3's OR labels kept 382 possessions
     (a third under 3 s) against ~237 in games 1-2, which have no CARRY events. Without them all
     three games follow the same rule. A CARRY still says who has the ball (step 2).

So the labels follow CHALLENGE.md's definition with Metrica's own human notion of who has the
ball, at the frame accuracy of the event loggers. They are an approximation of what our ground
truth would say (it comes from corrected tracking), the same kind of approximation as the DFL
labels (tracking owner + the same rule).

parse() reads the files once (~1 min for game 3) and load() keeps the result in
cache/external/metrica/game<k>.npz.
"""

import csv
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mtah.data.cache import CACHE_DIR
from mtah.external import controlled

# Params
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "external", "metrica", "data")
CACHE = os.path.join(CACHE_DIR, "external", "metrica")
FPS = 25
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
CACHE_VERSION = 2           # 2: event text saved as plain strings (1 saved objects)
ON_BALL = ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT",
           "FAULT RECEIVED")
LIVE = ("PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT")   # play is on
CONTROLLED = ("PASS", "SHOT", "SET PIECE", "FAULT RECEIVED")    # and RECOVERY THEFT
MIN_SECONDS = controlled.MIN_SECONDS        # 3.75 s, chosen on DFL (controlled.py)
TOLERANCE = controlled.TOLERANCE            # 12 frames


@dataclass
class MetricaMatch:
    """One Metrica sample match, all frames of both halves in a row.

    game         1, 2 or 3
    frame        (n,) Metrica frame number of each row (row i is frame frame[0] + i)
    half         (n,) 1 or 2 (0 for frames in neither, which the files do not have)
    ball_xy      (n, 2) 0-1 pitch coordinates, NaN where the ball is not tracked
    players_xy   (n, slots, 2) 0-1, NaN where the player is not tracked or not on the pitch
    slot_team    (slots,) 0 home, 1 away
    events       DataFrame, one row per event: team (0 home, 1 away), type, subtype (subtypes
                 joined with "-", "" if none), period, start, end (rows, not frame numbers)
    """
    game: int
    frame: np.ndarray
    half: np.ndarray
    ball_xy: np.ndarray
    players_xy: np.ndarray
    slot_team: np.ndarray
    events: pd.DataFrame

    @property
    def n_frames(self):
        """Number of frames (rows), both halves together."""
        return len(self.frame)


def _csv_team(path):
    """(period, frame, players (n, p, 2), ball (n, 2)) of one team's Metrica CSV tracking file.

    The file has two label rows (team, shirt numbers), then a header row "Period, Frame,
    Time [s], Player11, , Player1, , ..., Ball, " where each player and the ball take an x and a
    y column (the y column has no name).

        period, frame, xy, ball = _csv_team(".../Sample_Game_1_RawTrackingData_Home_Team.csv")
        xy.shape                       # -> (145006, 14, 2)
    """
    table = pd.read_csv(path, skiprows=2)
    cols = list(table.columns)
    players = [i for i, c in enumerate(cols) if c.startswith("Player")]
    ball = cols.index("Ball")
    values = table.to_numpy(dtype=np.float64)
    xy = np.stack([values[:, [i, i + 1]] for i in players], axis=1)
    return (values[:, 0].astype(np.int8), values[:, 1].astype(np.int64), xy,
            values[:, [ball, ball + 1]])


def _csv_events(path, first):
    """The events of a game-1/2 CSV file as a table (team 0 / 1, type, subtype, period, start,
    end), with rows = frame - first.

        Away,PASS,,1,1,0.04,3,0.12,Player19,Player21,...   (first frame 1)
        -> team 1, type "PASS", subtype "", period 1, start 0, end 2
    """
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"team": 0 if r["Team"] == "Home" else 1, "type": r["Type"],
                         "subtype": r["Subtype"] or "", "period": int(r["Period"]),
                         "start": int(r["Start Frame"]) - first,
                         "end": int(r["End Frame"]) - first})
    return pd.DataFrame(rows)


def _epts(game):
    """(frame, half, players (n, slots, 2), slot_team, ball (n, 2), home team name) of game 3.

    The metadata lists every player (id, team), the half bounds (first_half_start ...) and, per
    frame range, the order of the players on each tracking line (DataFormatSpecification ->
    PlayerChannelRef -> PlayerChannel -> player id). Every player gets one slot for the whole
    match, NaN outside his frame ranges. The home team is the Score's idLocalTeam.

        frame, half, xy, team, ball, home = _epts(3)
        xy.shape, home                 # -> ((143761, 35, 2), "Team A")
    """
    base = os.path.join(DATA_DIR, "Sample_Game_{0}".format(game), "Sample_Game_{0}_".format(game))
    root = ET.parse(base + "metadata.xml").getroot()
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    local = root.find(".//Score").get("idLocalTeam")
    team_names = {t.get("id"): t.findtext("Name") for t in root.iter("Team")}
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    slot = {pid: i for i, (pid, _) in enumerate(players)}
    slot_team = np.array([0 if team == local else 1 for _, team in players], dtype=np.int8)
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        order = [slot[channel[r]] for r in refs if r.endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    with open(base + "tracking.txt") as f:
        lines = f.read().split()
    n = len(lines)
    frame = np.empty(n, dtype=np.int64)
    xy = np.full((n, len(players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        number, people, b = line.split(":")
        number = int(number)
        frame[i] = number
        order = next(o for a, z, o in specs if a <= number <= z)
        vals = np.array([v.split(",") for v in people.rstrip(";").split(";")], dtype=np.float64)
        xy[i, order] = vals
        ball[i] = np.array(b.split(","), dtype=np.float64)
    half = np.zeros(n, dtype=np.int8)
    half[(frame >= int(params["first_half_start"])) & (frame <= int(params["first_half_end"]))] = 1
    half[(frame >= int(params["second_half_start"]))
         & (frame <= int(params["second_half_end"]))] = 2
    return frame, half, xy, slot_team, ball, team_names[local]


def _json_events(game, first, home):
    """The events of game 3's JSON file as the same table as _csv_events (team 0 = home).

    subtypes is null, one {"name": ...} or a list of them; they are joined with "-".

        {"team": {"name": "Team A"}, "type": {"name": "SET PIECE"},
         "subtypes": {"name": "KICK OFF"}, "start": {"frame": 361}, "end": {"frame": 361},
         "period": 1, ...}                 (home "Team A", first frame 1)
        -> team 0, type "SET PIECE", subtype "KICK OFF", period 1, start 360, end 360
    """
    path = os.path.join(DATA_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_events.json".format(game))
    with open(path) as f:
        data = json.load(f)["data"]
    rows = []
    for e in data:
        subs = e["subtypes"]
        subs = subs if isinstance(subs, list) else [subs] if subs else []
        rows.append({"team": 0 if e["team"]["name"] == home else 1, "type": e["type"]["name"],
                     "subtype": "-".join(s["name"] for s in subs), "period": int(e["period"]),
                     "start": int(e["start"]["frame"]) - first,
                     "end": int(e["end"]["frame"]) - first})
    return pd.DataFrame(rows)


def parse(game):
    """Read one sample match from external/metrica/data/ into a MetricaMatch (no cache).

    Games 1-2: the two team CSV files (home slots first) and the CSV events; game 3: EPTS and
    JSON (_epts, _json_events). The events are sorted by start row (game 3 lists a CARRY after
    the BALL OUT it led to), keeping the file order among equal starts.

        m = parse(1)
        m.n_frames, m.players_xy.shape[1], len(m.events)     # -> (145006, 28, 1745)
    """
    if game in (1, 2):
        base = os.path.join(DATA_DIR, "Sample_Game_{0}".format(game),
                            "Sample_Game_{0}_RawTrackingData_".format(game))
        period, frame, home_xy, ball = _csv_team(base + "Home_Team.csv")
        _, frame_away, away_xy, _ = _csv_team(base + "Away_Team.csv")
        if not np.array_equal(frame, frame_away):
            raise ValueError("game {}: the two teams' frames differ".format(game))
        xy = np.concatenate([home_xy, away_xy], axis=1)
        slot_team = np.r_[np.zeros(home_xy.shape[1]), np.ones(away_xy.shape[1])].astype(np.int8)
        events = _csv_events(os.path.join(DATA_DIR, "Sample_Game_{0}".format(game),
                                          "Sample_Game_{0}_RawEventsData.csv".format(game)),
                             frame[0])
        half = period
    elif game == 3:
        frame, half, xy, slot_team, ball, home = _epts(game)
        events = _json_events(game, frame[0], home)
    else:
        raise ValueError("game must be 1, 2 or 3")
    if not np.array_equal(frame, frame[0] + np.arange(len(frame))):
        raise ValueError("game {}: frames are not consecutive".format(game))
    events = events.sort_values("start", kind="stable").reset_index(drop=True)
    return MetricaMatch(game=game, frame=frame, half=np.asarray(half, dtype=np.int8),
                        ball_xy=ball.astype(np.float32), players_xy=xy.astype(np.float32),
                        slot_team=slot_team, events=events)


def load(game):
    """The MetricaMatch of one game: from cache/external/metrica/game<k>.npz, parsed if needed.

        m = load(3)                    # ~1 min the first time, ~1 s afterwards
    """
    path = os.path.join(CACHE, "game{}.npz".format(game))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) == CACHE_VERSION:
                ev = pd.DataFrame({c[3:]: data[c] for c in data.files if c.startswith("ev_")})
                return MetricaMatch(game=game, frame=data["frame"], half=data["half"],
                                    ball_xy=data["ball_xy"], players_xy=data["players_xy"],
                                    slot_team=data["slot_team"], events=ev)
    m = parse(game)
    os.makedirs(CACHE, exist_ok=True)
    ev = {"ev_" + c: (m.events[c].to_numpy() if pd.api.types.is_numeric_dtype(m.events[c])
                      else m.events[c].to_numpy().astype(str)) for c in m.events.columns}
    np.savez(path, version=CACHE_VERSION, frame=m.frame, half=m.half, ball_xy=m.ball_xy,
             players_xy=m.players_xy, slot_team=m.slot_team, **ev)
    return m


def stoppage_start(event):
    """The row where play stops because of this event, or None if it does not stop play.

    event is a row of MetricaMatch.events (type, subtype, start, end). The ball crossing a line
    is the END of a BALL OUT or a shot going OUT / in as a GOAL; a foul stops play at the START
    of a FAULT RECEIVED or a CHALLENGE with FAULT (not ADVANTAGE); the referee stopping play
    (FORCED), an OFFSIDE flag or a HAND BALL is at the END of that BALL LOST.

        stoppage_start({"type": "BALL OUT", "subtype": "", "start": 10, "end": 60})   # -> 60
        stoppage_start({"type": "CHALLENGE", "subtype": "TACKLE-FAULT-WON",
                        "start": 10, "end": 10})                                    # -> 10
        stoppage_start({"type": "CHALLENGE", "subtype": "TACKLE-ADVANTAGE-WON",
                        "start": 10, "end": 10})                                    # -> None
        stoppage_start({"type": "SHOT", "subtype": "ON TARGET-SAVED",
                        "start": 10, "end": 30})                                    # -> None
    """
    kind, sub = event["type"], event["subtype"].split("-")
    if kind == "BALL OUT" or (kind == "SHOT" and ("OUT" in sub or "GOAL" in sub)):
        return int(event["end"])
    if kind == "FAULT RECEIVED" or (kind == "CHALLENGE" and "FAULT" in sub
                                    and "ADVANTAGE" not in sub):
        return int(event["start"])
    if kind == "BALL LOST" and ({"FORCED", "OFFSIDE", "HAND BALL"} & set(sub)):
        return int(event["end"])
    return None


def is_restart(event):
    """True for an event that brings the ball back into play: a SET PIECE, or a RECOVERY with
    no subtype (Metrica's restart from a dead ball that is not a set piece: goal kicks in games
    1-2, drop balls).

        is_restart({"type": "SET PIECE", "subtype": "THROW IN"})       # -> True
        is_restart({"type": "RECOVERY", "subtype": ""})                # -> True
        is_restart({"type": "RECOVERY", "subtype": "INTERCEPTION"})    # -> False
    """
    return event["type"] == "SET PIECE" or (event["type"] == "RECOVERY"
                                            and event["subtype"] == "")


def graded(m):
    """bool per row: inside a half, from its first event (the kick-off) to the end of its last.

    Before the kick-off and after the half's last event (the referee's whistle, a ball out
    that is never restarted) there is no play to label, as half-time in our match.

        half = 1 on rows 0-99, events of period 1 from row 10 (start) to row 90 (latest end)
        graded(m)                      # -> True on rows 10-90
    """
    out = np.zeros(m.n_frames, dtype=bool)
    for number in (1, 2):
        ev = m.events[m.events.period == number]
        rows = np.flatnonzero(m.half == number)
        if len(ev) == 0 or len(rows) == 0:
            continue
        a = max(int(ev.start.min()), rows[0])
        b = min(int(max(ev.start.max(), ev.end.max())), rows[-1])
        out[a:b + 1] = True
    return out


def dead(m):
    """(bool per row: the ball is dead, list of (start, restart) stoppages kept, number dropped).

    For every event that stops play (stoppage_start, at row t), the restart is the first later
    event with is_restart() starting at or after t, in the same period. If an event of the run
    of play (LIVE types, not a restart) starts strictly between t + TOLERANCE and the restart,
    play went on (advantage, or a whistle logged early) and the stoppage is dropped. A stoppage
    with no restart in its period runs to the period's end. Overlapping stoppages (a foul and the
    FAULT RECEIVED of the same moment) merge.

        events: BALL OUT ending at row 100, SET PIECE THROW IN at row 300, PASS at 300
        dead(m)                        # -> D on rows 100-299, [(100, 300)], 0 dropped
        events: CHALLENGE TACKLE-FAULT at row 50, PASS of the same team at 90, SET PIECE at 400
        dead(m)                        # -> no D: play went on after the foul, 1 dropped
    """
    ev = m.events
    out = np.zeros(m.n_frames, dtype=bool)
    kept, dropped = [], 0
    restart = ev.apply(is_restart, axis=1).to_numpy()
    live = ev.type.isin(LIVE).to_numpy() & ~restart
    starts, periods = ev.start.to_numpy(), ev.period.to_numpy()
    for i, e in ev.iterrows():
        t = stoppage_start(e)
        if t is None:
            continue
        later = (periods == e["period"]) & (starts >= t)
        r = np.flatnonzero(later & restart)
        stop = starts[r[0]] if len(r) else np.flatnonzero(m.half == e["period"])[-1] + 1
        between = later & live & (starts > t + TOLERANCE) & (starts < stop)
        if between.any():
            dropped += 1
            continue
        if stop > t:
            out[t:stop] = True
            kept.append((t, int(stop)))
    return out, kept, dropped


def owner(m):
    """0 / 1 per row: the team of the latest on-ball event (ON_BALL types) at or before it.

    Each on-ball event says its team had the ball at its start row, so the owner is a step
    function of the event starts; rows before the first on-ball event take that event's team.
    Challenges and cards say nothing about who has the ball.

        events: home PASS at row 0, home BALL LOST at row 40, away RECOVERY at row 55
        owner(m)[[0, 39, 40, 54, 55, 99]]      # -> [0, 0, 0, 0, 1, 1]
    """
    ev = m.events[m.events.type.isin(ON_BALL)]
    starts = np.clip(ev.start.to_numpy(), 0, m.n_frames - 1)
    teams = ev.team.to_numpy()
    out = np.full(m.n_frames, teams[0], dtype=np.int64)
    for s, team in zip(starts, teams):      # sorted by start: later events overwrite
        out[s:] = team
    return out


def controlled_events(m):
    """The events table in the form controlled.with_action() reads: team, row, controlled.

    controlled is True for the controlled actions: PASS (always completed in Metrica), SHOT,
    SET PIECE, FAULT RECEIVED (a foul on the player in possession) and RECOVERY with THEFT (the
    ball stolen from an opponent in possession). Not CARRY (game 3 only, logged for almost any
    movement with the ball; module docstring, step 3).

        PASS -> True; RECOVERY THEFT -> True; RECOVERY INTERCEPTION -> False;
        CARRY -> False; BALL LOST -> False; CHALLENGE AERIAL-WON -> False
    """
    ev = m.events
    theft = (ev.type == "RECOVERY") & ev.subtype.str.split("-").apply(lambda s: "THEFT" in s)
    use = ev.type.isin(CONTROLLED) | theft
    return pd.DataFrame({"team": ev.team.to_numpy(), "row": ev.start.to_numpy(),
                         "controlled": use.to_numpy()})


def labels(m, rule="or", min_seconds=MIN_SECONDS):
    """H / A / D per row from the events, X outside the graded part of each half.

    dead() gives D, owner() the raw team, and controlled.labels() keeps only the changes the
    rule accepts (module docstring, step 3; rule as in controlled.RULES: "raw" keeps every
    change, "or" is the main label, "and" the strict check). Rows not graded() are X.

        a half: home passes from row 0, away RECOVERY INTERCEPTION at row 100, away BALL LOST
        at row 101 (a header straight back), home RECOVERY at row 130, ...
        labels(m)            # -> H on rows 0-129 and on: the away touch (1.2 s) is no change
        labels(m, "raw")     # -> H 0-99, A 100-129, H 130-...
    """
    from types import SimpleNamespace
    alive = graded(m) & ~dead(m)[0]
    view = SimpleNamespace(possession=owner(m) + 1, alive=alive,
                           half=np.where(graded(m), m.half, 0))
    out = controlled.labels(view, controlled_events(m), rule, min_seconds, TOLERANCE)
    out[~graded(m)] = "X"
    return out
