"""The labelled external matches the DFL-trained models can learn from: 7 DFL + 3 Metrica.

    from mtah.external import pool
    gaps = adapter.presence_gaps(load())             # our players' drop-out lengths
    entries = pool.metrica_entries(gaps)             # [(name, table, labels, half)] x 3
    entries[0][0], entries[0][1].shape               # -> ("metrica_game1", (142074, 36))

The boosting shape model (shape_transfer.py) and the TCN + LSTM sequence model
(shape_sequence.py) were trained on the 7 DFL matches only. Metrica's 3 sample matches
(metrica.py) can be added as training data (agreed 2026-11-07): their players come from the
company that tracked ours, in the same format, and their labels from human events (another view
of controlled possession than DFL's tracking owner). The models read only the players, so
Metrica's ball, which vanishes at stoppages, does not matter here.

An entry is (name, table, labels, half), the triple fit() takes plus a name:

  table    the team-shape features (shape_transfer.COLUMNS) of the match with its players
           degraded towards ours: Metrica's are cleaned (22 players always tracked), so they are
           degraded exactly as in the Metrica benchmark (metrica_bench.as_match, seed
           metrica_bench.DEGRADE_SEED + game, our drop-out lengths)
  labels   H / A / D, the OR rule (metrica.labels)
  half     1 / 2 per row

Only the graded rows of a Metrica match are kept (kick-off to each half's last event), so no row
is X; the two halves stay contiguous, so the sequence models read them as two sequences. DFL
entries come from shape_transfer.build (every frame labelled).

Metrica entries are cached in cache/external/metrica/game<k>_train_v<VERSION>.npz (~4 minutes a
match the first time, mostly the team outlines).

Leave-one-match-out: a Metrica game used as a TEST match must never be in its own training
pool; metrica_entries(gaps, exclude=game) leaves it out.
"""

import os

import numpy as np
import pandas as pd

from mtah.external import metrica

# Params
VERSION = 1                 # bump when the Metrica training tables change


def metrica_entry(game, gaps):
    """(name, table, labels, half) of one Metrica game for training, graded rows only, cached.

    gaps are our players' drop-out lengths (adapter.presence_gaps(load())); the players are
    degraded with metrica_bench.as_match and seed metrica_bench.DEGRADE_SEED + game, the labels
    are metrica.labels(m) (OR rule). The cache is used when its VERSION matches.

        name, table, labels, half = metrica_entry(1, gaps)
        name, len(table), set(labels), set(half)
        # -> ("metrica_game1", 142074, {"H", "A", "D"}, {1, 2})
    """
    from mtah.external import metrica_bench as mb
    from mtah.external import shape_transfer as st
    name = "metrica_game{}".format(game)
    path = os.path.join(metrica.CACHE, "game{}_train_v{}.npz".format(game, VERSION))
    if os.path.exists(path):
        with np.load(path) as data:
            if int(data["version"]) == VERSION:
                return (name, pd.DataFrame(data["table"], columns=st.COLUMNS),
                        data["labels"].astype("<U1"), data["half"])
    m = metrica.load(game)
    match = mb.as_match(m, np.random.default_rng(mb.DEGRADE_SEED + game), gaps)
    graded = metrica.graded(m)
    table = mb.team_table(match)[graded].reset_index(drop=True)
    labels = metrica.labels(m)[graded]
    half = np.asarray(m.half)[graded]
    os.makedirs(metrica.CACHE, exist_ok=True)
    np.savez(path, version=VERSION, table=table.to_numpy(np.float32), labels=labels, half=half)
    return name, table, labels, half


def metrica_entries(gaps, exclude=None):
    """The Metrica entries of every game in metrica.GAMES except `exclude` (a game number).

        [e[0] for e in metrica_entries(gaps, exclude=2)]   # -> ["metrica_game1", "metrica_game3"]
    """
    return [metrica_entry(g, gaps) for g in metrica.GAMES if g != exclude]


def dfl_entries(gaps):
    """(name, table, labels, half) of the 7 DFL matches, from shape_transfer.build (cached).

        [e[0] for e in dfl_entries(gaps)][:2]   # -> ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1"]
    """
    from mtah.external import dfl
    from mtah.external import shape_transfer as st
    out = []
    for k, match_id in enumerate(dfl.MATCHES):
        table, labels = st.build(match_id, gaps, seed=k)
        out.append((match_id, table, labels, dfl.load(match_id).half))
    return out
