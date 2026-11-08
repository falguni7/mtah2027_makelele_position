"""H / A / D from the players alone, learned on DFL with the ball hidden everywhere.

    from mtah.external import shape_transfer as st
    table, labels = st.build("DFL-MAT-J03WMX", gaps, seed=0)   # features + OR labels, cached
    model = st.fit([table, ...], [labels, ...])                # gradient boosting, 3 classes
    p = st.predict(model, table)                               # (n, 3): P(H), P(A), P(D)

Our current shape model (mtah/possession/shape_model.py) learns "which team has the ball" from
pseudo-labels of our own match: frames inside control events, where the ball is tracked. DFL
has true labels on every frame, gaps and stoppages included, so a model trained there can learn
what possession and dead ball look like from the players alone, also in the situations our
pseudo-labels never cover. It predicts three classes, H, A and D (agreed 2026-11-05).

The pieces:

  features   the same team-shape features as for BAR-ATH (mtah/features/team_shape.py),
             computed on the DFL match dressed up as our Match (mtah/external/adapter.py) with
             its players degraded to look like ours. The camera features do not exist on DFL
             and are left out: COLUMNS.
  labels     controlled possession, the OR rule (mtah/external/controlled.py): H / A / D.
  symmetry   "home" means nothing across matches: what the model may learn is how the team in
             possession stands relative to the other. So every training frame is used twice,
             as it is and mirrored (mirror(): the two teams' columns swapped, the differences
             negated, H and A swapped), and predict() averages the answer for a frame and its
             mirror. The model therefore cannot prefer "home", and treats both teams the same.
  model      HistGradientBoosting, as elsewhere in the pipeline (a neural net is kept for later).

In this file: COLUMNS, CLASSES, build() (one match's features and labels, cached), mirror(),
fit(), predict() and for_match() (the model trained on all 7 DFL matches applied to our match,
cached), which is how hmm_v9 gets it.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from mtah.external import adapter, controlled, dfl, dfl_events
from mtah.features import team_shape

# Params
CLASSES = ("H", "A", "D")
COLUMNS = [c for c in team_shape.COLUMNS if not c.startswith(("near_camera", "camera_"))]
TRAIN_STRIDE = 5            # every 5th frame: neighbours are near-copies
MODEL = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
             random_state=0)
FEATURE_VERSION = 1         # bump when the features or the degradation change


def build(match_id, gaps, seed):
    """(feature table with COLUMNS, H / A / D labels) of one DFL match, degraded, cached.

    Loads the match and its events, makes the OR labels, degrades the players with
    np.random.default_rng(seed) and our match's drop-out lengths `gaps`
    (adapter.presence_gaps), and computes the shape features (~4 minutes a match, mostly the
    team outlines). The result is kept in cache/external/dfl/<match>_shape.npz and read from
    there when the FEATURE_VERSION and the seed match. Rows are the DFL tracking rows.

        table, labels = build("DFL-MAT-J03WMX", gaps, seed=0)
        table.shape, labels[:3]       # -> (145967, 36), array(['A', 'A', 'A'])
    """
    path = os.path.join(dfl.CACHE, match_id + "_shape.npz")
    if os.path.exists(path):
        with np.load(path) as data:
            if int(data["version"]) == FEATURE_VERSION and int(data["seed"]) == seed:
                return pd.DataFrame(data["table"], columns=COLUMNS), data["labels"]
    m = dfl.load(match_id)
    labels = controlled.labels(m, dfl_events.load(m))
    match = adapter.as_match(m, np.random.default_rng(seed), gaps)
    table = team_shape.compute(match)[COLUMNS]
    np.savez(path, version=FEATURE_VERSION, seed=seed, table=table.to_numpy(np.float32),
             labels=labels)
    return table, labels


def mirror(table, labels=None):
    """The same frames seen with the teams swapped: (table, labels) or just the table.

    Every _h column takes the _a column's values and the other way round, every _diff column
    is negated (home minus away becomes away minus home), and H and A labels are swapped (D
    stays D). Columns are oriented by each team's own attacking direction, so nothing else
    changes.

        table: advance_h 5, advance_a -3, advance_diff 8; labels ["H"]
        mirror(table, labels)          # -> advance_h -3, advance_a 5, advance_diff -8; ["A"]
    """
    out = table.copy()
    for column in table.columns:
        if column.endswith("_h"):
            out[column] = table[column[:-2] + "_a"].to_numpy()
            out[column[:-2] + "_a"] = table[column].to_numpy()
        elif column.endswith("_diff"):
            out[column] = -table[column].to_numpy()
    if labels is None:
        return out
    swapped = np.where(labels == "H", "A", np.where(labels == "A", "H", labels))
    return out, swapped


def fit(tables, labels, stride=TRAIN_STRIDE):
    """A HistGradientBoosting classifier of H / A / D from COLUMNS, trained on several matches.

    tables and labels are lists, one entry per match (from build()). Every `stride`-th frame of
    every match is used, together with its mirror (mirror()), so H and A appear equally often
    and in swapped roles. The classes are coded 0 H, 1 A, 2 D (CLASSES).

        model = fit([table_1, table_2], [labels_1, labels_2])
        model.classes_                 # -> array([0, 1, 2])
    """
    X, y = [], []
    for table, lab in zip(tables, labels):
        table, lab = table.iloc[::stride][COLUMNS], np.asarray(lab)[::stride]
        flipped, flipped_lab = mirror(table, lab)
        X += [table.to_numpy(), flipped.to_numpy()]
        y += [lab, flipped_lab]
    code = {c: k for k, c in enumerate(CLASSES)}
    model = HistGradientBoostingClassifier(**MODEL)
    model.fit(np.concatenate(X), np.array([code[v] for v in np.concatenate(y)]))
    return model


def predict(model, table):
    """(n, 3) float32 probabilities P(H), P(A), P(D) per frame, symmetric in the two teams.

    The model's answer for each frame and for its mirror (with P(H) and P(A) of the mirror
    swapped back) are averaged, so swapping which team is called home swaps P(H) and P(A)
    exactly and leaves P(D) unchanged.

        p = predict(model, table)
        p[0]                           # -> e.g. [0.70, 0.20, 0.10]
        predict(model, mirror(table))[0]   # -> [0.20, 0.70, 0.10]
    """
    straight = model.predict_proba(table[COLUMNS].to_numpy())
    flipped = model.predict_proba(mirror(table[COLUMNS]).to_numpy())[:, [1, 0, 2]]
    return ((straight + flipped) / 2).astype(np.float32)


def for_match(match, path=None, gaps=None, extra=()):
    """(n_frames, 3) float32 P(H), P(A), P(D) of our match from the model trained on all DFL.

    The shape model is trained on all 7 DFL matches (build(), seed k for the k-th match, players
    degraded with our match's drop-out lengths, adapter.presence_gaps) and predicts every graded
    frame of `match` from the same team-shape features (mtah/features/team_shape.py); other
    frames are NaN. Nothing of our match is used for training except those drop-out lengths.
    The features use no ball, so a masked copy of the match (mtah/validation/masking) gives the
    same answer. The result is kept in cache/external/dfl/bar_ath_shape_v<FEATURE_VERSION>.npy
    (gitignored: it is derived from our match) and read from there when its length matches.
    Needs the DFL data in external/dfl/ (scripts/data/download_external.sh) on the first run.

    Another match (a Metrica benchmark match, mtah/external/metrica_bench.py) passes its own
    cache `path`, so our match's cache is never overwritten, and `gaps`, the drop-out lengths
    the DFL players were degraded with (ours; default: adapter.presence_gaps(match)).

    extra adds labelled matches to the 7 DFL ones, as (name, table, labels, half) entries
    (mtah/external/pool.py: the Metrica matches; hmm_v12). A caller passing extra must also pass
    its own path, so the DFL-only cache is never mixed up with it.

        p = for_match(load())
        p[graded][0]                   # -> e.g. [0.05, 0.03, 0.92]   a frame before kick-off
        for_match(metrica_match, path=".../metrica_game1_shape.npy", gaps=our_gaps)
    """
    if path is None:
        path = os.path.join(dfl.CACHE, "bar_ath_shape_v{}.npy".format(FEATURE_VERSION))
    if os.path.exists(path):
        p = np.load(path)
        if len(p) == match.n_frames:
            return p
    if gaps is None:
        gaps = adapter.presence_gaps(match)
    built = [build(match_id, gaps, seed=k) for k, match_id in enumerate(dfl.MATCHES)]
    model = fit([b[0] for b in built] + [e[1] for e in extra],
                [b[1] for b in built] + [e[2] for e in extra])
    graded = np.asarray(match.graded)
    p = np.full((match.n_frames, 3), np.nan, dtype=np.float32)
    p[graded] = predict(model, team_shape.compute(match)[graded])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, p)
    return p
