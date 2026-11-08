"""Leave-one-match-out DFL predictions of the DFL models trained with the 3 Metrica matches added.

    sbatch --time=04:00:00 scripts/slurm/run.sbatch scripts/external/pool_oof.py
    python scripts/external/pool_oof.py --quick        # 1 fold, 1 seed, 2 epochs: a smoke test

The DFL benchmark (scripts/external/dfl_benchmark.py) scores the whole method on each DFL match
with predictions of models that never trained on that match (dfl_shape.py, dfl_sequence.py).
This script makes the same predictions for models whose training data also holds the 3
Metrica matches (mtah/external/pool.py), so the *_pool variants of the benchmark (hmm_v12) can
be compared with the DFL-only ones on exactly the same footing. For every DFL match k:

  boosting   shape_transfer.fit on the other 6 DFL matches + the 3 Metrica matches
  sequence   for each kind (TCN, BiLSTM) and seed in shape_sequence.SEEDS: fit on the other 5
             DFL matches + the 3 Metrica matches, early stopping on DFL match k + 1 (wrapping
             round), as in dfl_sequence.py; the seeds of a kind averaged

Saved to cache/external/dfl/<match>_shape_pool_oof.npy and <match>_<kind>_pool_oof.npy (public
data). Printed per match: team accuracy on alive frames and D AUC of the boosting model and of
the TCN + LSTM average, old (DFL only, the cached *_oof.npy) against new (pool).
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.external import adapter, dfl, pool  # noqa: E402
from mtah.external import benchmark as bm  # noqa: E402
from mtah.external import shape_sequence as sq  # noqa: E402
from mtah.external import shape_transfer as st  # noqa: E402


def quality(labels, p):
    """(team accuracy on alive frames, D AUC) of (n, 3) probabilities against H / A / D labels.

        labels = H A D D, p rows [.8 .1 .1], [.3 .6 .1], [.1 .1 .8], [.2 .2 .6]
        quality(labels, p)             # -> (1.0, 1.0)
    """
    labels = np.asarray(labels)
    alive = labels != "D"
    team = np.where(p[:, 0] >= p[:, 1], "H", "A")
    return float((team[alive] == labels[alive]).mean()), float(roc_auc_score(~alive, p[:, 2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="1 fold, 1 seed, 2 epochs")
    args = parser.parse_args()
    seeds = list(sq.SEEDS)
    if args.quick:
        sq.MAX_EPOCHS, sq.PATIENCE, seeds = 2, 1, seeds[:1]

    gaps = adapter.presence_gaps(load())
    data = pool.dfl_entries(gaps)
    extra = pool.metrica_entries(gaps)
    print("device {}; {} DFL + {} Metrica matches; seeds {}".format(
        sq._device(), len(data), len(extra), seeds), flush=True)
    print("{:<16}{:>22}{:>22}".format("held out", "boosting team / D AUC",
                                      "sequence team / D AUC"))
    print("{:<16}{:>22}{:>22}".format("", "old -> new", "old -> new"))
    folds = range(1) if args.quick else range(len(data))
    for k in folds:
        name, table, labels, half = data[k]
        started = time.time()
        others = [d for j, d in enumerate(data) if j != k] + extra
        model = st.fit([d[1] for d in others], [d[2] for d in others])
        boost = st.predict(model, table)
        v = (k + 1) % len(data)
        train = [d for j, d in enumerate(data) if j not in (k, v)] + extra
        kinds = {}
        for kind in sq.KINDS:
            runs = [sq.predict(sq.fit([d[1] for d in train], [d[2] for d in train],
                                      [d[3] for d in train], kind, seed=seed,
                                      valid=(data[v][1], data[v][2], data[v][3])),
                               table, half) for seed in seeds]
            kinds[kind] = np.mean(runs, axis=0).astype(np.float32)
        if not args.quick:
            np.save(os.path.join(dfl.CACHE, name + "_shape_pool_oof.npy"), boost)
            for kind, p in kinds.items():
                np.save(os.path.join(dfl.CACHE, "{}_{}_pool_oof.npy".format(name, kind)), p)
        seq = (kinds["tcn"] + kinds["lstm"]) / 2
        old_b = quality(labels, bm.dfl_probs(name, "boosting"))
        old_s = quality(labels, bm.dfl_probs(name, "sequence"))
        new_b, new_s = quality(labels, boost), quality(labels, seq)
        print("{:<16}  {:.3f}/{:.3f} -> {:.3f}/{:.3f}  {:.3f}/{:.3f} -> {:.3f}/{:.3f}  "
              "({:.0f} s)".format(name, *old_b, *new_b, *old_s, *new_s,
                                  time.time() - started), flush=True)
