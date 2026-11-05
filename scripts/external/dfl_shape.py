"""How well do the players alone tell H / A / D? Leave-one-match-out on the 7 DFL matches.

    sbatch --time=02:00:00 scripts/slurm/run.sbatch scripts/external/dfl_shape.py

For each DFL match, a shape model (mtah/external/shape_transfer.py) is trained on the other six
and predicts the held-out one, with the ball hidden everywhere: only the degraded players are
used. These are public matches with true labels (controlled possession, OR rule), so the
numbers are real held-out accuracy, not agreement with pseudo-labels.

The first run builds every match's features (~4 min a match; cached afterwards). Then, per
held-out match and on average:

  accuracy        share of frames whose most likely class is the label
  F1 H / A / D    per-class F1 of the most likely class, and macroF1 (their mean, as the
                  challenge's tie-break)
  team acc        on alive frames, how often the more likely of H and A is the right team: the
                  question our current shape model answers
  D AUC           how well P(D) ranks dead frames above alive ones (0.5 chance, 1 perfect)
  calibration     over all held-out frames, binned by the probability of the predicted class:
                  how often it is right. Honest probabilities give about the bin's value.

The held-out probabilities are saved to cache/external/dfl/<match>_shape_oof.npy, for
fitting how the HMM should weigh them later (gitignored cache; DFL is public data).
"""

import os
import sys

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.external import adapter, dfl  # noqa: E402
from mtah.external import shape_transfer as st  # noqa: E402

# Params
BINS = (0.33, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def scores(labels, p):
    """Dict of the per-match numbers in the module docstring, for labels and (n, 3) probabilities.

    labels is H / A / D per frame, p the predict() output in that order.

        labels = ["H", "A", "D", "D"], p = [[.8, .1, .1], [.2, .7, .1], [.1, .1, .8], [.6, .3, .1]]
        scores(labels, p)
        # -> accuracy 0.75, F1_H 0.667, F1_A 1.0, F1_D 0.667, macroF1 0.778,
        #    team_acc 1.0 (both alive frames right), D_AUC 0.75 (the last D frame ties with
        #    both alive frames at 0.1)
    """
    labels = np.asarray(labels)
    p = np.asarray(p)
    predicted = np.array(st.CLASSES)[p.argmax(axis=1)]
    f1 = f1_score(labels, predicted, labels=list(st.CLASSES), average=None, zero_division=0)
    alive = labels != "D"
    team = np.where(p[:, 0] >= p[:, 1], "H", "A")
    return {"accuracy": float((predicted == labels).mean()),
            "F1_H": f1[0], "F1_A": f1[1], "F1_D": f1[2], "macroF1": float(f1.mean()),
            "team_acc": float((team[alive] == labels[alive]).mean()),
            "D_AUC": float(roc_auc_score(~alive, p[:, 2]))}


if __name__ == '__main__':
    gaps = adapter.presence_gaps(load())
    data = []
    for k, match_id in enumerate(dfl.MATCHES):
        table, labels = st.build(match_id, gaps, seed=k)
        data.append((match_id, table, labels))
        print("built {}: {} frames, D share {:.2f}".format(match_id, len(table),
                                                           (labels == "D").mean()), flush=True)

    keys = ["accuracy", "F1_H", "F1_A", "F1_D", "macroF1", "team_acc", "D_AUC"]
    print("\nheld-out match   " + "".join("{:>10}".format(c) for c in keys))
    rows, every_label, every_p = [], [], []
    for k, (match_id, table, labels) in enumerate(data):
        others = [d for j, d in enumerate(data) if j != k]
        model = st.fit([d[1] for d in others], [d[2] for d in others])
        p = st.predict(model, table)
        np.save(os.path.join(dfl.CACHE, match_id + "_shape_oof.npy"), p)
        row = scores(labels, p)
        rows.append(row)
        every_label.append(labels)
        every_p.append(p)
        print("{:<17}".format(match_id) + "".join("{:>10.3f}".format(row[c]) for c in keys),
              flush=True)
    print("{:<17}".format("mean") + "".join(
        "{:>10.3f}".format(np.mean([r[c] for r in rows])) for c in keys))

    labels = np.concatenate(every_label)
    p = np.concatenate(every_p)
    top = p.max(axis=1)
    right = np.array(st.CLASSES)[p.argmax(axis=1)] == labels
    print("\ncalibration: probability of the predicted class vs how often it is right")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        rows_ = (top >= lo) & (top < hi)
        print("   {:.2f}-{:.2f}  frames {:7d}  right {:.3f}".format(
            lo, min(hi, 1.0), int(rows_.sum()), right[rows_].mean() if rows_.any() else np.nan))
