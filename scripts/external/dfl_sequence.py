"""TCN vs BiLSTM vs boosting: H / A / D from the players alone, leave-one-match-out on DFL.

    sbatch --partition=a100 --gres=gpu:a100:1 --time=04:00:00 scripts/slurm/run.sbatch \\
        scripts/external/dfl_sequence.py
    python scripts/external/dfl_sequence.py --quick      # 2 folds, 2 epochs: a smoke test

The question: does reading 30 s of team shape at once (mtah/external/shape_sequence.py) label
possession better than the boosting model, which sees one frame (mtah/external/shape_transfer.py)?
And which of the two sequence models is better? All three get the same 36 ball-free features of
the same degraded DFL players and the same OR-rule labels (shape_transfer.build(), cached).

Splits, for each of the 7 DFL matches in turn:

  test        the held-out match: only scored, never used to train or to stop training
  validation  the next match in dfl.MATCHES (wrapping round): early stopping only
  train       the other 5 matches

The boosting numbers are its existing leave-one-match-out predictions
(cache/external/dfl/<match>_shape_oof.npy, from scripts/external/dfl_shape.py), trained on all
6 other matches: it has one training match more than the sequence models.

Every sequence model is trained with SEEDS (3) seeds per fold, since neural nets vary from run
to run; the rows are:

  tcn, lstm         each seed scored on its own, averaged over seeds (the expected result of
                    one training run); the spread is the std over seeds of the 7-fold mean
  tcn x3, lstm x3   the 3 seeds' probabilities averaged, then scored (a small ensemble)
  tcn+lstm          the two ensembles averaged
  boosting          the baseline

Columns per held-out match and as a 7-fold mean (std over the folds):

  spellF1         the challenge's ranking score on the argmax labels of every frame, each half a
                  scored run (controlled.spell_f1), against the true DFL labels
  spellF1 1s      the same after averaging the probabilities over a centred 1 s window, for
                  every model alike: a per-frame model's labels flicker, and this shows how
                  much of a difference is only smoothness
  macroF1, F1 H / A / D, accuracy, team acc (alive frames), D AUC   as in dfl_shape.py

The seed ensembles' held-out probabilities are saved to
cache/external/dfl/<match>_<kind>_oof.npy (public data) for the HMM work that follows.
Aggregates only are printed.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "external"))
from dfl_shape import scores  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.external import adapter, controlled, dfl  # noqa: E402
from mtah.external import shape_sequence as sq  # noqa: E402
from mtah.external import shape_transfer as st  # noqa: E402

# Params
FPS = 25
SEEDS = (0, 1, 2)
SMOOTH = FPS                # 1 s window for "spellF1 1s"
KEYS = ["spellF1", "spellF1 1s", "macroF1", "F1_H", "F1_A", "F1_D", "accuracy", "team_acc",
        "D_AUC"]


def smoothed(p, half, size=SMOOTH):
    """p averaged over a centred window of `size` frames, within each half (rows renormalised).

        p = [[1, 0, 0], [0, 1, 0], [1, 0, 0]], half = [1, 1, 1], size 3
        smoothed(p, half, 3)[1]        # -> [0.667, 0.333, 0.0]   (the edges repeat their row)
    """
    p = np.asarray(p, dtype=np.float64)
    half = np.asarray(half)
    out = np.empty_like(p)
    for h in np.unique(half):
        rows = half == h
        out[rows] = uniform_filter1d(p[rows], size, axis=0, mode="nearest")
    return out / out.sum(axis=1, keepdims=True)


def all_scores(labels, p, half):
    """scores() of dfl_shape.py plus spellF1 of the argmax labels, raw and after smoothed().

        labels = H H H A A D D, p one-hot on the same, one half
        all_scores(labels, p, half)["spellF1"]  # -> 1.0
    """
    out = scores(labels, p)
    classes = np.array(st.CLASSES)
    out["spellF1"] = controlled.spell_f1(labels, classes[np.asarray(p).argmax(axis=1)], half)
    out["spellF1 1s"] = controlled.spell_f1(labels, classes[smoothed(p, half).argmax(axis=1)],
                                            half)
    return out


def line(name, values):
    return "{:<16}".format(name) + "".join("{:>11.3f}".format(v) for v in values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="+", default=list(sq.KINDS), choices=sq.KINDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--quick", action="store_true", help="2 folds, 2 epochs, 1 seed")
    args = parser.parse_args()
    if args.quick:
        sq.MAX_EPOCHS, sq.PATIENCE, args.seeds = 2, 1, args.seeds[:1]

    gaps = adapter.presence_gaps(load())
    data = []
    for k, match_id in enumerate(dfl.MATCHES):
        table, labels = st.build(match_id, gaps, seed=k)
        data.append((match_id, table, labels, dfl.load(match_id).half))
    print("device {}, {} matches, seeds {}".format(sq._device(), len(data), args.seeds),
          flush=True)

    folds = range(2) if args.quick else range(len(data))
    results = {}            # method -> list over folds of score dicts (per seed: list of lists)
    per_seed = {kind: {s: [] for s in args.seeds} for kind in args.kinds}
    for k in folds:
        match_id, table, labels, half = data[k]
        v = (k + 1) % len(data)
        train = [d for j, d in enumerate(data) if j not in (k, v)]
        ensembles = {}
        for kind in args.kinds:
            probs = []
            for seed in args.seeds:
                started = time.time()
                fitted = sq.fit([d[1] for d in train], [d[2] for d in train],
                                [d[3] for d in train], kind, seed=seed,
                                valid=(data[v][1], data[v][2], data[v][3]))
                p = sq.predict(fitted, table, half)
                probs.append(p)
                row = all_scores(labels, p, half)
                per_seed[kind][seed].append(row)
                print("fold {} {:<5} seed {}  best epoch {:2d} of {:2d}  val loss {:.3f}  "
                      "{:4.0f} s  spellF1 {:.3f}  macroF1 {:.3f}".format(
                          match_id, kind, seed, fitted.best_epoch, len(fitted.history),
                          min(fitted.history), time.time() - started, row["spellF1"],
                          row["macroF1"]), flush=True)
            ensembles[kind] = np.mean(probs, axis=0)
            if not args.quick:
                np.save(os.path.join(dfl.CACHE, "{}_{}_oof.npy".format(match_id, kind)),
                        ensembles[kind].astype(np.float32))
        methods = {"{} x{}".format(kind, len(args.seeds)): p for kind, p in ensembles.items()}
        if len(ensembles) == 2:
            methods["tcn+lstm"] = (ensembles["tcn"] + ensembles["lstm"]) / 2
        methods["boosting"] = np.load(os.path.join(dfl.CACHE, match_id + "_shape_oof.npy"))
        for name, p in methods.items():
            results.setdefault(name, []).append(all_scores(labels, p, half))

    names = list(results)
    for kind in args.kinds:     # one training run: mean over seeds of each fold's scores
        results[kind] = [{c: np.mean([per_seed[kind][s][f][c] for s in args.seeds])
                          for c in KEYS} for f in range(len(folds))]
    order = list(args.kinds) + names

    print("\nper held-out match: spellF1 / macroF1")
    print("{:<16}".format("match") + "".join("{:>18}".format(n) for n in order))
    for f, k in enumerate(folds):
        print("{:<16}".format(data[k][0]) + "".join(
            "{:>11.3f} / {:.3f}".format(results[n][f]["spellF1"], results[n][f]["macroF1"])
            for n in order))

    print("\nmean over the {} held-out matches (std over matches below)".format(len(folds)))
    print("{:<16}".format("method") + "".join("{:>11}".format(c) for c in KEYS))
    for n in order:
        print(line(n, [np.mean([r[c] for r in results[n]]) for c in KEYS]))
        print(line("   +-", [np.std([r[c] for r in results[n]]) for c in KEYS]))

    print("\nseed spread: std over seeds of the {}-fold mean".format(len(folds)))
    for kind in args.kinds:
        print(line(kind, [np.std([np.mean([r[c] for r in per_seed[kind][s]])
                                  for s in args.seeds]) for c in KEYS]))

    print("\npaired: folds where the method beats boosting on spellF1 / macroF1")
    for n in order:
        if n == "boosting":
            continue
        wins = [(r["spellF1"] > b["spellF1"], r["macroF1"] > b["macroF1"])
                for r, b in zip(results[n], results["boosting"])]
        print("   {:<14} {} / {} of {}".format(n, sum(w[0] for w in wins),
                                              sum(w[1] for w in wins), len(wins)))
