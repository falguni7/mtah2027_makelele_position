"""Fine-tuning the DFL sequence models on BAR-ATH's confident frames: does it help?

    sbatch --time=02:00:00 scripts/slurm/run.sbatch scripts/external/bar_ath_finetune.py
    python scripts/external/bar_ath_finetune.py --quick     # 1 seed, 2 blocks, few epochs

The TCN and BiLSTM of mtah/external/shape_sequence.py, used as their average (chosen on DFL,
scripts/external/dfl_sequence.py), are pretrained on DFL and then fine-tuned on our match's
confident frames (shape_sequence.fine_tune(): control-event team frames, dead-ball rule frames,
alive frames as "H or A"; half of every batch DFL windows). Aggregates only are printed; the
per-frame probabilities go to gitignored files under outputs/ and are not opened here.

Splits:

  DFL         pretraining on 5 matches with a 6th for early stopping (as in dfl_sequence.py);
              the 7th (DFL_TEST) is never trained on, and measures forgetting: how much worse
              the models get on DFL after fine-tuning on our match
  BAR-ATH     the graded frames cut into the shape model's 10 time blocks
              (shape_model.blocks()); for each block, the pretrained model is fine-tuned on the
              confident frames of the other 9 and predicts that block. Every frame's prediction
              therefore comes from a model that did not train on its block's labels (out of
              fold), as for the current shape model. Windows still read the neighbouring
              blocks' players as context; no labels cross.

Each kind is trained with SEEDS seeds and the seeds averaged; "avg" is the TCN + LSTM average.
Rows: pretrained (DFL only, nothing of our match) and fine-tuned, for tcn, lstm and avg; and for
comparison the DFL boosting model (shape_transfer.for_match(), all 7 DFL matches) and the
current evidence of hmm_v7 (our shape model's out-of-fold P(home), the detector's P(dead)).

BAR-ATH columns (none is ground truth; the first two are exactly what fine-tuning trains on, so
they show fitting, not generalisation, even out of fold):

  team acc      on control-event team frames: the more likely of H and A is the event's team
  D AUC         P(D) ranking the rules' DEAD frames above their ALIVE ones; "gaps": the same
                inside ball gaps of 3 s or more. The detector is trained on these labels, so its
                AUC is flattered
  tag far       Smart Tagging's in-possession team on the frames far from control events (where
                the HMM uses the shape model); "no ball": of those, without a tracked ball. The
                hard test: current 0.747 / 0.700, DFL boosting 0.746 / 0.694
  unexpl D      mean P(D) in unexplained ball gaps (mtah/dead_ball/explain.py); undecided
  phase D       mean P(D) inside in-possession phases (should be low)
  D share       mean P(D) over the graded frames (the D share the model expects)

DFL_TEST columns: spellF1, macroF1, D AUC, as in dfl_sequence.py; for fine-tuned models the mean
over the 10 block models.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "external"))
from dfl_sequence import all_scores  # noqa: E402
from transfer_check import auc, tag_agreement, team_call  # noqa: E402
from mtah import features  # noqa: E402
from mtah.audio.whistle import match_whistles  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball import explain  # noqa: E402
from mtah.external import adapter, dfl  # noqa: E402
from mtah.external import shape_sequence as sq  # noqa: E402
from mtah.external import shape_transfer as st  # noqa: E402
from mtah.features import team_shape  # noqa: E402
from mtah.methods import hmm_v2  # noqa: E402
from mtah.possession import shape_model  # noqa: E402
from mtah.validation import tagging  # noqa: E402

# Params
SEEDS = (0, 1, 2)
DFL_TRAIN = dfl.MATCHES[:5]
DFL_VALID = dfl.MATCHES[5]
DFL_TEST = dfl.MATCHES[6]
OUT = os.path.join(ROOT, "outputs", "seq_shape_bar_ath{}.npy")
BAR_KEYS = ["team acc", "D AUC", "D AUC gaps", "tag far", "no ball", "unexpl D", "phase D",
            "D share"]
DFL_KEYS = ["spellF1", "macroF1", "D_AUC"]


def bar_metrics(p, ctx):
    """The BAR-ATH columns of the module docstring for (n, 3) probabilities p.

    ctx holds the frame selections and labels (built in the main block). Frames without a
    prediction (NaN) are left out; a team-only opinion (P(D) all NaN) gets NaN in the P(D)
    columns.

        p = [[.8, .1, .1], [.3, .6, .1], [.1, .1, .8]], ctx["codes"] = [0, 0, 2]
        bar_metrics(p, ctx)["team acc"]          # -> 0.5: two team frames, both H; P(home)
                                                 #    0.89 and 0.33, so only the first is right
    """
    p = np.asarray(p, dtype=np.float64)
    home = p[:, 0] / (p[:, 0] + p[:, 1])
    team = ((ctx["codes"] == 0) | (ctx["codes"] == 1)) & np.isfinite(home)
    out = {"team acc": float(((home >= 0.5) == (ctx["codes"] == 0))[team].mean())}
    calls = team_call(home)
    out["tag far"] = tag_agreement(calls, ctx["tagged"], ctx["far"])[1]
    out["no ball"] = tag_agreement(calls, ctx["tagged"], ctx["far"] & ~ctx["present"])[1]
    d = p[:, 2]
    if np.isnan(d[ctx["graded"]]).all():
        return {**out, **{k: np.nan for k in BAR_KEYS if k not in out}}
    out["D AUC"] = auc(ctx["dead_labels"], d, ctx["graded"])
    out["D AUC gaps"] = auc(ctx["dead_labels"], d, ctx["graded"] & ctx["in_gap"])
    out["unexpl D"] = float(np.nanmean(d[ctx["unexplained"]]))
    out["phase D"] = float(np.nanmean(d[ctx["graded"] & (ctx["tagged"] != "")]))
    out["D share"] = float(np.nanmean(d[ctx["graded"]]))
    return out


def line(name, row, keys):
    return "{:<22}".format(name) + "".join("{:>11.3f}".format(row[k]) for k in keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="1 seed, 2 blocks, 2 epochs")
    args = parser.parse_args()
    seeds, n_folds = SEEDS, shape_model.N_BLOCKS
    if args.quick:
        sq.MAX_EPOCHS, sq.PATIENCE, sq.FT_EPOCHS, seeds, n_folds = 2, 1, 1, SEEDS[:1], 2

    # ------------------------------------------------------------- our match
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    table = team_shape.compute(match)[st.COLUMNS]
    events, p_dead, dead_labels, p_home, far, _ = hmm_v2(version=7).evidence(match)
    codes = sq.confident_codes(shape_model.pseudo_labels(match, events), dead_labels)
    codes[~graded] = sq.NO_LABEL
    block = shape_model.blocks(graded)
    whole = sq.segments(graded)
    gap_list = explain.explain_gaps(match, features.compute(match), match_whistles())
    in_gap = np.zeros(match.n_frames, dtype=bool)
    for g in gap_list:
        in_gap[g["start"]:g["stop"]] = True
    ctx = dict(codes=codes, dead_labels=dead_labels, graded=graded, tagged=tagging.phase_team(match),
               far=far, present=np.asarray(match.ball_present), in_gap=in_gap,
               unexplained=explain.unexplained_frames(match.n_frames, gap_list) & graded)
    print("graded frames {}: H {}, A {}, D {}, alive (team unknown) {}, unlabelled {}".format(
        int(graded.sum()), *[int((codes[graded] == c).sum()) for c in (0, 1, 2, sq.ALIVE_ONLY,
                                                                         sq.NO_LABEL)]),
          flush=True)

    # ------------------------------------------------------------- DFL
    gaps = adapter.presence_gaps(match)
    dfl_data = {m: st.build(m, gaps, seed=k) + (dfl.load(m).half,)
                for k, m in enumerate(dfl.MATCHES)}
    train = [dfl_data[m] for m in DFL_TRAIN]
    tune_dfl = tuple(list(x) for x in zip(*[dfl_data[m] for m in DFL_TRAIN + (DFL_VALID,)]))
    test_table, test_labels, test_half = dfl_data[DFL_TEST]

    bar, dfl_rows = {}, {}
    for kind in sq.KINDS:
        pre_bar, ft_bar, pre_dfl, ft_dfl = [], [], [], []
        for seed in seeds:
            pre = sq.fit([d[0] for d in train], [d[1] for d in train], [d[2] for d in train],
                         kind, seed=seed, valid=dfl_data[DFL_VALID])
            pre_bar.append(sq.predict(pre, table, whole))
            pre_dfl.append(sq.predict(pre, test_table, test_half))
            oof = np.full_like(pre_bar[-1], np.nan)
            for b in range(n_folds):
                tuned = sq.fine_tune(pre, table, sq.segments(graded & (block != b)), codes,
                                     tune_dfl, seed=seed)
                p = sq.predict(tuned, table, whole)
                oof[block == b] = p[block == b]
                ft_dfl.append(all_scores(test_labels, sq.predict(tuned, test_table, test_half),
                                         test_half))
            ft_bar.append(oof)
            print("{} seed {}: pretrained best epoch {}, fine-tuned {} blocks".format(
                kind, seed, pre.best_epoch, n_folds), flush=True)
        bar[kind + " pretrained"] = np.mean(pre_bar, axis=0)
        bar[kind + " fine-tuned"] = np.mean(ft_bar, axis=0)
        dfl_rows[kind + " pretrained"] = all_scores(test_labels, np.mean(pre_dfl, axis=0),
                                                    test_half)
        dfl_rows[kind + " fine-tuned"] = {k: np.mean([r[k] for r in ft_dfl]) for k in DFL_KEYS}
    for stage in ("pretrained", "fine-tuned"):
        bar["avg " + stage] = (bar["tcn " + stage] + bar["lstm " + stage]) / 2
    if not args.quick:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        np.save(OUT.format(""), bar["avg fine-tuned"].astype(np.float32))
        np.save(OUT.format("_pretrained"), bar["avg pretrained"].astype(np.float32))

    # ------------------------------------------------------------- comparison
    bar["DFL boosting"] = st.for_match(match)
    current = np.full((match.n_frames, 3), np.nan)
    current[:, 0], current[:, 1] = p_home, 1 - p_home
    rows = {name: bar_metrics(p, ctx) for name, p in bar.items()}
    detector = np.full((match.n_frames, 3), np.nan)
    detector[:, 2] = p_dead
    team_part, dead_part = bar_metrics(current, ctx), bar_metrics(detector, ctx)
    rows["current (hmm_v7)"] = {k: team_part[k] if k in ("team acc", "tag far", "no ball")
                                else dead_part[k] for k in BAR_KEYS}

    print("\nBAR-ATH (fine-tuned: out of fold over {} time blocks)".format(n_folds))
    print("{:<22}".format("method") + "".join("{:>11}".format(k) for k in BAR_KEYS))
    for name in ["tcn pretrained", "tcn fine-tuned", "lstm pretrained", "lstm fine-tuned",
                 "avg pretrained", "avg fine-tuned", "DFL boosting", "current (hmm_v7)"]:
        print(line(name, rows[name], BAR_KEYS))

    print("\nDFL held-out match {} (forgetting)".format(DFL_TEST))
    print("{:<22}".format("method") + "".join("{:>11}".format(k) for k in DFL_KEYS))
    for name, row in dfl_rows.items():
        print(line(name, row, DFL_KEYS))
