"""Our methods on the 3 Metrica sample matches, scored against labels from their human events.

    sbatch --time=03:00:00 scripts/slurm/run.sbatch scripts/external/metrica_benchmark.py
    python scripts/external/metrica_benchmark.py --detector   # the detector's own dead ball

mtah/external/metrica_bench.py makes each Metrica match look like ours (players degraded with
our drop-out lengths, the ball also hidden in play on the same share of alive frames as in the
DFL benchmark, no sound, camera or Smart Tagging) and runs the core of our method on it in the
benchmark variants (mtah/external/benchmark.py):

  core          nothing from the DFL models                          (~ hmm_v7)
  deadA_boost   the boosting model's P(D) and team margin in the detector   (~ hmm_v9_deadA)
  A_seq         the sequence model's P(D) in the detector and its team evidence  (~ hmm_v10A)
  mix           the two averaged in log-odds                           (~ hmm_v11_mix)
  *_pool        the same three with the DFL models also trained on the OTHER two Metrica games
                (mtah/external/pool.py; mix_pool ~ hmm_v12_mix)

All at today's HMM weights (benchmark.DEFAULT): nothing is chosen here. By default the decoder
gets the TRUE dead ball (metrica_bench.true_dead): Metrica's ball vanishes at stoppages, so the
detector has almost nothing to learn from there, and this benchmark tests the team side. Then
core and deadA_boost are the same (their only difference is the detector's input), A_seq is the
sequence model's team evidence (hmm_v10A's) and mix the average (hmm_v11_mix's). --detector
keeps the detector's own dead ball (the first run, kept for the record). Each prediction is
scored against the OR labels (the main ones) and against the strict "and" labels
(metrica.labels), on the graded frames, with team_acc the share of truly alive frames given the
right team. Metrica's data is public, so per-game scores are
printed; they are also saved to cache/external/metrica/benchmark_v1.npz.

Each game is one worker process (both placements): the shape table (~4 min), the DFL models'
predictions (the sequence model fine-tuned per placement, ~7 min each on the GPU, cached) and
the evidence of each variant (the detector and our shape model train in the match).
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"     # 3 workers on a 16-core node; set before numpy loads

import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.external import benchmark as bm  # noqa: E402
from mtah.external import metrica  # noqa: E402
from mtah.external import metrica_bench as mb  # noqa: E402

# Params
VARIANTS = ("core", "deadA_boost", "A_seq", "mix", "deadA_boost_pool", "A_seq_pool", "mix_pool")
RULES = ("or", "and")
KEYS = ("spellF1", "macroF1", "possessions", "mean_s", "team_acc")
OUT = os.path.join(metrica.CACHE, "benchmark_v3{}.npz")      # "" true dead ball, "_detector"


def run_game(job):
    """Scores of every placement x variant x label rule on one game, and the detectors' D AUC.

    job is (game, ctx, shares, detector): detector False gives the decoder the true dead ball.
    Returns (game, scores of shape (placements, variants, rules, len(KEYS)), d_auc of shape
    (placements, variants)).

        run_game((1, ctx, {"random": 0.409, "dead": 0.336}, False))
        # -> (1, array (2, 7, 2, 5), array (2, 7))   placements, variants, rules, keys
    """
    game, ctx, shares, detector = job
    started = time.time()
    scores = np.zeros((len(mb.PLACEMENTS), len(VARIANTS), len(RULES), len(KEYS)))
    d_auc = np.zeros((len(mb.PLACEMENTS), len(VARIANTS)))
    labels = {rule: metrica.labels(metrica.load(game), rule) for rule in RULES}
    table = None
    for p, placement in enumerate(mb.PLACEMENTS):
        match, truth = mb.bench_match(game, ctx, placement, shares[placement])
        if table is None:                    # the players are the same for both placements
            table = mb.team_table(match)
        probs = {False: mb.model_probs(match, game, placement, ctx),
                 True: mb.model_probs(match, game, placement, ctx, pool=True)}
        for v, variant in enumerate(VARIANTS):
            pooled = bm.VARIANTS[variant].get("pool", False)
            ev = bm.evidence(match, truth, variant, team_table=table, probs=probs[pooled])
            d_auc[p, v] = ev["d_auc"]
            for r, rule in enumerate(RULES):
                t = bm.Truth(labels[rule], truth.half, truth.k, truth.match_id)
                pred = bm.decode(match, ev if detector else mb.true_dead(ev, t), bm.DEFAULT)
                row = dict(mb.score(t, pred), team_acc=mb.team_accuracy(t, pred))
                scores[p, v, r] = [row[k] for k in KEYS]
        print("game {} {:<6} done after {:.0f} s".format(game, placement, time.time() - started),
              flush=True)
    return game, scores, d_auc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", action="store_true",
                        help="the detector's own dead ball instead of the true one")
    args = parser.parse_args()
    ctx = bm.context()
    shares = mb.in_play_shares(ctx)
    print("ball hidden in play (share of alive frames, from the DFL benchmark): {}".format(
        {k: round(v, 3) for k, v in shares.items()}), flush=True)
    games = list(metrica.GAMES)
    with get_context("spawn").Pool(len(games)) as pool:   # spawn: each worker its own CUDA
        done = pool.map(run_game, [(g, ctx, shares, args.detector) for g in games], chunksize=1)
    scores = np.zeros((len(games), len(mb.PLACEMENTS), len(VARIANTS), len(RULES), len(KEYS)))
    d_auc = np.zeros((len(games), len(mb.PLACEMENTS), len(VARIANTS)))
    for game, s, a in done:
        scores[games.index(game)], d_auc[games.index(game)] = s, a
    os.makedirs(metrica.CACHE, exist_ok=True)
    np.savez(OUT.format("_detector" if args.detector else ""), scores=scores, d_auc=d_auc, games=games, placements=mb.PLACEMENTS,
             variants=VARIANTS, rules=RULES, keys=KEYS, shares=[shares[p] for p in mb.PLACEMENTS])

    print("\ntruth (OR labels): possessions per game / mean s")
    for game in games:
        lab = metrica.labels(metrica.load(game))
        g = np.isin(lab, ["H", "A", "D"])
        row = bm.score(lab[g], lab[g], metrica.load(game).half[g])
        print("   game {}  {:4d}  {:5.1f}".format(game, row["possessions"], row["mean_s"]))

    S, M = KEYS.index("spellF1"), KEYS.index("macroF1")
    for p, placement in enumerate(mb.PLACEMENTS):
        print("\n== ball placement: {} (in play {:.3f}), dead ball: {} ==  weights {}".format(
            placement, shares[placement], "detector" if args.detector else "true",
            bm.DEFAULT))
        print("{:<13}{:>22}{:>9}{:>9}{:>7}{:>8}{:>10}{:>8}{:>12}".format(
            "variant", "spellF1 per game", "spellF1", "macroF1", "poss", "mean s", "team acc",
            "D AUC", "spellF1 and"))
        for v, variant in enumerate(VARIANTS):
            per = scores[:, p, v, 0, S]
            mean = scores[:, p, v, 0].mean(axis=0)
            print("{:<13}{:>22}{:>9.3f}{:>9.3f}{:>7.0f}{:>8.1f}{:>10.3f}{:>8.3f}{:>12.3f}".format(
                variant, " ".join("{:.3f}".format(x) for x in per), mean[S], mean[M], mean[2],
                mean[3], mean[4], d_auc[:, p, v].mean(), scores[:, p, v, 1, S].mean()))
