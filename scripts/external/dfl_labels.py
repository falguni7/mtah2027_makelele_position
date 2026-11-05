"""Controlled-possession labels for the 7 DFL matches: choose T, then profile every rule.

    python scripts/external/dfl_labels.py

Everything here is about other, public matches, so all numbers can be looked at freely.

1. Sync. Per match, the clock shift that best lines the logged passes up with the tracking's
   owner (dfl_events.best_shift), the agreement it reaches, and, as an independent check, the
   median and spread of restart events against BallStatus dead -> alive AFTER the shift (should
   be near 0).
2. Choosing T. For T from 0.5 to 5 s, the spellF1 of the duration rule (a change needs T
   seconds) scored against the events rule (a change needs a controlled action), per match and
   on average. The T where the two agree best is the one used for the OR rule: chosen on DFL
   alone, as agreed (never by matching our public 5 minutes).
3. Profile per rule (raw, duration, events, or, and) at that T, averaged over the matches:
   possessions per match, mean and median length, share under 2 s, and spellF1 against the OR
   label. The last shows how much the choice of label could move a benchmark score.

Nothing is written except the event-clock cache (cache/external/dfl/<match>_clock.json).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.external import controlled, dfl, dfl_events  # noqa: E402
from mtah.utils import runs_of  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
GRID = np.arange(0.5, 5.01, 0.25)


def spell_lengths(labels, half):
    """Lengths in seconds of the possession spells (maximal H or A runs inside a half).

    Spells as the official scorer counts them (possession_metrics.possession_spells): a D frame
    ends a spell, and a spell never crosses half-time.

        labels = H H D A A A (25 fps, one half)
        spell_lengths(labels, half)         # -> array([0.08, 0.12])
    """
    runs = [r for h in (1, 2) for r in runs_of(np.asarray(half) == h)]
    spells = pm.possession_spells(pm.label_codes(labels), runs)
    return np.array([b - a for a, b, *_ in spells]) / FPS


if __name__ == '__main__':
    data = []
    print("1. sync: shift (frames), agreement with the owner, restarts vs BallStatus after the shift")
    for match_id in dfl.MATCHES:
        m = dfl.load(match_id)
        events = dfl_events.load(m)
        off = dfl_events.sync_offsets(m, events)
        print("   {}  shift {:+4d}  agreement {:.3f}  restarts: median {:+.0f}, IQR {:.0f} to {:.0f} "
              "frames ({} restarts)".format(match_id, events.attrs["shift"], events.attrs["agreement"],
                                            np.median(off), *np.percentile(off, [25, 75]), len(off)))
        data.append((m, events, controlled.labels(m, events, rule="events")))

    print("\n2. spellF1 of the duration rule against the events rule, by T")
    table = np.array([[controlled.spell_f1(ev_labels, controlled.labels(m, events, "duration", t),
                                           m.half) for m, events, ev_labels in data] for t in GRID])
    for t, row in zip(GRID, table):
        print("   T {:4.2f} s  mean {:.3f}  per match {}".format(t, row.mean(),
                                                               " ".join("{:.3f}".format(v) for v in row)))
    best = float(GRID[table.mean(axis=1).argmax()])
    print("   best T = {:.2f} s".format(best))

    print("\n3. profile per rule at T = {:.2f} s (mean over matches; min-max in brackets)".format(best))
    print("   {:<9}{:>16}{:>16}{:>16}{:>16}{:>18}".format(
        "rule", "possessions", "mean s", "median s", "under 2 s", "spellF1 vs or"))
    for rule in controlled.RULES:
        stats = []
        for m, events, _ in data:
            lab = controlled.labels(m, events, rule, best)
            main = controlled.labels(m, events, "or", best)
            lengths = spell_lengths(lab, m.half)
            stats.append((len(lengths), lengths.mean(), np.median(lengths), (lengths < 2).mean(),
                          controlled.spell_f1(main, lab, m.half)))
        stats = np.array(stats)
        print("   {:<9}".format(rule) + "".join(
            "{:>7.{p}f} [{:.{p}f}-{:.{p}f}]".format(c.mean(), c.min(), c.max(), p=0 if k == 0 else 2)
            .rjust(16 if k < 4 else 18) for k, c in enumerate(stats.T)))
