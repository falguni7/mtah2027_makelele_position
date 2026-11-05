"""Can we hear the referee's whistle, and does it line up with the tags and our dead balls?

    python scripts/audio/whistle_check.py
    sbatch scripts/slurm/run.sbatch scripts/audio/whistle_check.py

Needs the sound extracted first (scripts/data/extract_audio.py). Prints aggregates only: no
sound, no frames, nothing written to disk.

The whistle detector (mtah/audio/whistle.py) looks for the referee's dual-tone whistle: every
10 ms, a low tone (3.55-4.05 kHz) and a high tone (4.05-4.55 kHz), 1.06-1.16 times apart, that
both rise above their own usual level of the surrounding 30 s, beyond what the rest of the
1.8-5 kHz band does at the same moment. "Strength" is the weaker of the two rises, in dB, and
an event is a stretch where it stays high at a steady pitch. This script checks it in four
steps:

1. The measure over the whole match: percentiles of the strength, and at which pitch the
   strong moments have their low tone. A referee whistle should show up as a pile-up there.
2. Certain whistles: the two half kick-offs, the four kick-offs after goals (each GOALS tag
   ends 1 s after its kick-off) and the end of each half (the first frame after its last graded one). For each:
   the highest strength within +/- ANCHOR_WINDOW seconds, how far from the moment it is, at
   which two pitches, and the events detect() finds there (start relative to the moment,
   duration).
3. Detected events at several thresholds: how many, how long, at what pitch, and how often per
   minute during play, at half-time and before / after the match. Half-time has no refereeing,
   so events there are false alarms (before and after the match there can be warm-up and
   ceremony whistles).
4. Agreement with the Smart Tagging restarts (by type: free kick, throw-in, corner, goal kick,
   kick-off), the goals, and the step-1 dead-ball detector's stoppages: the share that have a
   whistle event in a window around them, against the same window at random graded moments
   (chance). A fit is a share well above chance. Referees whistle for fouls, offside,
   kick-offs and the end of a half, but usually not when the ball simply goes out for a
   throw-in, corner or goal kick, so those may well show little.

How to read the output: in part 2, a whistle has a strength far above the match's typical
values (part 1 percentiles). In part 4, "share" is the share of tags (or stoppages) with a
whistle event in the window, "chance" the same at random moments, "lift" their ratio, and
"offset" the median time from the tag to the nearest whistle event in the window.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import audio, dead_ball, features  # noqa: E402
from mtah.audio import whistle  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball.detector import KICKOFF_2ND  # noqa: E402
from mtah.utils import runs_of  # noqa: E402

# Params
FPS = 25
STEPS_PER_S = whistle.STEPS_PER_S   # the whistle measure gives one value per 10 ms
ANCHOR_WINDOW = 5.0                 # seconds either side of a certain whistle
THRESHOLDS = (10.0, 15.0, 20.0, 25.0)
AGREEMENT_THRESHOLDS = (10.0, 15.0, 20.0)
N_RANDOM = 3000
# (seconds before, seconds after) the tag or moment in which a whistle event counts
WINDOWS = {"Free Kick": (-30, 3), "Throw-in": (-30, 3), "Corner Kick": (-30, 3),
           "Goal Kick": (-30, 3), "Kick Off (Start)": (-10, 10),
           "Kick Off (After Goal)": (-10, 10), "goal": (-10, 20),
           "stoppage start": (-3, 3), "restart": (-3, 2)}
MIN_STOPPAGE = 3 * FPS              # dead-ball stretches at least 3 s long


def hits(moments_s, event_starts_s, before, after):
    """(share, median offset) of moments with an event starting in [moment + before, + after].

    moments_s and event_starts_s are seconds on the video timeline. The offset is from each
    moment to the nearest event start in its window (negative = the event came first), median
    over the moments that have one; NaN if none do.

        moments_s = [10.0, 50.0], event_starts_s = [8.0, 100.0]
        hits(moments_s, event_starts_s, -3, 3)     # -> (0.5, -2.0)   only the first has one
    """
    starts = np.sort(np.asarray(event_starts_s, dtype=np.float64))
    offsets = []
    for m in moments_s:
        inside = starts[(starts >= m + before) & (starts <= m + after)] - m
        if len(inside):
            offsets.append(inside[np.argmin(np.abs(inside))])
    share = len(offsets) / len(moments_s) if len(moments_s) else np.nan
    return share, float(np.median(offsets)) if offsets else np.nan


if __name__ == '__main__':
    match = load(mmap=False)
    graded = np.asarray(match.graded)
    samples = audio.load()
    tone = whistle.pair_strength(whistle.rise(samples))
    strength, low_hz, _, high_hz = tone
    print("sound: {:.1f} s, {} steps of 10 ms".format(len(samples) / audio.SAMPLE_RATE,
                                                      len(strength)))

    # ---------------------------------------------------------------- 1. the measure
    step_frame = (np.arange(len(strength)) * FPS // STEPS_PER_S).clip(0, match.n_frames - 1)
    in_play = graded[step_frame]
    print("\n1. strength (dB) percentiles p50/p90/p99/p99.9/max, graded frames: {}".format(
        np.percentile(strength[in_play], [50, 90, 99, 99.9, 100]).round(1)))
    for t in (15.0, 20.0):
        strong = in_play & (strength >= t)
        counts, edges = np.histogram(low_hz[strong], bins=np.arange(3400, 4201, 200))
        print("   steps >= {:.0f} dB: {} ({:.3%}); low tone (Hz: share) {}".format(
            t, int(strong.sum()), strong.mean(), "  ".join(
                "{:.0f}-{:.0f}: {:.2f}".format(edges[i], edges[i + 1],
                                               counts[i] / max(strong.sum(), 1))
                for i in range(len(counts)))))

    # ---------------------------------------------------------------- 2. certain whistles
    tags = match.smart_tagging
    goals = tags[tags["code"] == "GOALS"]
    half_runs = runs_of(graded)
    anchors = [("kick-off half 1", 11185 / FPS), ("kick-off half 2", KICKOFF_2ND / FPS)]
    anchors += [("kick-off after goal {}".format(i + 1), end - 1.0)
                for i, end in enumerate(goals["end_s"])]
    anchors += [("end of half {}".format(i + 1), stop / FPS)
                for i, (_, stop) in enumerate(half_runs)]
    events = whistle.detect(tone)
    print("\n2. certain whistles: highest strength within +/- {:.0f} s, and events >= {:.0f} "
          "dB there".format(ANCHOR_WINDOW, whistle.MIN_STRENGTH))
    for name, t in anchors:
        lo = max(0, int((t - ANCHOR_WINDOW) * STEPS_PER_S))
        hi = int((t + ANCHOR_WINDOW) * STEPS_PER_S)
        i = lo + int(np.argmax(strength[lo:hi]))
        near = [e for e in events if t - ANCHOR_WINDOW <= e[0] <= t + ANCHOR_WINDOW]
        print("   {:<22} max {:5.1f} dB at {:+5.1f} s, {:4.0f} + {:4.0f} Hz; events {} {}".format(
            name, strength[i], i / STEPS_PER_S - t, low_hz[i], high_hz[i], len(near),
            " ".join("[{:+.1f} s, {:.2f} s]".format(e[0] - t, e[1] - e[0]) for e in near)))

    # ---------------------------------------------------------------- 3. detected events
    halftime = np.zeros(match.n_frames, dtype=bool)
    halftime[half_runs[0][1]:half_runs[1][0]] = True
    outside = ~graded & ~halftime

    def per_minute(events):
        """Events per minute in play / at half-time / before and after the match, as text.

        events are whistle events (start_s, ...); each counts where its start falls. Each rate
        is the count in that part divided by the part's length in minutes. Half-time has no
        refereeing, so its rate is the false-alarm rate of the detector.

            # 30 events in 90 min of play, 3 in 15 min of half-time, 2 in 20 min outside
            per_minute(events)                   # -> "0.33 / 0.20 / 0.10"
        """
        frames = np.array([min(int(e[0] * FPS), match.n_frames - 1) for e in events], dtype=int)
        return " / ".join("{:.2f}".format(np.isin(frames, np.flatnonzero(m)).sum()
                                          / (m.sum() / FPS / 60))
                          for m in (graded, halftime, outside))

    print("\n3. detected events by threshold (events per minute: play / half-time / outside)")
    detected = {}
    for t in THRESHOLDS:
        events = whistle.detect(tone, min_strength=t)
        detected[t] = events
        durations = np.array([e[1] - e[0] for e in events])
        pitches = np.array([e[2] for e in events])
        print("   >= {:.0f} dB: {:>4} events, per minute {}; duration p50/p90 {} s; "
              "low tone p10/p50/p90 {} Hz".format(
                  t, len(events), per_minute(events),
                  np.percentile(durations, [50, 90]).round(2) if len(events) else "-",
                  np.percentile(pitches, [10, 50, 90]).round(0) if len(events) else "-"))

    # ---------------------------------------------------------------- 4. agreement
    is_dead, _, _ = dead_ball.detect(match, features.compute(match))
    stoppages = [(a, b) for a, b in runs_of(is_dead & graded) if b - a >= MIN_STOPPAGE]
    pieces = tags[tags["code"] == "SET PIECES"]
    groups = {kind: pieces.loc[pieces["type"] == kind, "start_s"].to_numpy()
              for kind in pieces["type"].dropna().unique()}
    groups["goal"] = goals["start_s"].to_numpy()
    groups["stoppage start"] = np.array([a for a, _ in stoppages]) / FPS
    groups["restart"] = np.array([b for _, b in stoppages]) / FPS
    rng = np.random.default_rng(0)
    random_s = rng.choice(np.flatnonzero(graded), N_RANDOM, replace=False) / FPS
    for t in AGREEMENT_THRESHOLDS:
        starts = [e[0] for e in detected[t]]
        print("\n4. agreement, events >= {:.0f} dB ({} events)".format(t, len(starts)))
        print("   {:<22} {:>4}  {:>11}  {:>6}  {:>6}  {:>5}  {:>8}".format(
            "moment", "n", "window s", "share", "chance", "lift", "offset s"))
        for name in sorted(groups, key=lambda k: list(WINDOWS).index(k) if k in WINDOWS else 99):
            before, after = WINDOWS.get(name, (-30, 3))
            share, offset = hits(groups[name], starts, before, after)
            chance, _ = hits(random_s, starts, before, after)
            row = "   {:<22} {:>4}  {:>+4d} to {:>+3d}  {:6.2f}  {:6.2f}  {:5.1f}  {:>+8.1f}"
            print(row.format(
                name, len(groups[name]), before, after, share, chance,
                share / chance if chance else np.nan, offset))
        frames = np.array([min(int(s * FPS), match.n_frames - 1) for s in starts], dtype=int)
        frames = frames[graded[frames]]
        print("   events in play: {} -- on frames the detector calls dead {:.2f} "
              "(dead share of graded frames {:.2f})".format(
                  len(frames), is_dead[frames].mean() if len(frames) else np.nan,
                  is_dead[graded].mean()))
