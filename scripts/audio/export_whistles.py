"""Write the detected whistles to a CSV file, one row per whistle, in time order.

    python scripts/audio/export_whistles.py
    sbatch scripts/slurm/run.sbatch scripts/audio/export_whistles.py

Needs the sound extracted first (scripts/data/extract_audio.py). Detects the referee's
dual-tone whistles with mtah/audio/whistle.py (events >= whistle.MIN_STRENGTH dB) and writes
them to outputs/whistles.csv, which is gitignored and stays on the cluster. The script itself
prints only how many events it wrote.

Columns:

  start_frame, stop_frame   video frames the event covers (half-open; frame = time x 25)
  start_time, stop_time     the same as a video timestamp, H:MM:SS.ss
  duration_s                length of the event
  low_hz, high_hz           median pitch of the whistle's two tones
  strength_db               the weaker tone's rise above its usual level, beyond the rest of
                            the band (higher = clearer)
  period                    "1st half", "2nd half", "half-time", "before" or "after" the match
  near                      Smart Tagging restarts or goals starting within 30 s after the
                            event (type and seconds after the whistle), or kick-off / end of
                            half moments within 5 s

    # what a row looks like (illustrative values):
    start_frame,stop_frame,start_time,stop_time,duration_s,low_hz,high_hz,strength_db,period,near
    11147,11171,0:07:25.88,0:07:26.83,0.95,3891,4289,27.5,1st half,kick-off half 1 (-1.5 s)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah import audio  # noqa: E402
from mtah.audio import whistle  # noqa: E402
from mtah.data import load  # noqa: E402
from mtah.dead_ball.detector import KICKOFF_2ND  # noqa: E402
from mtah.utils import runs_of  # noqa: E402

# Params
FPS = 25
TAG_WINDOW = 30.0                   # seconds after a whistle in which a restart tag is listed
MOMENT_WINDOW = 5.0                 # seconds either side for kick-offs and ends of halves
OUT_PATH = os.path.join(ROOT, "outputs", "whistles.csv")


def timestamp(seconds):
    """Seconds on the video timeline as H:MM:SS.ss.

        timestamp(445.72)        # -> "0:07:25.72"
        timestamp(3725.5)        # -> "1:02:05.50"
    """
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return "{:d}:{:02d}:{:05.2f}".format(int(hours), int(minutes), secs)


def period_of(frame, half_runs):
    """Which part of the video a frame is in: "1st half", "2nd half", "half-time", "before",
    "after". half_runs are the two graded (start, stop) frame runs.

        half_runs = [(100, 200), (300, 400)]
        [period_of(f, half_runs) for f in (50, 150, 250, 350, 450)]
        # -> ["before", "1st half", "half-time", "2nd half", "after"]
    """
    (a1, b1), (a2, b2) = half_runs
    if frame < a1:
        return "before"
    if frame < b1:
        return "1st half"
    if frame < a2:
        return "half-time"
    if frame < b2:
        return "2nd half"
    return "after"


if __name__ == '__main__':
    match = load(mmap=False)
    half_runs = runs_of(np.asarray(match.graded))
    tags = match.smart_tagging
    restarts = [(row.start_s, row.type if row.code == "SET PIECES" else "goal")
                for row in tags.itertuples() if row.code in ("SET PIECES", "GOALS")]
    goals_end = tags.loc[tags["code"] == "GOALS", "end_s"].to_numpy()
    moments = [("kick-off half 1", 11185 / FPS), ("kick-off half 2", KICKOFF_2ND / FPS)]
    moments += [("kick-off after goal {}".format(i + 1), end - 1.0)
                for i, end in enumerate(goals_end)]
    moments += [("end of half {}".format(i + 1), stop / FPS)
                for i, (_, stop) in enumerate(half_runs)]

    tone = whistle.pair_strength(whistle.rise(audio.load()))
    high_hz = tone[3]
    rows = []
    for start, stop, low, strength in whistle.detect(tone):
        near = ["{} ({:+.1f} s)".format(name, t - start) for name, t in moments
                if abs(t - start) <= MOMENT_WINDOW]
        near += ["{} (+{:.1f} s)".format(kind, t - start) for t, kind in restarts
                 if 0 <= t - start <= TAG_WINDOW]
        steps = slice(int(round(start * whistle.STEPS_PER_S)),
                      int(round(stop * whistle.STEPS_PER_S)))
        rows.append({
            "start_frame": int(np.floor(start * FPS)), "stop_frame": int(np.ceil(stop * FPS)),
            "start_time": timestamp(start), "stop_time": timestamp(stop),
            "duration_s": round(stop - start, 2), "low_hz": round(low),
            "high_hz": round(float(np.median(high_hz[steps]))),
            "strength_db": round(strength, 1),
            "period": period_of(int(start * FPS), half_runs),
            "near": "; ".join(near),
        })
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print("{}: {} whistle events".format(OUT_PATH, len(rows)))
