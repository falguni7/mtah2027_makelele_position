"""Extract the tactical video's sound track into the cache, as one mono WAV file.

    python scripts/data/extract_audio.py
    sbatch scripts/slurm/run.sbatch scripts/data/extract_audio.py

The tactical video carries the stadium sound (AAC, 48 kHz, stereo, as long as the video), and
the referee's whistle is in it. To look for whistles we only need the sound as plain numbers,
so ffmpeg decodes it once, mixes the two channels into one and resamples to SAMPLE_RATE
(24,000 samples per second, enough for everything up to 12 kHz, well above a whistle's tone).

The result is cache/<game>_tactical_audio/audio_24k.wav, 16-bit, about 380 MB. Like the rest
of the cache it is derived from the match data, so it is gitignored and never leaves the
cluster. Sample i is at time i / SAMPLE_RATE seconds on the video timeline (the audio and video
streams both start at 0), so video frame f is at sample f * SAMPLE_RATE / 25.

Takes a few minutes, as the whole video file is read. Prints the output path and its length.
"""

import os
import subprocess
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.audio import AUDIO_PATH, SAMPLE_RATE  # noqa: E402
from games import asset_paths  # noqa: E402

FFMPEG = os.path.join(os.path.dirname(sys.executable), "ffmpeg")

if __name__ == '__main__':
    video = asset_paths()["video"]
    os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", video, "-vn", "-ac", "1",
                    "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", AUDIO_PATH], check=True)
    with wave.open(AUDIO_PATH) as handle:
        seconds = handle.getnframes() / handle.getframerate()
    print("{}: {:.1f} s at {} Hz".format(AUDIO_PATH, seconds, SAMPLE_RATE))
