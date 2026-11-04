"""The match's stadium sound, as extracted by scripts/data/extract_audio.py.

    from mtah import audio
    samples = audio.load()                   # int16 array, SAMPLE_RATE samples per second
    samples[audio.SAMPLE_RATE * 60]          # the sound 60 s into the tactical video

The sound comes from the tactical video, so its timeline is the video's: sample i is at
i / SAMPLE_RATE seconds, and video frame f starts at sample f * SAMPLE_RATE / 25. The referee's
whistle is found in it by whistle.py.
"""

import os
import wave

import numpy as np

from games import GAME
from mtah.data.cache import CACHE_DIR

# Params
SAMPLE_RATE = 24000
AUDIO_PATH = os.path.join(CACHE_DIR, "{}_tactical_audio".format(GAME), "audio_24k.wav")


def load(path=AUDIO_PATH):
    """The whole sound track as a 1-D int16 array (mono), read from the extracted WAV file.

    Raises FileNotFoundError with a hint if scripts/data/extract_audio.py has not been run, and
    ValueError if the file is not mono 16-bit at SAMPLE_RATE (e.g. extracted by older code).

        samples = load()
        len(samples) / SAMPLE_RATE           # -> e.g. 7972.7   seconds, the video's length
    """
    if not os.path.exists(path):
        raise FileNotFoundError("{} not found: run scripts/data/extract_audio.py".format(path))
    with wave.open(path) as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (
                1, 2, SAMPLE_RATE):
            raise ValueError("{}: expected mono 16-bit {} Hz".format(path, SAMPLE_RATE))
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
