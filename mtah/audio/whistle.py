"""Find the referee's whistle in the stadium sound, by its two simultaneous tones.

    from mtah import audio
    from mtah.audio import whistle
    tone = whistle.pair_strength(whistle.rise(audio.load()))   # per 10 ms: how whistle-like
    events = whistle.detect(tone)          # [(start_s, stop_s, low_hz, strength_db)]
    events = whistle.match_whistles()      # the same for the whole match, computed once

How a referee's whistle behaves: elite referees blow the Molten Valkeen (standard for FIFA and
UEFA referees), a pealess whistle that sounds TWO tones at the same time, specified at about
3.67 + 4.15 kHz (or 3.74 + 4.10 kHz on some specifications). Almost all its energy sits in those
two narrow pitches, it holds them steadily, it lasts from a short blast (~0.2 s, a foul) to a
long one (~1 s, kick-off, end of a half), and it is loud on the pitch microphones. On this match
the kick-off whistles indeed show two tones together, at 3.77-3.91 kHz and 4.22-4.43 kHz, the
higher one 1.10-1.13 times the lower one. So this detector looks for exactly that: a PAIR of
tones rising at the same moment, at that spacing. Few other sounds do that. Crowd noise and
cheers are spread over all pitches; chants are mostly below 1 kHz; a crowd's whistling is many
tones at scattered pitches; a single-tone whistle or horn has no partner at the right spacing.

Two steps lead there:

  * how far each pitch is above ITS OWN usual level (rise()): the power of every pitch against
    its median over the surrounding BACKGROUND_S seconds. This matters on this match because
    the sound track carries a constant tone at ~1.85 kHz, there all the time (also at
    half-time): measured against its own usual level it never stands out, while a whistle
    rises by tens of dB within a second;
  * the pair (pair_strength()): in every 10 ms step, the pitch that rises most in a LOW band
    (PAIR_LOW, 3.55-4.05 kHz) and the one that rises most in a HIGH band (PAIR_HIGH,
    4.05-4.55 kHz). If the high one is PAIR_RATIO (1.06-1.16) times the low one, the strength
    is the WEAKER of the two rises, minus how much the whole 1.8-5 kHz band rises at that moment
    (a cheer lifts every pitch, a whistle only its own two), so both tones must stand out.

detect() then keeps the stretches where that strength stays high, at a steady pitch, for long
enough. The bands come from the whistle's specification and the kick-off whistles of this
match, and the threshold from how the strength is distributed over the whole match; see
scripts/audio/whistle_check.py and the work log. No labels are used.

In this file, in pipeline order: band_power() and background() give each pitch's power and
usual level, rise() compares the two, pair_strength() turns that into one whistle-likeness
value per 10 ms, detect() turns it into whistle events, events_to_frames() marks the video
frames each event covers, and match_whistles() runs it all once on the match's sound.
"""

from functools import lru_cache

import numpy as np
from scipy.ndimage import median_filter

from mtah.audio import AUDIO_PATH, SAMPLE_RATE, load
from mtah.utils import runs_of

# Params
FPS = 25
N_FFT = 1024                # samples per spectrum, 43 ms at 24 kHz (23 Hz per frequency bin)
HOP = SAMPLE_RATE // 100    # one spectrum every 10 ms
STEPS_PER_S = SAMPLE_RATE // HOP
BAND = (1800.0, 5000.0)     # Hz, the band whose overall rise is subtracted (cheers, crowd noise)
BACKGROUND_S = 15           # seconds either side used for each pitch's usual level
PAIR_LOW = (3550.0, 4050.0)     # Hz, the lower tone of a dual-tone referee whistle
PAIR_HIGH = (4050.0, 4550.0)    # Hz, the higher tone
PAIR_RATIO = (1.06, 1.16)       # high tone / low tone
MIN_STRENGTH = 15.0         # dB: weaker steps are not whistle-like
MAX_JUMP = 150.0            # Hz the low tone may move between consecutive 10 ms steps
MAX_DROP = 3                # steps (30 ms) a whistle may dip below the threshold and continue
MIN_DURATION = 0.15         # seconds; shorter blips are not whistles
CHUNK = 6000                # spectra computed per batch (60 s of sound), to bound memory


def band_power(samples):
    """(power, freqs): the power spectrum between BAND[0] and BAND[1] for every 10 ms step.

    power is (steps, bins) float32, freqs the (bins,) frequency of each column in Hz (23.4 Hz
    apart). Step i covers N_FFT samples (43 ms) from i * 10 ms on, with a Hann window; the
    sound is processed CHUNK steps at a time so memory stays bounded.

        power, freqs = band_power(one second of a 3,000 Hz tone)
        power.shape                               # -> (96, 137)
        freqs[power[50].argmax()]                 # -> ~3000.0
    """
    samples = np.asarray(samples, dtype=np.float32)
    window = np.hanning(N_FFT).astype(np.float32)
    all_freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    band = (all_freqs >= BAND[0]) & (all_freqs <= BAND[1])
    n_steps = max(0, (len(samples) - N_FFT) // HOP + 1)
    power = np.zeros((n_steps, int(band.sum())), dtype=np.float32)
    for first in range(0, n_steps, CHUNK):
        steps = np.arange(first, min(first + CHUNK, n_steps))
        index = steps[:, None] * HOP + np.arange(N_FFT)[None, :]
        power[steps] = np.abs(np.fft.rfft(samples[index] * window, axis=1)[:, band]) ** 2
    return power, all_freqs[band]


def background(power):
    """Each pitch's usual level around each moment: (steps, bins), same units as power.

    The median power of every frequency bin over +/- BACKGROUND_S seconds. To keep it fast, the
    median is first taken per second, then a running median over the seconds; every 10 ms step
    gets the value of its second. A tone that is always there (like the ~1.85 kHz hum on this
    match) has a background equal to itself, so it does not rise above it; a whistle lasting
    a second or two barely moves a median over 30 s.

        power of one bin: 1.0 everywhere, 100.0 for 0.5 s in the middle of 60 s
        background(power) for that bin           # -> 1.0 everywhere, the blast included
    """
    n_steps, n_bins = power.shape
    n_seconds = -(-n_steps // STEPS_PER_S)
    padded = np.full((n_seconds * STEPS_PER_S, n_bins), np.nan, dtype=np.float32)
    padded[:n_steps] = power
    per_second = np.nanmedian(padded.reshape(n_seconds, STEPS_PER_S, n_bins), axis=1)
    smooth = median_filter(per_second, size=(2 * BACKGROUND_S + 1, 1), mode="nearest")
    return np.repeat(smooth, STEPS_PER_S, axis=0)[:n_steps]


def rise(samples):
    """(rise_db, freqs, power): how far every pitch is above its usual level, per 10 ms step.

    rise_db is (steps, bins): 10 * log10(power / background(power)) for every pitch in BAND,
    0 where undefined (silence). freqs is the pitch of each column, power the band_power(). The
    input of pair_strength().

        a constant 1,850 Hz tone                  # -> rise ~0 dB at 1,850 Hz all the time
        a 3,800 Hz blast at 30 s                  # -> rise ~+50 dB at 3,800 Hz during the blast
    """
    power, freqs = band_power(samples)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10 * np.log10(power / background(power))
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32), freqs, power


def pair_strength(r):
    """(strength_db, low_hz, level_db, high_hz) per 10 ms step: how much it sounds like a whistle.

    r is rise()'s output. In each step, the pitch that rises most in PAIR_LOW (3.55-4.05 kHz)
    and the one that rises most in PAIR_HIGH (4.05-4.55 kHz) are found. If the high one is
    PAIR_RATIO (1.06-1.16) times the low one, strength_db is the WEAKER of their two rises
    minus the median rise of the whole band (so both tones must stand out, and a sound that
    lifts every pitch does not count); otherwise it is 0. low_hz and high_hz are the two
    pitches, level_db the power of the low one (dB on the int16 scale), to tell a loud whistle
    from a faint one. detect() follows the low tone to decide whether the whistle holds its
    pitch.

    Toy example: 60 s of noise plus a constant 1,850 Hz tone, with a whistle at 30-30.5 s.

        strength_db, low_hz, level_db, high_hz = pair_strength(rise(samples))
        # whistle of 3,850 + 4,250 Hz (ratio 1.10):
        low_hz[3010], high_hz[3010]               # -> ~3844, ~4242
        strength_db[3010]                         # -> e.g. 40 dB   both tones rise
        strength_db[1000]                         # -> e.g. 6 dB    tone and noise only
        # a single 3,850 Hz tone instead          # -> e.g. 7 dB    no partner tone
        # 3,850 + 4,500 Hz (ratio 1.17)           # -> 0            wrong spacing
    """
    rise_db, freqs, power = r
    rows = np.arange(len(rise_db))
    median = np.median(rise_db, axis=1)
    low = np.flatnonzero((freqs >= PAIR_LOW[0]) & (freqs < PAIR_LOW[1]))
    high = np.flatnonzero((freqs >= PAIR_HIGH[0]) & (freqs < PAIR_HIGH[1]))
    low_peak = low[rise_db[:, low].argmax(axis=1)]
    high_peak = high[rise_db[:, high].argmax(axis=1)]
    ratio = freqs[high_peak] / freqs[low_peak]
    spaced = (ratio >= PAIR_RATIO[0]) & (ratio <= PAIR_RATIO[1])
    weaker = np.minimum(rise_db[rows, low_peak], rise_db[rows, high_peak])
    strength = np.where(spaced, weaker - median, 0.0).astype(np.float32)
    with np.errstate(divide="ignore"):
        level = np.where(power[rows, low_peak] > 0, 10 * np.log10(power[rows, low_peak]), 0.0)
    return (strength, freqs[low_peak].astype(np.float32), level.astype(np.float32),
            freqs[high_peak].astype(np.float32))


def detect(tone, min_strength=MIN_STRENGTH, min_duration=MIN_DURATION):
    """Whistle events [(start_s, stop_s, low_hz, strength_db)] from pair_strength()'s output.

    tone is the pair_strength() tuple: its first value is the strength per 10 ms step, its
    second the low tone's pitch. A step is whistle-like when its strength is at least
    min_strength (15 dB). Consecutive whistle-like steps form one event as long as the low tone
    moves by at most MAX_JUMP (150 Hz) per step; gaps of up to MAX_DROP steps (30 ms) are
    bridged when the pitch on both sides matches, as a whistle can dip for an instant. Events
    shorter than min_duration (0.15 s) are dropped. Each event gives its start and end in
    seconds on the video timeline, the median pitch of its low tone and its median strength.

        strength_db: 5 5 30 31 29 5 30 30 ... (35 steps at ~30 dB) ... 5
        low_hz:          3800 3820 3810   3805 ...
        detect(tone)
        # -> [(0.02, 0.42, 3810.0, 30.0)]   one blast of 0.40 s, low tone ~3.8 kHz; the
        #    one-step dip at 0.05 s is bridged because the pitch is the same on both sides
    """
    strength, freq = tone[0], tone[1]
    loud = strength >= min_strength
    runs = []
    for start, stop in runs_of(loud):
        # split a run where the pitch jumps, e.g. two different sounds back to back
        jumps = np.flatnonzero(np.abs(np.diff(freq[start:stop])) > MAX_JUMP) + 1
        edges = [start] + list(start + jumps) + [stop]
        runs += [(a, b) for a, b in zip(edges[:-1], edges[1:])]
    merged = []
    for start, stop in runs:
        if merged:
            last_start, last_stop = merged[-1]
            if (start - last_stop <= MAX_DROP
                    and abs(np.median(freq[last_start:last_stop])
                            - np.median(freq[start:stop])) <= MAX_JUMP):
                merged[-1] = (last_start, stop)
                continue
        merged.append((start, stop))
    step = HOP / SAMPLE_RATE
    return [(start * step, stop * step, float(np.median(freq[start:stop][loud[start:stop]])),
             float(np.median(strength[start:stop][loud[start:stop]])))
            for start, stop in merged if (stop - start) * step >= min_duration]


def events_to_frames(events, n_frames):
    """Bool per video frame: True on frames that a whistle event overlaps.

    Frame f covers [f / 25, (f + 1) / 25) seconds.

        events = [(1.00, 1.10, 3800.0, 30.0)]     # a whistle from 1.00 s to 1.10 s
        events_to_frames(events, 30)[24:28]       # -> [False, True, True, True]  frames 25-27
    """
    out = np.zeros(n_frames, dtype=bool)
    for start, stop, _, _ in events:
        out[int(np.floor(start * FPS)):int(np.ceil(stop * FPS))] = True
    return out


@lru_cache(maxsize=None)
def match_whistles(path=AUDIO_PATH):
    """The match's whistle events, detected once per Python process and then reused.

    The whole chain on the extracted sound: load(), rise(), pair_strength(), detect() with the
    default thresholds. That takes a minute or two, and the methods call it on every run (the
    masking test runs a method dozens of times on masked copies of the match, which never
    change the sound), so the result is kept for the rest of the process. Returns a tuple of
    (start_s, stop_s, low_hz, strength_db) events, the same as detect().

        events = match_whistles()             # ~1-2 min the first time
        events = match_whistles()             # instant afterwards
        len(events)                           # -> 70 on this match
    """
    return tuple(detect(pair_strength(rise(load(path)))))
