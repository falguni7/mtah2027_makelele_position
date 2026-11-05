"""Regenerate the submission file from the provided match data, with one command.

    python scripts/run_pipeline.py                         # everything from the raw data
    python scripts/run_pipeline.py --reuse-cache           # skip steps 1-2 if already done
    sbatch scripts/slurm/run.sbatch scripts/run_pipeline.py

This is the command the submission rules ask for: someone with the data bundle in data/ (see
data/README.md) and this repository gets the same file, byte for byte. It runs, in order:

  1. scripts/data/build_cache.py     parse the tracking, homography and Smart Tagging into the
                                     numpy cache (cache/, about a minute)
  2. scripts/data/extract_audio.py   decode the tactical video's sound into a mono 24 kHz WAV
                                     in the cache, for the referee-whistle detector (about a
                                     minute on a compute node, as ffmpeg reads the whole video)
  3. scripts/make_submission.py      run the submitted method (SUBMISSION_METHOD, hmm_v7) on the
                                     full match, write the frame,label CSV and check it with the
                                     official evaluation/validate_submission.py

and finally prints the file's SHA-256, so two runs can be compared at a glance. Each step is run
as its own Python process, exactly as it runs on its own, and the pipeline stops at the first
step that fails (exit code of that step). Nothing is random: the classifiers have a fixed
random_state and the masking test is not part of it.

By default steps 1 and 2 are always rerun, so the file comes from the raw data and not from an
older cache. With --reuse-cache they are skipped when their output already exists (the cache's
meta.json, the WAV file), when only the code has changed.

Output: submissions/BAR-ATH_possession.csv by default (gitignored, like all full-match
labels), or --out. Runtime 3-4 minutes on an Alex compute node (A100 node, 16 cores), most of
it step 3; tested from an empty cache (MTAH_CACHE_DIR pointing at a new folder), giving the
same SHA-256 as the file made step by step.

    python scripts/run_pipeline.py --reuse-cache --out /tmp/try.csv
    # == 1/3 build cache: skipped, cache/BAR-ATH_tactical exists
    # == 2/3 extract audio: skipped, cache/BAR-ATH_tactical_audio/audio_24k.wav exists
    # == 3/3 hmm_v7 submission
    # ... validator output ...
    # /tmp/try.csv  sha256 3f2a...
"""

import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.audio import AUDIO_PATH  # noqa: E402
from mtah.data.cache import cache_dir  # noqa: E402

# Params
SUBMISSION_METHOD = "hmm_v7"
DEFAULT_OUT = os.path.join(ROOT, "submissions", "BAR-ATH_possession.csv")


def sha256(path):
    """Hex SHA-256 of a file's bytes: equal for two files exactly when their contents are equal.

    Read in 1 MB chunks, so the file never has to fit in memory at once.

        open("a.txt", "w").write("frame,label\\n")
        sha256("a.txt")        # -> "c770ccb0...c145" (64 hex digits), the same for every
                               #    file with exactly these bytes
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step(label, script, args=()):
    """Run one of the repository's scripts as its own Python process; stop the pipeline if it
    fails.

    label is printed as the step's header, script is a path relative to the repository root,
    args are extra command-line arguments. The script runs with the same Python interpreter as
    this one, from the repository root; its output goes straight to the terminal (or the Slurm
    log). On a non-zero exit code the whole pipeline exits with that code.

        run_step("1/3 build cache", "scripts/data/build_cache.py")
        # == 1/3 build cache
        # parsing ATD ... (the script's own output)
    """
    print("== " + label, flush=True)
    code = subprocess.call([sys.executable, "-W", "ignore", os.path.join(ROOT, script), *args],
                           cwd=ROOT)
    if code != 0:
        print("step '{}' failed with exit code {}".format(label, code), flush=True)
        sys.exit(code)


def skip_or_run(label, done_path, reuse, script):
    """Run a data-preparation step, unless reuse is set and its output done_path exists.

        skip_or_run("2/3 extract audio", AUDIO_PATH, reuse=True, script=...)
        # == 2/3 extract audio: skipped, cache/BAR-ATH_tactical_audio/audio_24k.wav exists
        skip_or_run("2/3 extract audio", AUDIO_PATH, reuse=False, script=...)
        # == 2/3 extract audio   (then the script runs)
    """
    if reuse and os.path.exists(done_path):
        print("== {}: skipped, {} exists".format(label, os.path.relpath(done_path, ROOT)),
              flush=True)
    else:
        run_step(label, script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="submission CSV to write")
    parser.add_argument("--reuse-cache", action="store_true",
                        help="skip building the cache and extracting the sound when they exist")
    args = parser.parse_args()

    skip_or_run("1/3 build cache", os.path.join(cache_dir(), "meta.json"), args.reuse_cache,
                "scripts/data/build_cache.py")
    skip_or_run("2/3 extract audio", AUDIO_PATH, args.reuse_cache,
                "scripts/data/extract_audio.py")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run_step("3/3 {} submission".format(SUBMISSION_METHOD), "scripts/make_submission.py",
             [SUBMISSION_METHOD, "--out", out])
    print("{}  sha256 {}".format(out, sha256(out)), flush=True)
