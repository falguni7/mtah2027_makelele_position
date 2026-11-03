"""Check the cache against independent facts about the match. Prints aggregates only.

    python scripts/data/check_cache.py
    sbatch scripts/slurm/run.sbatch scripts/data/check_cache.py

Most checks compare the cache with something it was NOT built from: published frame counts and
kickoff frames, the Smart Tagging clock, the camera, and the official baseline submissions. A few
are internal consistency checks (players per frame, teams swap ends, homography valid, tags
inside the video). A failure means the parsing or the alignment is wrong, and every later
result would be too. Exit code 1 if any check fails. Run it after every build_cache.py.

Output: a first line with the cache size and team names, then one block per topic (frames and
halves, kickoffs, smart tagging, homography and camera, baseline reproduced from the cache), one
line per check:

      [ok] <what is checked, padded to 44 characters> <the measured value>
      [FAIL] ...                                      (the text starts two columns later)

The right-hand detail is the number the verdict was based on, e.g. "3 of 10 frames differ" for
a baseline file. Lines starting with "info:", indented by 7 spaces, are facts worth knowing,
not checks. The last line is "<n> checks, <m> failed". Nothing is written to disk.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from mtah.data import load  # noqa: E402
from mtah.possession.baseline import nearest_player_codes  # noqa: E402
from baseline import smooth, VARIANTS  # noqa: E402

# Params
N_GRADED = 158857                       # CHALLENGE.md / DATA.md
KICKOFFS = {1: 11185, 2: 113742}        # DATA.md / FAQ.md, video frames
IMAGE_SIZE = (1920, 1080)
KICKOFF_SEARCH = 25                     # look this far for a tracked ball at kickoff
CENTRE_SPOT_TOLERANCE = 5.0             # metres; the tracked ball is noisy
OWN_HALF_SHARE = 0.8                    # share of each team that must be in its own half
OWN_HALF_TOLERANCE = 1.0                # metres past the halfway line still counted as own half
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

results = []


def check(name, ok, detail):
    """Record one check's verdict and print it as one aligned line.

    `ok` is the verdict, `detail` the measured value that justifies it. The verdict is kept in
    the module-level `results` list, which main() counts at the end for the exit code.

        check("toy check", False, "3 of 10 frames differ")
        # prints   "  [FAIL] toy check   ...   3 of 10 frames differ"  (name padded to 44;
        #          after "[FAIL]" the name starts two columns later than after "[ok]")
        # results  -> [..., False]
    """
    results.append(ok)
    print("  [{}] {:<44} {}".format("ok" if ok else "FAIL", name, detail))


def project(matrices, x, y):
    """Normalised pitch -> pixels, one matrix per point.

    Maps pitch positions given as fractions of the pitch (x of the length, y of the width, both
    0..1) to image pixels (u, v) with each point's own 3 x 3 homography. Used to ask whether the
    tracked ball lies inside the camera picture.

        matrices = np.array([[[1920, 0, 0], [0, 1080, 0], [0, 0, 1]]])   # one point, a scaling
        project(matrices, np.array([0.5]), np.array([0.25]))       # -> ([960.], [270.])
    """
    denominator = matrices[:, 2, 0] * x + matrices[:, 2, 1] * y + matrices[:, 2, 2]
    return ((matrices[:, 0, 0] * x + matrices[:, 0, 1] * y + matrices[:, 0, 2]) / denominator,
            (matrices[:, 1, 0] * x + matrices[:, 1, 1] * y + matrices[:, 1, 2]) / denominator)


def inside_image(u, v):
    """True where the pixel (u, v) lies inside the 1920 x 1080 video frame.

    u counts pixels from the left edge, v from the top; the far edges themselves are outside.

        inside_image(np.array([100, 1920, -1]), np.array([100, 500, 10]))
        # -> [True, False, False]
    """
    return (u >= 0) & (u < IMAGE_SIZE[0]) & (v >= 0) & (v < IMAGE_SIZE[1])


def main():
    """Run every check on the cache, print them, and return the exit code.

    Loads the cache (memory-mapped), prints the topic blocks described at the top of this file
    and returns 1 if any check failed, 0 if all passed; the script exits with that code.
    """
    match = load()
    length, width = match.pitch
    print("cache: {} frames, teams H = {}, A = {}".format(
        match.n_frames, *match.team_names))

    print("\nframes and halves")
    check("graded frame count", int(match.graded.sum()) == N_GRADED,
          "{} (expected {})".format(int(match.graded.sum()), N_GRADED))
    players_per_frame = match.player_present.sum(axis=1)
    graded_mean = players_per_frame[match.graded].mean()
    check("players per graded frame", graded_mean > 18, "{:.1f} on average".format(graded_mean))
    # Not checks, just facts worth knowing: tracking does not switch off at the half boundaries.
    # People are tracked through halftime, and it collapses before the end of the first half.
    (_, _, first_end), (_, second_start, _) = match.meta["half_blocks"]
    sparse = match.graded & (players_per_frame < 5)
    print("       info: {:.1f} players per frame during halftime; {} graded frames have < 5 "
          "players, {} of them in the last 2000 of half 1".format(
              players_per_frame[first_end + 1:second_start].mean(), int(sparse.sum()),
              int(sparse[first_end - 1999:first_end + 1].sum())))
    ball_rate = match.ball_present[match.graded].mean()
    check("ball tracked on graded frames", 0.55 < ball_rate < 0.65,
          "{:.1%} (dataset notes: 59.3%)".format(ball_rate))

    print("\nkickoffs")
    for number, frame in KICKOFFS.items():
        window = np.arange(frame - KICKOFF_SEARCH, frame + KICKOFF_SEARCH + 1)
        tracked = window[match.ball_present[window]]
        if len(tracked):
            nearest = tracked[np.abs(tracked - frame).argmin()]
            spot = np.hypot(*(match.ball_xy[nearest] - (length / 2, width / 2)))
            check("half {} ball on the centre spot".format(number),
                  spot < CENTRE_SPOT_TOLERANCE,
                  "{:.1f} m away, {} frames from kickoff".format(spot, abs(nearest - frame)))
        else:
            check("half {} ball on the centre spot".format(number), False,
                  "no tracked ball within {} frames".format(KICKOFF_SEARCH))

        present = match.player_present[frame]
        x = match.players_xy[frame, :, 0]
        home = present & (match.team_of_slot == 0)
        away = present & (match.team_of_slot == 1)
        if match.home_attacks_right[number]:
            home_ok = x[home] <= length / 2 + OWN_HALF_TOLERANCE
            away_ok = x[away] >= length / 2 - OWN_HALF_TOLERANCE
        else:
            home_ok = x[home] >= length / 2 - OWN_HALF_TOLERANCE
            away_ok = x[away] <= length / 2 + OWN_HALF_TOLERANCE
        # Slots are tracklets, so a stray one (a misassigned person) is normal: require a clear
        # majority, which is what tells the playing direction apart.
        check("half {} both teams in their own half".format(number),
              home_ok.mean() >= OWN_HALF_SHARE and away_ok.mean() >= OWN_HALF_SHARE,
              "H {}/{}, A {}/{}  (home attacks {})".format(
                  int(home_ok.sum()), len(home_ok), int(away_ok.sum()), len(away_ok),
                  "right" if match.home_attacks_right[number] else "left"))
    check("teams swap ends at halftime",
          match.home_attacks_right[1] != match.home_attacks_right[2],
          "home attacks right: {}".format(match.home_attacks_right))

    print("\nsmart tagging")
    tags = match.smart_tagging
    kickoff_tags = tags[(tags["code"].str.upper() == "SET PIECES")
                        & (tags["type"].str.lower() == "kick off (start)")]
    for number, frame in KICKOFFS.items():
        hit = ((kickoff_tags["start_frame"] <= frame) & (frame <= kickoff_tags["end_frame"])).any()
        check("half {} kickoff inside a kick-off tag".format(number), bool(hit),
              "{} kick-off tags in total".format(len(kickoff_tags)))
    in_range = ((tags["start_frame"] >= 0) & (tags["end_frame"] <= match.n_frames)).all()
    check("all tags inside the video", bool(in_range),
          "{} instances, team H {} / A {} / none {}".format(
              len(tags), int((tags["team"] == "H").sum()), int((tags["team"] == "A").sum()),
              int(tags["team"].isna().sum())))

    print("\nhomography and camera")
    graded = np.flatnonzero(match.graded)
    check("homography valid on graded frames", bool(match.homography_valid[graded].all()),
          "{:.1%}".format(match.homography_valid[graded].mean()))
    with_ball = graded[match.ball_present[graded]]
    u, v = project(np.asarray(match.homography[with_ball]),
                   match.ball_xy[with_ball, 0] / length, match.ball_xy[with_ball, 1] / width)
    ball_in_view = inside_image(u, v).mean()
    check("tracked ball projects into the image", ball_in_view > 0.95,
          "{:.1%} of graded frames with a ball".format(ball_in_view))
    # Frame alignment of tracking to video is not testable from the camera's motion alone (the
    # operator pans after the ball, so a lag looks like a better fit). The baseline check below
    # shows the cache uses the same frame index as the official tooling.

    print("\nbaseline reproduced from the cache")
    codes = nearest_player_codes(match)
    for name, window in VARIANTS:
        path = os.path.join(RESULTS_DIR, name + ".csv")
        if not os.path.isfile(path):
            check(name, False, "official file missing")
            continue
        official = pd.read_csv(path)["label"].to_numpy()
        ours = smooth(codes, window)
        differ = int((official != ours).sum())
        check(name, differ == 0, "{} of {} frames differ".format(differ, len(official)))

    failed = results.count(False)
    print("\n{} checks, {} failed".format(len(results), failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
