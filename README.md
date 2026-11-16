# Makelele_position — our solution to More than a Hack 2027, Phase 1

This is the solution repository of team **Makelele_position** for the *More than a Hack 2027,
Phase 1: Ball Possession* challenge (match BAR-ATH). Everything from
[Original challenge README](#original-challenge-readme-from-the-organisers) onwards is the
organisers' text, unchanged.

| What | Where |
|---|---|
| Team name | Makelele_position |
| Submitted method | `hmm_v19_bridge` in [mtah/methods.py](mtah/methods.py) |
| Submission file | made by `python scripts/run_pipeline.py` as `submissions/BAR-ATH_possession.csv` (SHA-256 `d96dfdff…`); sent on the form as `Makelele_position.csv`. Not committed: it contains match labels |
| Technical document | [docs/technical_document.md](docs/technical_document.md); sent on the form as `Makelele_position.pdf` |
| Our code | [mtah/](mtah/) (the method), [scripts/](scripts/) (pipeline, checks, cluster jobs) |
| Organisers' files | the challenge docs (`*.md` at the top level), [evaluation/](evaluation/), [starter/](starter/), `games.py` |
| Match data | not included (licensed for the competition only); put it in `data/` as in [data/README.md](data/README.md) |

## Reproducing our submission

Put the data bundle in `data/` as described in [data/README.md](data/README.md), install the
environment, then run one command:

```bash
conda create -n mtah2027 python=3.11.12 ffmpeg=7.1.1 -c conda-forge
conda activate mtah2027
pip install -r requirements-lock.txt
python scripts/run_pipeline.py
```

**⚠ WARNING: `scripts/run_pipeline.py` downloads external data. On its first run it fetches the
DFL Bundesliga open data (about 2.5 GB, from figshare) into `external/dfl/` if it is not there
already.**

It builds the numpy cache from the tracking, homography and Smart Tagging, extracts the
tactical video's sound (for the referee-whistle detector), runs our method `hmm_v19_bridge` and
checks the result with the official validator. The file is `submissions/BAR-ATH_possession.csv`,
and its SHA-256 is printed at the end. Nothing is random, so every run gives the same file. It
takes about 25 minutes on a 16-core node, plus the one-off download of the DFL data below. See
`scripts/run_pipeline.py` for the steps.

**Hardware used for the submitted file:** one NVIDIA A40 GPU with CUDA 13.0 (PyTorch
2.11.0+cu130, cuDNN 9.19, NVIDIA driver 615.71.10). On this setup a fresh clone reproduces the
submitted file byte for byte (SHA-256 `d96dfdffe9c3e374f4bb0b06b901e1d2b31e3833115d403cb34bfad20e582089`).

The method also learns from public labelled matches of other games: the DFL Bundesliga open data
(CC BY 4.0), about 2.5 GB in `external/dfl/`. The pipeline downloads it when it is missing; on a
machine whose compute nodes have no internet access, fetch it first with

```bash
bash scripts/data/download_external.sh --dfl-only
```

(without `--dfl-only` it also fetches the Metrica and SkillCorner sample data, ~6.2 GB in all).

The submission does not need WASB-SBDT. Only the ball-detection variants we tried and dropped
(`hmm_v13_wasb`, `hmm_v14_gap`, and `scripts/vision/`) use it. It is not part of this repository:
we synced its code (nttcom/WASB-SBDT, MIT) and its soccer weights into `third_party/WASB-SBDT`
with

```bash
git clone https://github.com/nttcom/WASB-SBDT.git third_party/WASB-SBDT
git -C third_party/WASB-SBDT checkout 923462cacdeb3353b84ddebdedb3f4b7a8553b0f
mkdir -p third_party/WASB-SBDT/pretrained_weights
gdown 1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l -O third_party/WASB-SBDT/pretrained_weights/wasb_soccer_best.pth.tar
```

(`pip install gdown`; the weights' Google Drive id is the soccer WASB entry of the repository's
`MODEL_ZOO.md`, SHA-256 `d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c`).

---

# Original challenge README (from the organisers)

> The text below is the challenge README from the organisers' starter kit
> ([metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027)),
> kept as it was.

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
