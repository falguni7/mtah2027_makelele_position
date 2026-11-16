# Ball possession in BAR-ATH without a single human label of the match

**More than a Hack 2027, Phase 1: Ball Possession — technical document**

Team: Makelele_position · Repository: https://github.com/falguni7/mtah2027_makelele_position ·
Submitted method: `hmm_v19_bridge`

---

## 1. Summary

The final model is one hidden Markov model (HMM) over H / A / D, run twice on our match, a standard run (`hmm_v11_mix`)
and a stricter one that pins fewer frames as alive (`hmm_v18_doubtB`); the submission keeps the
standard run's labels and takes the stricter run's dead ball only where it joins two of the
standard run's stoppages. Both runs read the same per-frame evidence: control events and a
team-shape model and dead-ball detector trained on our match without human labels, plus a
frame-by-frame boosting model and a TCN + BiLSTM sequence model trained on seven labelled matches
of other games (the public DFL Bundesliga open data).

Every frame is labelled H / A / D by a hidden Markov model (HMM), decoded with Viterbi per half.
Its evidence comes from four sources: a **dead-ball
detector** trained on rule-based confident frames, **control events** from the tracking, a
**team-shape model** trained on this match's control events, and two **models trained on public
labelled matches of other games** (DFL Bundesliga open data). The core, `hmm_v11_mix`, averages
two HMM variants (one for each DFL model) and decodes once; the submission, `hmm_v19_bridge`, also
joins two of the core's stoppages where a stricter second run says the short stretch between them
is dead (4 stretches, 0.4 % of graded frames). It reports 287 possessions of mean 13.1 s (H 34.7 %,
A 24.4 %, D 40.9 % of graded frames). Every choice was made with label-free checks on our match and
benchmarks on other matches with labels (section 7). `python scripts/run_pipeline.py` regenerates
the file byte for byte.

## 2. The problem as we saw it

* The ball is tracked on only **59.3 %** of the 158,857 graded frames (150 gaps, median 10.7 s,
  longest 3.7 min). The public window has it on 88 %, so it is not representative.
* **A missing ball is not a dead ball:** many gaps are open play where the tracker lost the ball.
* `spellF1` rewards whole possessions: the raw baseline reports 131 possessions in 5 minutes
  against 17.
* Track slots are tracklets with a fixed team; the ball track is heavily smoothed (touches blur).

The two hard parts are therefore **telling stoppages from open play inside ball gaps** and
**naming the team where the ball is not seen**, while keeping the output in whole possessions.

## 3. Approach

### 3.0 Why this design

* **Two stages: per-frame evidence, then one sequence decision.** Each source (dead-ball detector,
  control events, shape models) answers a narrow question frame by frame; the HMM combines them
  over time. Each part can be checked on its own, and a weak source (team shape in a long gap) only
  nudges the result instead of overriding it.
* **Why a sequence model at all.** Possession persists (13 s on average), but per-frame
  classifiers flicker: the raw baseline reports 131 possessions in 5 minutes against 17, and
  `spellF1` punishes that. The sequence step is where "possession lasts" and our possession rules
  (a change only at a control event) are built in.
* **Why an HMM, not a learned sequence model (CRF, RNN) on our match.** There are no labels of
  BAR-ATH to train one with. An HMM needs only a handful of costs, which we set from arguments and
  can inspect; Viterbi finds the exact best sequence, deterministically and quickly; pinned frames
  and rule costs are easy to add.
* **Why gradient boosting on our match, not deep learning.** Everything learned on BAR-ATH learns
  from pseudo-labels of one match: tens of thousands of strongly correlated frames described by a
  few dozen hand-made features. Boosted trees do well on such small tabular data, handle missing
  values (absent players, no ball) natively, train quickly on a CPU and are reproducible. A
  deep network needs many more independent examples and fits the pseudo-labels' errors more easily.
* **Where we did use deep learning.** On the DFL data, with real labels on about 1 M frames, a TCN +
  BiLSTM over 30 s of team shape beat boosting (macroF1 0.878 vs 0.835), so the time context helps
  there; its output enters our match only as evidence (section 3.4). A deep ball detector
  fine-tuned on our video improved no check and was dropped (section 6).

### 3.1 Dead-ball detector (`mtah/dead_ball/`)

**Confident labels** from rules: *dead* inside a GOALS tag until the kick-off, the first 3 s of a
SET PIECES tag, the second half before its kick-off, the ball more than 1 m out for 0.5 s; *alive*
the ball tracked for 3 s, faster than 2.5 m/s, more than 2 m inside and away from tags, short
open-play ball gaps, and sustained running (team speed over 3 s above 2 m/s, more than 12 % of
players above 4 m/s). Alive examples inside gaps stop the classifier from learning "ball missing =
dead".

**Classifier:** gradient boosting on features the rules do not use: the players' speeds, spreads
and positions, the ball-gap context (time since / until the ball is seen, where), the **tactical
camera's turning speed** from the homography (it rests in stoppages; AUC 0.89 inside gaps), and
the DFL models' P(dead) (section 3.4).

**Referee whistles** (`mtah/audio/whistle.py`): two tones at ~3.8 + ~4.2 kHz rising together in
the stadium sound (70 events). A whistle does not say whether play stops or restarts, so it only
makes entering or leaving D cheaper next to it. **Unexplained gaps** (≥ 3 s, no whistle, no ball
near a line, no tag; 14 % of gap time) get their odds of dead divided by 4.

### 3.2 Control events (`mtah/possession/control.py`)

A frame is **in control** for the nearest player's team when the ball is within **1.25 m**, moves
with them (relative speed below **3 m/s**) and the nearest opponent is at least **0.5 m** farther;
a **control event** is at least 5 such frames (0.2 s). Thresholds come from measured distributions,
not labels. 852 events; single-frame ball spikes above 40 m/s are removed first.

### 3.3 Team-shape model (`mtah/possession/shape_model.py`)

Who has the ball, from the players alone: per team and over 1 s, advance, front and defensive
line, length, width, forward velocity, pressing speed, hull, and where the camera points (42
columns). Trained on control-event pseudo-labels (33,360 frames), out of fold over 10 time
blocks: accuracy 0.880 against 0.644 for the majority class.

### 3.4 Models trained on other matches (`mtah/external/`)

The DFL open data has per-frame `BallPossession` and `BallStatus`. Its per-touch possession is
converted to *controlled* possession (a run of the other team counts if it makes a controlled
action — pass, shot, ball claimed, duel won — or lasts 3.75 s; ≈ 252 possessions a match), and
its tracking is degraded to our player drop-outs, with the ball hidden. Two models predict H / A /
D from players only: a **boosting model** (leave-one-match-out macroF1 0.835) and a **TCN +
BiLSTM sequence model** over 30 s windows (macroF1 0.878), the latter fine-tuned on our match's
confident frames.

### 3.5 Temporal model (`mtah/temporal/hmm.py`)

**HMM and Viterbi in brief.** A hidden Markov model treats the true label of each frame (H, A or
D) as a hidden state we never see directly. Each frame's *evidence* gives every state a cost (a
negative log-probability), and *switching* state costs a fixed amount while staying is free. The
**Viterbi algorithm** finds, by dynamic programming in one pass over the frames, the single label
sequence of a half with the lowest total cost. A short burst of contrary evidence cannot pay for
two switches, so the output comes in whole possessions rather than flickering frame by frame. As
we label a recorded match offline, evidence after a frame counts as much as evidence before it.

Per-frame costs: 0.1 × log P(dead) (below 1 because 25 near-identical frames a second are not 25
independent observations), confident dead frames pinned; during a control event, the
other team and D pay 5 per frame; between events a team other than the last one pays 0.02 per
frame, so changes land at the start of the new team's event (after a stoppage, the restart
taker's team); far from events, 0.05 × log P(team). H ↔ A costs 2; entering or leaving D costs 4,
or 1 next to a whistle. All values were argued from first principles, not tuned.

### 3.6 Core ensemble: `hmm_v11_mix`

Two variants differ only in the DFL model feeding them (boosting or sequence). Rather than pick
one on noisy checks, both inputs are averaged half and half in log-odds (two dead-ball detectors,
two team opinions) and decoded once; the 50/50 weight was fixed in advance.

### 3.7 Submission: joining stoppages (`hmm_v19_bridge`)

Switches into possession on a ball nobody played (section 6) pointed to stoppages missed inside
tracked play. Their cause is the ball rule for confident alive frames: after a foul the ball
often rolls on faster than 2.5 m/s while the players stop, and those frames were pinned alive.
`hmm_v18_doubtB` leaves such frames unknown when the players walk (below 1.5 m/s) or nobody is
within 3 m of the ball. On its own it adds many stoppages in open play, so `hmm_v19_bridge` keeps
`hmm_v11_mix` and accepts `hmm_v18_doubtB`'s D only where it fills a whole alive stretch between
two of `hmm_v11_mix`'s own stoppages.

## 4. Possession rules we implemented

Possession changes when the opponent gains *controlled* possession (CHALLENGE.md).

| situation | our rule |
|---|---|
| controlled possession | ball within 1.25 m of a player, moving with them, for at least 0.2 s |
| loose ball, pass, ball in flight | previous team kept until another control event; passes between team-mates never change it |
| deflection, block, touch | shorter than 0.2 s or not moving with the player: no change |
| challenge / duel | no control while an opponent is about as close; the winner gets it when they control the ball |
| when a change happens | at the **start** of the new team's control event |
| dead ball | D from the detector and the HMM; ball out, goals, set-piece starts and half-time pinned |
| restarts | possession goes to the team of the first control event after the stoppage |
| ball not seen | team from the carry rule and the shape models; D only where the dead-ball evidence says so |
| X | never used: confidences are not calibrated well enough to gain from it |

## 5. Missing and noisy tracking

| problem | handling |
|---|---|
| ball missing on 41 % of graded frames | dead-ball detector with gap context and in-gap alive examples; unexplained-gap softening; team from ball-free shape models |
| ball spikes, smoothed ball track | spikes removed; changes timed by control, not ball acceleration (uninformative here) |
| tracklets, identity switches, ≈ 20.5 of 22 players | team aggregates only; DFL training data degraded to our drop-outs |
| tracking collapse at the end of half 1 | the HMM carries the state; camera and whistles do not depend on tracking |

## 6. Assets: used, tried and dropped

| asset | status | how |
|---|---|---|
| tactical tracking | **used** | players and ball, all steps |
| tactical homography | **used** | camera turning speed and aim; geometry |
| Smart Tagging | **used, partly** | GOALS and SET PIECES as confident dead labels. In-possession phases are **never an input**: they are our main independent check |
| tactical video, sound | **used** | referee whistles |
| DFL open data (7 matches) | **used** | training both shape models; benchmark |
| tactical video, images | dropped | fine-tuned ball detector (below) |
| broadcast video and tracking | dropped | camera cuts and replays as dead-ball evidence (below) |
| Metrica (3 matches) | benchmark only | as extra training data (`hmm_v12_mix`) no change |
| SkillCorner | dropped | 10 fps broadcast tracking, unclear possession semantics |

**Built and not adopted** (in `mtah/methods.py`; each failed an adoption rule):

* **X abstention** (`hmm_v8`): posteriors poorly calibrated.
* **HMM weights tuned on DFL**: better on DFL, worse on every check on our match.
* **Own ball detection** (`hmm_v13_wasb`, `hmm_v14_gap`): WASB-SBDT fine-tuned on our video with
  the tracking ball as pseudo-label (held-out hits 18 % → 42 %); correcting or filling the tracking
  ball improved no check.
* **Smart Tagging phases as input** (`hmm_v15_tag`): no clear gain, and it would cost our validation check.
* **Broadcast cuts and replays**: AUC 0.91 alone, nothing added to the classifier (the director
  cuts away when the players stop, which their speed already shows).
* **No touch, no change of possession** (`hmm_v16_touch`): the smoothed ball shows no clear jump at
  switches, so it cannot place them. But a change into possession needs a touch and a change into
  D does not, so a switch into possession on a ball nobody played points to a missed stoppage.
  Penalising such switches only moved them; it led to section 3.7.
* **Wider stoppage fixes** (`hmm_v18_doubtB`, `hmm_v20_adjacent`): better on DFL (spellF1 up to
  0.546 vs 0.531), but Smart Tagging agreement fell (to 0.654 and 0.683).

## 7. Validation without labels of our match

Hand labels of BAR-ATH are forbidden and the public 5 minutes may not be used for tuning, so each
variant had an adoption rule fixed before its results: Smart Tagging agreement must not fall,
the masking test must not fall by more than one standard deviation, and external benchmarks must
not get worse.

* **Smart Tagging agreement:** share of in-possession-phase frames (63,718) where a method names
  the tagged team. Automatic and coarse, so consistency, not accuracy; never an input.
* **Masking test** (`mtah/validation/masking.py`): hide the ball in blocks with our real gap
  lengths, rerun, score against the method's own unmasked output: robustness to a missing ball.
* **Physical checks:** a held-out restart check (detector trained without set-piece labels, share
  of D 10–20 s after a set-piece tag should be low); stoppage boundaries near 39 % of whistles
  (4 × chance) without using them; quiet switches (section 6); possession counts against DFL and
  Metrica (230–280 a match).
* **Benchmarks with labels:** the whole pipeline on the 7 DFL matches degraded to look like ours
  (40.7 % of the ball hidden), and the team side on 3 Metrica matches with the true dead ball.
* **Public 5 minutes:** pipeline and format checks; one look after choosing gave spellF1 0.522
  and macroF1 0.719 (equal for `hmm_v11_mix`) against 0.326 / 0.621 for `baseline_smooth50`.
  With 17 possessions, differences below ~0.3 are noise. Used only to check the pipeline, never for choosing or training.

| check | baseline_smooth50 | control rule, no HMM | hmm_v11_mix | **hmm_v19_bridge** |
|---|---|---|---|---|
| possessions, mean length | 594, 7.2 s | 253, 14.5 s | 291, 13.0 s | **287, 13.1 s** |
| Smart Tagging: agree / other / D | 0.621 / 0.261 / 0.118 | 0.684 / 0.137 / 0.179 | 0.692 / 0.142 / 0.167 | **0.692 / 0.139 / 0.169** |
| masking spellF1 (sd) / macroF1 | 0.645 (0.029) / 0.559 | 0.635 (0.069) / 0.746 | 0.775 (0.012) / 0.856 | **0.777 (0.013) / 0.856** |
| restart check, D 10–20 s after a set piece | | | 0.245 | **0.245** |
| DFL, 7 matches: spellF1 / macroF1 | | | 0.531 / 0.799 | **0.535 / 0.803** |
| Metrica team side: spellF1 | | | 0.731 | **0.731** |

## 8. Training data

No human label of BAR-ATH was used for anything. Everything learned on BAR-ATH learns from
algorithmic pseudo-labels (rule-based confident frames, control events, the projected tracking
ball).

| dataset | source, licence | labelled by | amount | used for |
|---|---|---|---|---|
| **DFL Bundesliga open data** | DFL, CC BY 4.0 (Bassek et al., 2025, *Scientific Data*) | DFL's official position data (BallPossession, BallStatus) and event data collectors | 7 matches, 2022/23, ≈ 1 M frames at 25 Hz | both shape models (**in the submission**); benchmark |
| Metrica sample data | Metrica Sports, public sample, no licence stated | Metrica's human event data | 3 matches | benchmark only |
| WASB-SBDT soccer weights | nttcom/WASB-SBDT, MIT | the authors' soccer set | — | dropped ball detector |

## 9. Limitations and what we would do with more time

* **Dead ball inside ball gaps** is the main uncertainty; without labels we cannot tell how much
  of the gap time we call D is real.
* **Short stoppages inside tracked play are missed**: after a foul the ball often keeps rolling
  while the players stop, so those frames are kept alive, and switches into possession on a ball
  nobody played (section 6) point to such stoppages. Only the safest fix is submitted; removing the moving-ball pins everywhere also called
  slow build-up play dead.
* **Team evidence in long gaps is moderate**, and **all validation is indirect**: Smart Tagging is
  automatic, masking rewards stability, benchmarks are other leagues with simulated noise.
* **Confidences are not calibrated**, so we never output X. The sequence model is trained on GPU;
  on the same GPU type a retrain gave identical labels.

With more time we would build:

* a **restart-type classifier** (throw-in, corner, goal kick, free kick, kick-off) from player
  formations, trained on DFL set pieces, to show where stoppages end and who restarts;
* **touch spotting** from the ball's speed and direction changes, trained on DFL pass and shot
  times, to place possession switches;
* a **video touch spotter** for the ball gaps, taught by the tracking-based one where the ball is
  tracked;
* a **self-training round of the ball detector**, to find the ball in more than 10 % of gap frames;
* **calibrated confidences**, so the method can output X where it is unsure;
* an **audit of the player tracks**, repairing team switches with jersey-colour re-identification;
* **goalkeeper possession** as its own case instead of an unexplained gap.
* **action spotting** trained on public video datasets of other games (such as SoccerNet), with
  the spotted passes, receptions and restarts turned into possession labels.

## 10. Reproduction

Needs the data bundle in `data/` (see `data/README.md`), ffmpeg, and internet once for the DFL data.

```bash
conda create -n mtah2027 python=3.11.12 ffmpeg=7.1.1 -c conda-forge
conda activate mtah2027
pip install -r requirements-lock.txt
bash scripts/data/download_external.sh --dfl-only      # once, 2.5 GB
python scripts/run_pipeline.py
```

`run_pipeline.py` builds the cache from the raw data, extracts the sound, runs `hmm_v19_bridge`,
writes `submissions/BAR-ATH_possession.csv`, validates it with `evaluation/validate_submission.py`
and prints its SHA-256. About 25 minutes on a 16-core node with an NVIDIA A40
(the sequence model trains on the GPU), with CUDA 13.0 (PyTorch 2.11.0+cu130, cuDNN 9.19, NVIDIA
driver 615.71.10). Tested on 2026-11-15 from a fresh clone (commit 157f7ad)
with an empty cache: byte-identical to the submitted file, SHA-256
`d96dfdffe9c3e374f4bb0b06b901e1d2b31e3833115d403cb34bfad20e582089`.

## 11. Compliance with the data terms

No match data, cache or per-frame output is in the repository or its history. No hand labels of
BAR-ATH, no use of the public 5 minutes for training or choosing, no frame-specific rules.

**Use of a hosted LLM.** We used Claude (Anthropic), a hosted LLM, to help write code and
documentation. The match data never left our workspace. The assistant was never given the videos,
images, sound, tracking, Smart Tagging, homography or any per-frame output. It saw the code,
aggregates, the public challenge documentation, and the public data of other games (DFL, Metrica,
SkillCorner).
