"""H / A / D from the players alone, over time: a TCN and a bidirectional LSTM, trained on DFL.

    from mtah.external import shape_sequence as sq
    fitted = sq.fit(tables, labels, halves, kind="tcn", seed=0, valid=(table, lab, half))
    p = sq.predict(fitted, table, half)          # (n, 3): P(H), P(A), P(D) per 25 fps frame

The boosting shape model (mtah/external/shape_transfer.py) looks at one frame at a time: its
features are smoothed over +-0.5 s, but it cannot see that the away team was pressing ten
seconds ago, or that everybody stands still a few seconds later. These two sequence models
read 30 s of team shape at once and label every moment in it, so the context before AND after
a frame counts (we label whole matches offline, so looking ahead is allowed; whether the ball
went dead is often clearest from what happens next).

The inputs and labels are exactly the boosting model's: the 36 team-shape columns of
shape_transfer.COLUMNS (no ball, no camera) computed on the DFL players degraded like ours, and
the OR-rule controlled-possession labels, from shape_transfer.build(). Only the model differs,
so a comparison between them measures what the time context adds.

The pieces:

  steps      both datasets are 25 fps, and neighbouring frames are near-copies, so the models
             work on every 5th frame (STEP): 5 steps a second, a 30 s window is 150 steps
             (WINDOW). To label all 25 frames of a second, predict() runs the model on the 5
             interleaved step sequences (frames 0, 5, 10 ...; 1, 6, 11 ...; ...) and puts the
             answers back in place, so nothing is interpolated.
  scale      every column is standardised with the training matches' mean and standard
             deviation (standardiser()), computed on the frames AND their mirrors, so that
             mirroring and standardising can be done in either order. The networks cannot
             take a missing value (boosting can), so a NaN becomes the training mean.
  symmetry   as for boosting, "home" means nothing across matches. Training windows are
             mirrored at random (teams' columns swapped, differences negated, H and A swapped:
             mirror_index()), and predict() averages the answer for a sequence and its mirror,
             so swapping the teams swaps P(H) and P(A) exactly.
  models     TCN: a stack of residual dilated 1-D convolutions that look both ways in time
             (dilations 1, 2, ... 32: each step sees 127 steps, about 25 s). BiLSTM: two layers
             of LSTM units run forwards and backwards over the window. Both have ~100k
             parameters, so neither wins by size.
  training   random 30 s windows of the training halves, cross-entropy on every step, AdamW;
             after every epoch the loss on a separate validation match decides early stopping,
             and the best epoch's weights are kept. The match a model is tested on is never
             used for this.
  whole half predict_steps() slides the window over a whole half every 10 s (HOP) and averages
             the overlapping answers, weighting each by how far it is from its window's edge
             (an answer in the middle of a window has context on both sides).

  our match  fine_tune() trains a DFL model further on our confident frames: the dead-ball
             rules' DEAD frames, the control events' team frames, and ALIVE frames without a
             team as a partial label "H or A" (confident_codes()). Each batch is half our
             windows, half DFL windows, with a tenth of the learning rate, so the model adapts
             to our tracking without forgetting the gaps and stoppages only DFL labels. Our
             usable frames are numbered into sequences with segments().

In this file: KINDS, mirror_index(), standardiser(), TCN, BiLSTM, make(), to_codes(),
confident_codes(), segments(), predict_steps(), fit() (returns a Fitted), fine_tune(),
predict(), and for_match() (the whole recipe for our match, out of fold and cached), which is
how hmm_v10 gets it. Needs PyTorch; trains on the GPU when there is one (~10 min on an A40).
"""

import copy
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from mtah.dead_ball.detector import ALIVE, DEAD
from mtah.external import adapter, dfl
from mtah.external import shape_transfer as st
from mtah.utils import runs_of

# Params
STEP = 5                    # 25 fps -> 5 steps a second: neighbouring frames are near-copies
WINDOW = 150                # steps per window: 30 s
HOP = 50                    # when predicting a whole half, a window starts every 10 s
WIDTH = 64                  # TCN channels; the LSTM's input projection
DILATIONS = (1, 2, 4, 8, 16, 32)   # TCN: a step sees 1 + 2 * 63 = 127 steps, about 25 s
HIDDEN = 48                 # LSTM units per direction: ~100k parameters, as the TCN
DROPOUT = 0.1
BATCH = 64                  # windows per gradient step
WINDOWS_PER_EPOCH = 4096
MAX_EPOCHS = 50
PATIENCE = 8                # stop after 8 epochs without a lower validation loss
LR = 1e-3
WEIGHT_DECAY = 1e-2
CLIP = 1.0                  # gradient norm clipping (keeps the LSTM stable)
FT_EPOCHS = 5               # fine-tuning on our match: epochs, fixed in advance
FT_LR = 1e-4                # a tenth of LR: adjust, do not relearn
FT_SHARE = 0.5              # share of our windows in a fine-tuning batch (the rest DFL)
KINDS = ("tcn", "lstm")
SEEDS = (0, 1, 2)           # for_match(): each kind trained 3 times, the runs averaged
VERSION = 1                 # for_match() cache; bump when the models or their training change
CODE = {c: k for k, c in enumerate(st.CLASSES)}     # H 0, A 1, D 2
ALIVE_ONLY = 3              # our match: the ball is in play, the team is not known
NO_LABEL = -1


def mirror_index(columns=st.COLUMNS):
    """(perm, sign): the mirror of a feature row x is x[perm] * sign.

    The mirror is the same frame with the teams swapped (as shape_transfer.mirror() does on a
    table): a column ending in _h takes the value of the matching _a column and the other way
    round, a _diff column (home minus away) changes sign, any other column stays.

        mirror_index(["advance_h", "advance_a", "advance_diff"])
        # -> (array([1, 0, 2]), array([ 1.,  1., -1.], dtype=float32))
        x = [5, -3, 8]   ->   x[perm] * sign = [-3, 5, -8]
    """
    columns = list(columns)
    perm = np.arange(len(columns))
    sign = np.ones(len(columns), dtype=np.float32)
    for k, column in enumerate(columns):
        if column.endswith("_h"):
            perm[k] = columns.index(column[:-2] + "_a")
        elif column.endswith("_a"):
            perm[k] = columns.index(column[:-2] + "_h")
        elif column.endswith("_diff"):
            sign[k] = -1.0
    return perm, sign


def standardiser(tables, columns=st.COLUMNS):
    """(mean, std) per column over all rows of `tables` and their mirrors, as float32 arrays.

    Counting every row twice, as it is and mirrored, makes the statistics themselves symmetric:
    an _h column gets the same mean and std as its _a column, and a _diff column has mean 0.
    Then standardising a mirrored row gives the mirror of the standardised row, so fit() can
    standardise once and mirror windows later. A column that never varies gets std 1. Missing
    values (NaN: marking_rate is undefined on a few frames at the edges of a half, 4 frames a
    DFL match) are left out of the statistics.

        table: advance_h [1, 3], advance_a [5, 7], advance_diff [-4, -4]
        standardiser([table], ["advance_h", "advance_a", "advance_diff"])
        # -> mean [4, 4, 0], std [2.236, 2.236, 4]
        #    (advance_h and _a both see 1, 3, 5, 7; the diff sees -4, -4, 4, 4)
    """
    perm, sign = mirror_index(columns)
    x = np.concatenate([np.asarray(t[list(columns)], dtype=np.float64) for t in tables])
    both = np.concatenate([x, x[:, perm] * sign])
    std = np.nanstd(both, axis=0)
    return (np.nanmean(both, axis=0).astype(np.float32),
            np.where(std > 0, std, 1.0).astype(np.float32))


class TCN(nn.Module):
    """Temporal convolutional network: (batch, steps, features) -> (batch, steps, 3) logits.

    A 1x1 projection to WIDTH channels, then one residual block per dilation in DILATIONS. Each
    block normalises every step over its channels (LayerNorm), applies a kernel-3 convolution
    whose taps are `dilation` steps apart (one before, the step itself, one after; zero-padded,
    so the output has as many steps as the input), a GELU, a 1x1 convolution and dropout, and
    adds the result to its input. With dilations 1 ... 32 a step's output depends on the 63
    steps on either side: 127 steps, about 25 s. A final LayerNorm and 1x1 layer give one logit
    per class (H, A, D) per step; softmax turns them into probabilities.

        model = TCN(36)
        model(torch.zeros(2, 150, 36)).shape    # -> torch.Size([2, 150, 3])
        sum(p.numel() for p in model.parameters())   # -> 102531
    """

    def __init__(self, n_features, width=WIDTH, dilations=DILATIONS, dropout=DROPOUT):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in dilations)
        self.convs = nn.ModuleList(nn.Conv1d(width, width, 3, dilation=d, padding=d)
                                   for d in dilations)
        self.mixes = nn.ModuleList(nn.Linear(width, width) for _ in dilations)
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(st.CLASSES))

    def forward(self, x):
        h = self.inp(x)                                         # (batch, steps, width)
        for norm, conv, mix in zip(self.norms, self.convs, self.mixes):
            y = conv(norm(h).transpose(1, 2)).transpose(1, 2)   # convolve over steps
            h = h + self.drop(mix(nn.functional.gelu(y)))
        return self.head(self.out_norm(h))


class BiLSTM(nn.Module):
    """Bidirectional LSTM: (batch, steps, features) -> (batch, steps, 3) logits.

    A linear projection to WIDTH, then two LSTM layers of HIDDEN units in each direction: one
    reads the window from start to end, the other from end to start, so a step's output
    summarises everything before it and everything after it within the window. Dropout sits
    between the layers; a LayerNorm and a linear layer on the 2 * HIDDEN outputs give one logit
    per class (H, A, D) per step.

        model = BiLSTM(36)
        model(torch.zeros(2, 150, 36)).shape    # -> torch.Size([2, 150, 3])
        sum(p.numel() for p in model.parameters())   # -> 102691
    """

    def __init__(self, n_features, width=WIDTH, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.lstm = nn.LSTM(width, hidden, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.out_norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Linear(2 * hidden, len(st.CLASSES))

    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.head(self.out_norm(h))


def make(kind, n_features):
    """A new, untrained model of the given kind ("tcn" or "lstm") for n_features inputs.

        make("lstm", 36)                # -> BiLSTM(...)
        make("gru", 36)                 # -> ValueError: kind must be one of ('tcn', 'lstm')
    """
    if kind not in KINDS:
        raise ValueError("kind must be one of {}".format(KINDS))
    return TCN(n_features) if kind == "tcn" else BiLSTM(n_features)


@dataclass
class Fitted:
    """A trained sequence model with what predict() needs to apply it to a new match.

    model        the TCN or BiLSTM, in eval mode, on the device it was trained on
    kind         "tcn" or "lstm"
    mean, std    the standardiser() of the training matches (float32, one per column)
    best_epoch   the epoch (1-based) whose weights were kept; the last one without validation
    history      validation loss after each epoch (empty without validation)
    """
    model: nn.Module
    kind: str
    mean: np.ndarray
    std: np.ndarray
    best_epoch: int
    history: list = field(default_factory=list)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_steps(model, x):
    """(L, 3) probabilities for one sequence of steps x, (L, features) on the model's device.

    The sequence is cut into windows of WINDOW steps starting every HOP steps, the last one
    ending exactly at the end, and each window goes through the model. Every step gets the
    average of the answers from all windows that contain it, each weighted by its distance to
    that window's nearer edge plus one (1 at the edge, 75 in the middle of a 150-step window),
    because an answer from the middle of a window has seen context on both sides. A sequence of
    at most WINDOW steps is one window.

        L = 250, WINDOW 150, HOP 50  ->  windows start at 0, 50, 100
        step 120 is in all three, at positions 120, 70 and 20, weights 30, 71 and 21
        predict_steps(model, x).shape           # -> torch.Size([250, 3]), rows sum to 1
    """
    L = len(x)
    if L <= WINDOW:
        return torch.softmax(model(x[None]), dim=-1)[0]
    starts = list(range(0, L - WINDOW + 1, HOP))
    if starts[-1] != L - WINDOW:
        starts.append(L - WINDOW)
    position = torch.arange(WINDOW, device=x.device)
    weight = (torch.minimum(position, WINDOW - 1 - position) + 1).float()[:, None]
    total = torch.zeros(L, len(st.CLASSES), device=x.device)
    count = torch.zeros(L, 1, device=x.device)
    for chunk in range(0, len(starts), 256):
        s = torch.tensor(starts[chunk:chunk + 256], device=x.device)
        rows = s[:, None] + position[None]                      # (windows, WINDOW)
        p = torch.softmax(model(x[rows]), dim=-1) * weight      # (windows, WINDOW, 3)
        total.index_add_(0, rows.reshape(-1), p.reshape(-1, p.shape[-1]))
        count.index_add_(0, rows.reshape(-1), weight.expand(len(s), -1, -1).reshape(-1, 1))
    return total / count


def to_codes(labels):
    """int64 class codes of H / A / D labels: 0 H, 1 A, 2 D (CODE).

        to_codes(["H", "D", "A"])        # -> array([0, 2, 1])
    """
    return np.array([CODE[v] for v in np.asarray(labels)], dtype=np.int64)


def confident_codes(pseudo, dead_labels):
    """Training codes for our match from its confident frames: 0 H, 1 A, 2 D, 3 alive, -1 none.

    pseudo is the control-event pseudo-label per frame (shape_model.pseudo_labels(): 1 home,
    0 away, -1 unknown), dead_labels the dead-ball detector's confident labels (DEAD, ALIVE or
    UNKNOWN; mtah/dead_ball/detector.py). Neither covers every frame, and they answer different
    questions, so a frame gets:

      2 (D)          where the rules are sure the ball is dead
      0 or 1 (H, A)  where a control event (or a pass between two) names the team and the rules
                     do not call the ball dead
      3 (ALIVE)      where the rules are sure the ball is in play but no event names the team:
                     a partial label, "H or A" (the loss only asks P(D) to be low there)
      -1             everywhere else: no label, the frame is context only

        pseudo      = [1, 0, -1, -1, 1]
        dead_labels = [ALIVE, UNKNOWN, ALIVE, DEAD, DEAD]
        confident_codes(pseudo, dead_labels)   # -> [0, 1, 3, 2, 2]  (the last: dead wins)
    """
    pseudo, dead_labels = np.asarray(pseudo), np.asarray(dead_labels)
    codes = np.full(len(pseudo), NO_LABEL, dtype=np.int64)
    codes[(dead_labels == ALIVE) & (pseudo < 0)] = ALIVE_ONLY
    team = (pseudo >= 0) & (dead_labels != DEAD)
    codes[team] = np.where(pseudo[team] == 1, CODE["H"], CODE["A"])
    codes[dead_labels == DEAD] = CODE["D"]
    return codes


def segments(mask):
    """int32 per frame: 1, 2, 3 ... numbering the runs of True in mask, 0 elsewhere.

    The sequences the models read, for a match whose usable frames are not simply two halves:
    every stretch of consecutive usable frames is one sequence (fit, fine_tune and predict
    treat 0 as "not used"). For DFL the half (1 or 2) is used directly.

        segments([F, T, T, F, T])        # -> [0, 1, 1, 0, 2]
    """
    out = np.zeros(len(mask), dtype=np.int32)
    for k, (start, stop) in enumerate(runs_of(np.asarray(mask, dtype=bool)), start=1):
        out[start:stop] = k
    return out


def _sequences(table, segment, mean, std, codes=None):
    """[(rows, standardised x, codes or None)] per sequence of one match, numpy.

    segment numbers the sequences per row (the half for DFL, segments() for our match); rows
    with segment 0 are left out. rows are the table rows of a sequence, x their standardised
    features, codes the class codes of those rows (None without codes). A missing value (NaN)
    becomes 0 after standardising: the training mean, the most neutral input.

        table: advance_h [1, NaN], segment [1, 1], mean 4, std 2
        _sequences(table, segment, mean, std)[0][1][:, advance_h]   # -> [-1.5, 0.0]
    """
    z = (np.asarray(table[st.COLUMNS], dtype=np.float32) - mean) / std
    z = np.nan_to_num(z, nan=0.0)
    segment = np.asarray(segment)
    out = []
    for h in np.unique(segment[segment > 0]):
        rows = np.flatnonzero(segment == h)
        out.append((rows, z[rows], None if codes is None else np.asarray(codes)[rows]))
    return out


def _loss(logits, codes):
    """Mean loss over the labelled steps: cross-entropy for H / A / D, -log P(H or A) for ALIVE.

    codes as from confident_codes(); steps coded -1 do not count. With only H / A / D codes,
    as on DFL, this is the plain cross-entropy. 0 (with a gradient) when nothing is labelled.

        logits all 0 (P = 1/3 each), codes [0, 3, -1]
        _loss(logits, codes)             # -> (log 3 + log 1.5) / 2 = 0.752
    """
    logp = torch.log_softmax(logits, dim=-1).reshape(-1, logits.shape[-1])
    codes = codes.reshape(-1)
    full = (codes >= 0) & (codes < ALIVE_ONLY)
    alive = codes == ALIVE_ONLY
    n = int(full.sum()) + int(alive.sum())
    if n == 0:
        return logits.sum() * 0.0
    loss = -logp[full, codes[full]].sum() - torch.logsumexp(logp[alive, :2], dim=-1).sum()
    return loss / n


def _validation_loss(model, sequences, device):
    """_loss() over the steps of every sequence (first of the 5 interleaved step sequences)."""
    total, n = 0.0, 0
    with torch.no_grad():
        for _, z, codes in sequences:
            p = predict_steps(model, torch.from_numpy(z[::STEP]).to(device))
            y = torch.from_numpy(codes[::STEP]).to(device)
            labelled = int((y >= 0).sum())
            if labelled:
                total += float(_loss(torch.log(p.clamp_min(1e-7)), y)) * labelled
                n += labelled
    return total / n


class _Windows:
    """Random training windows from a list of _sequences() parts, kept on the device.

    All parts are joined into one tensor; a window is WINDOW steps, every STEP-th frame, from a
    random start inside one part (parts drawn in proportion to how many starts they have), so a
    window never crosses from one sequence into the next. Parts shorter than one window are
    skipped. sample() mirrors each window with probability one half (features with
    mirror_index(), codes H and A swapped; D, ALIVE and -1 stay).

        windows = _Windows(parts, device)
        x, y = windows.sample(rng, 64)   # -> (64, WINDOW, 36) floats, (64, WINDOW) codes
    """

    def __init__(self, parts, device):
        span = (WINDOW - 1) * STEP + 1
        parts = [p for p in parts if len(p[1]) >= span]
        self.x = torch.from_numpy(np.concatenate([z for _, z, _ in parts])).to(device)
        self.y = torch.from_numpy(np.concatenate([c for _, _, c in parts])).to(device)
        self.offsets = np.cumsum([0] + [len(z) for _, z, _ in parts[:-1]])
        self.room = np.array([len(z) - span + 1 for _, z, _ in parts])
        self.lag = torch.arange(WINDOW, device=device) * STEP
        perm, sign = mirror_index()
        self.perm, self.sign = torch.from_numpy(perm).to(device), torch.from_numpy(sign).to(device)
        self.device = device

    def sample(self, rng, n):
        which = rng.choice(len(self.room), size=n, p=self.room / self.room.sum())
        start = self.offsets[which] + rng.integers(0, self.room[which])
        rows = torch.from_numpy(start).to(self.device)[:, None] + self.lag[None]
        x, y = self.x[rows], self.y[rows]
        flip = torch.from_numpy(rng.random(n) < 0.5).to(self.device)
        x = torch.where(flip[:, None, None], x[..., self.perm] * self.sign, x)
        y = torch.where(flip[:, None] & (y >= 0) & (y < 2), 1 - y, y)
        return x, y


def _step(model, optimiser, x, y):
    """One AdamW step on the batch (x, y) with _loss(), gradients clipped at CLIP."""
    loss = _loss(model(x), y)
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimiser.step()


def fit(tables, labels, halves, kind, seed=0, valid=None, epochs=None, device=None):
    """Train a TCN or BiLSTM on several matches; returns a Fitted.

    tables, labels and halves are lists with one entry per training match: the feature table
    and H / A / D labels from shape_transfer.build(), and the half (1 or 2) of every row
    (dfl.load(match).half). valid is the same triple for ONE other match, used only to decide
    when to stop: after every epoch (WINDOWS_PER_EPOCH random windows in batches of BATCH) the
    validation loss is computed, and training stops PATIENCE epochs after the best one, whose
    weights are kept. Without valid, it trains for `epochs` epochs (MAX_EPOCHS by default) and
    keeps the last weights.

    A training window is WINDOW steps (every STEP-th frame) from a random start in a random
    half, halves drawn in proportion to their length, so windows never cross half-time; each
    window is mirrored with probability one half (_Windows). The loss is the cross-entropy of
    every step (_loss); AdamW with LR and WEIGHT_DECAY, gradients clipped at CLIP. `seed` fixes
    the initial weights and the windows drawn (the GPU's LSTM kernels may still differ in the
    last digits between runs).

        fitted = fit([table_1, table_2], [labels_1, labels_2], [half_1, half_2], "tcn",
                     seed=0, valid=(table_3, labels_3, half_3))
        fitted.best_epoch, len(fitted.history)    # -> e.g. (14, 22): stopped 8 epochs later
    """
    device = device or _device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = standardiser(tables)
    windows = _Windows([s for t, lab, h in zip(tables, labels, halves)
                        for s in _sequences(t, h, mean, std, to_codes(lab))], device)
    valid_parts = None if valid is None else _sequences(valid[0], valid[2], mean, std,
                                                        to_codes(valid[1]))

    model = make(kind, len(st.COLUMNS)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_state, best_epoch, history = np.inf, None, 0, []
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        for _ in range(WINDOWS_PER_EPOCH // BATCH):
            _step(model, optimiser, *windows.sample(rng, BATCH))
        if valid_parts is None:
            continue
        model.eval()
        history.append(_validation_loss(model, valid_parts, device))
        if history[-1] < best:
            best, best_state, best_epoch = history[-1], copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs or MAX_EPOCHS
    model.eval()
    return Fitted(model, kind, mean, std, best_epoch, history)


def fine_tune(fitted, table, segment, codes, dfl, seed=0, epochs=FT_EPOCHS, lr=FT_LR,
              share=FT_SHARE):
    """A copy of a DFL-trained Fitted, trained further on our match's confident frames.

    table is our match's feature table (shape_transfer.COLUMNS), segment the sequences to train
    on (segments(); 0 = not used, e.g. a held-out block), codes the confident_codes() per frame.
    dfl = (tables, labels, halves) of DFL matches, as for fit(). Every batch of BATCH windows
    is `share` from our match and the rest from DFL, so the model keeps what DFL taught it
    about the frames we have no labels for (gaps, stoppages), instead of drifting towards the
    easy confident frames only. Our windows hold unlabelled frames too: they are context, and
    the loss (_loss) counts only the labelled steps, with ALIVE as "H or A". It trains for
    `epochs` epochs of WINDOWS_PER_EPOCH windows with the small learning rate `lr`, all layers,
    and keeps the last weights: there is no validation, as none of our frames may decide it.
    The standardiser stays the DFL one. The original `fitted` is not changed.

        tuned = fine_tune(fitted, table, segments(graded & (block != 3)), codes,
                          (dfl_tables, dfl_labels, dfl_halves), seed=0)
        tuned.best_epoch                 # -> 5  (FT_EPOCHS)
    """
    device = next(fitted.model.parameters()).device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ours = _Windows(_sequences(table, segment, fitted.mean, fitted.std, codes), device)
    theirs = _Windows([s for t, lab, h in zip(*dfl)
                       for s in _sequences(t, h, fitted.mean, fitted.std, to_codes(lab))], device)
    n_ours = int(round(BATCH * share))
    model = copy.deepcopy(fitted.model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(epochs * (WINDOWS_PER_EPOCH // BATCH)):
        xa, ya = ours.sample(rng, n_ours)
        xb, yb = theirs.sample(rng, BATCH - n_ours)
        _step(model, optimiser, torch.cat([xa, xb]), torch.cat([ya, yb]))
    model.eval()
    return Fitted(model, fitted.kind, fitted.mean, fitted.std, epochs, [])


def predict(fitted, table, half):
    """(n, 3) float32 P(H), P(A), P(D) for every row of one match, symmetric in the teams.

    half numbers the sequences (the DFL half, or segments() for our match); rows numbered 0
    are not predicted and stay NaN. For each sequence, the frames are split into the STEP interleaved step sequences (frames
    0, 5, 10 ...; 1, 6, 11 ...; ...); each is labelled with predict_steps(), as it is and
    mirrored (with P(H) and P(A) of the mirror swapped back), the two averaged, and the answers
    put back on their frames. So every 25 fps frame gets its own prediction, and swapping which
    team is called home swaps P(H) and P(A) exactly.

        p = predict(fitted, table, half)
        p.shape, p.sum(axis=1)[:2]      # -> ((145967, 3), array([1., 1.], dtype=float32))
    """
    device = next(fitted.model.parameters()).device
    perm, sign = mirror_index()
    perm, sign = torch.from_numpy(perm).to(device), torch.from_numpy(sign).to(device)
    out = np.full((len(table), len(st.CLASSES)), np.nan, dtype=np.float32)
    fitted.model.eval()
    with torch.no_grad():
        for rows, z, _ in _sequences(table, half, fitted.mean, fitted.std):
            z = torch.from_numpy(z).to(device)
            for phase in range(STEP):
                seq = z[phase::STEP]
                straight = predict_steps(fitted.model, seq)
                flipped = predict_steps(fitted.model, seq[:, perm] * sign)[:, [1, 0, 2]]
                out[rows[phase::STEP]] = ((straight + flipped) / 2).cpu().numpy()
    return out



def for_match(match, path=None, gaps=None, extra=()):
    """(n_frames, 3) float32 P(H), P(A), P(D) of our match: the fine-tuned TCN + LSTM average.

    The recipe measured in scripts/external/bar_ath_finetune.py, for hmm_v10:

      1. for each kind ("tcn", "lstm") and seed in SEEDS, fit() on the first 5 DFL matches
         (dfl.MATCHES), early stopping on the 6th; players degraded with our match's drop-out
         lengths (adapter.presence_gaps), features and labels from shape_transfer.build()
      2. our confident frames: confident_codes() of the control-event pseudo-labels
         (shape_model.pseudo_labels) and the dead-ball rules (detector.confident_labels), on
         the graded frames
      3. out of fold over the shape model's time blocks (shape_model.blocks(), 10): for each
         block, fine_tune() on the other blocks' frames (DFL batches from the 6 matches above)
         and predict() that block
      4. the seeds of a kind averaged, then the two kinds averaged

    So no frame's prediction comes from a model trained on its block's labels, and the model
    reads only players, never the ball. Frames off the graded halves are NaN. The result is
    kept in cache/external/dfl/bar_ath_sequence_v<VERSION>.npy (gitignored: derived from our
    match) and read from there when its length matches. GPU training is not bit-for-bit
    repeatable, so a rerun without the cache gives probabilities that agree closely, not
    exactly.

    The masking test (mtah/validation/masking.py) runs methods on a copy of the match with the
    ball hidden in stretches; that copy has the same length, so it gets these cached
    probabilities. That is still out of fold for the hidden frames: their block's labels were
    never trained on. The one difference from rerunning on the masked copy is that the OTHER
    blocks would lose the labels of their hidden stretches (control events need the ball), so
    the fine-tuning there would see a little less.

    Another match (a Metrica benchmark match, mtah/external/metrica_bench.py) passes its own
    cache `path`, so our match's cache is never overwritten, and `gaps`, the drop-out lengths
    the DFL players were degraded with (ours; default: adapter.presence_gaps(match)). Its own
    confident frames are then the ones fine-tuned on, exactly as ours are here.

    extra adds labelled matches, as (name, table, labels, half) entries (mtah/external/pool.py:
    the Metrica matches; hmm_v12): they join the 5 DFL training matches of step 1 (the 6th DFL
    match stays the only validation match) and the DFL batches of the fine-tuning in step 3. A
    caller passing extra must also pass its own path.

        p = for_match(load())
        for_match(metrica_match, path=".../metrica_game1_random_sequence.npy", gaps=our_gaps)
        p[graded][0]                   # -> e.g. [0.04, 0.03, 0.93]   a frame before kick-off
    """
    from mtah import features
    from mtah.dead_ball.detector import confident_labels
    from mtah.possession import shape_model
    from mtah.possession.control import control_events
    from mtah.features import team_shape

    if path is None:
        path = os.path.join(dfl.CACHE, "bar_ath_sequence_v{}.npy".format(VERSION))
    if os.path.exists(path):
        p = np.load(path)
        if len(p) == match.n_frames:
            return p
    graded = np.asarray(match.graded)
    table = team_shape.compute(match)[st.COLUMNS]
    pseudo = shape_model.pseudo_labels(match, control_events(match))
    codes = confident_codes(pseudo, confident_labels(match, features.compute(match)))
    codes[~graded] = NO_LABEL
    block = shape_model.blocks(graded)
    whole = segments(graded)

    if gaps is None:
        gaps = adapter.presence_gaps(match)
    data = {m: st.build(m, gaps, seed=k) + (dfl.load(m).half,) for k, m in enumerate(dfl.MATCHES)}
    more = [(e[1], e[2], e[3]) for e in extra]
    train = [data[m] for m in dfl.MATCHES[:5]] + more
    tune = tuple(list(x) for x in zip(*([data[m] for m in dfl.MATCHES[:6]] + more)))
    kinds = []
    for kind in KINDS:
        runs = []
        for seed in SEEDS:
            pre = fit([d[0] for d in train], [d[1] for d in train], [d[2] for d in train], kind,
                      seed=seed, valid=data[dfl.MATCHES[5]])
            oof = np.full((match.n_frames, len(st.CLASSES)), np.nan, dtype=np.float32)
            for b in range(shape_model.N_BLOCKS):
                tuned = fine_tune(pre, table, segments(graded & (block != b)), codes, tune,
                                  seed=seed)
                oof[block == b] = predict(tuned, table, whole)[block == b]
            runs.append(oof)
        kinds.append(np.mean(runs, axis=0))
    p = np.mean(kinds, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, p)
    return p
