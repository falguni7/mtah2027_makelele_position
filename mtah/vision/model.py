"""The WASB network (HRNet, nttcom/WASB-SBDT) and the loss used to fine-tune it.

    from mtah.vision import model
    net = model.build()                              # pretrained soccer weights
    net = model.build(model.FINETUNED)               # our fine-tuned checkpoint
    logits = net(x)[0]                               # (B, 3, 288, 512) heatmap logits
    loss = model.wbce(logits, target)

The code and the weights live in third_party/WASB-SBDT (gitignored; commit 923462c, soccer
weights SHA-256 d0369572...2646c, see the work log 2026-11-11). That release has no training
code, so the loss is written here: WASB's `wbce` heatmap loss (TrackNetV2's weighted binary
cross-entropy, a focal loss with gamma 2), computed from logits so it stays finite.
"""

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WASB_DIR = os.path.join(ROOT, "third_party", "WASB-SBDT")
PRETRAINED = os.path.join(WASB_DIR, "pretrained_weights", "wasb_soccer_best.pth.tar")
FINETUNED = os.path.join(ROOT, "cache", "vision", "wasb_finetuned_v1.pt")


def build(weights=PRETRAINED):
    """WASB's HRNet with the given weights loaded strictly (pretrained file or our checkpoint).

        net = build()
        sum(p.numel() for p in net.parameters())     # -> 1481427
    """
    from omegaconf import OmegaConf
    if os.path.join(WASB_DIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(WASB_DIR, "src"))
    from models import build_model
    cfg = OmegaConf.create({"model": OmegaConf.load(os.path.join(WASB_DIR, "src", "configs",
                                                                 "model", "wasb.yaml"))})
    net = build_model(cfg)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    net.load_state_dict(state["model_state_dict"], strict=True)
    return net


def wbce(logits, target):
    """Mean of -[(1 - p)^2 y log p + p^2 (1 - y) log(1 - p)], p = sigmoid(logits).

    Confident correct pixels cost almost nothing, so the few ball pixels are not drowned by the
    many background ones.

        wbce(torch.tensor([10.0, -10.0]), torch.tensor([1.0, 0.0]))     # -> ~0 (both right)
        wbce(torch.tensor([0.0]), torch.tensor([1.0]))                  # -> 0.25 * log 2 = 0.173
    """
    p = torch.sigmoid(logits)
    loss = ((1 - p) ** 2) * target * F.logsigmoid(logits) \
        + (p ** 2) * (1 - target) * F.logsigmoid(-logits)
    return -loss.mean()
