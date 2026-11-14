"""Ball detection from the video: fine-tuning the WASB ball detector on our own match.

WASB (nttcom/WASB-SBDT, MIT, cloned in third_party/WASB-SBDT) is a heatmap ball detector: it
takes 3 consecutive frames and gives one ball heatmap per frame. Its pretrained soccer weights
find our ball on only part of the frames, so it is fine-tuned on the tactical video with labels
from the tracking: the ball's pitch position projected into the image by the homography. No
person decides any label (RULES.md §1, pseudo-labels produced by an algorithm).

    split.py    which frames are valid labels, and the train / validation / test split of the
                tactical match (six 15-minute patches, 1-minute chunks, buffers)

    from mtah.vision import split
    s = split.load()                     # the saved split, see scripts/vision/wasb_split.py
    s["split"][frame]                    # -> 0 train, 1 validation, 2 test, -1 none
"""
