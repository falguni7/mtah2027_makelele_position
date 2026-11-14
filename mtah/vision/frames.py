"""Half-resolution tactical frames for fine-tuning, extracted once per job to the node's local disk.

    from mtah.vision import frames
    path = frames.extract(video, wanted, os.environ["TMPDIR"])   # one pass over the video
    store = frames.FrameStore(path)
    img = store[11186]                    # (540, 960, 3) uint8 RGB, half resolution

The teammate's tiles are 1024x576 crops of the full frame shrunk by exactly 0.5 to the 512x288
network input, so a frame shrunk to 960x540 holds every tile at network scale: a tile is then a
plain 512x288 crop (mtah.vision.data). Shrinking by 0.5 with cv2.INTER_AREA averages each 2x2
block, which is what the teammate's bilinear resize at exactly 0.5 does too.

Storage (agreed 2026-11-11): nothing on our storage. Each job writes the frames it needs to
$TMPDIR (node-local, wiped after the job) as JPEG (quality QUALITY) in one data file plus an
index, and reads them back by random access. No frame is ever written to the repository or the
cache, and Claude never opens one.
"""

import os

import cv2
import numpy as np

# Params
HALF_W, HALF_H = 960, 540
QUALITY = 95


def half(frame):
    """The full 1920x1080 RGB frame shrunk by 0.5 (2x2 block means), uint8.

        f = np.zeros((1080, 1920, 3), np.uint8); f[0:2, 0:2] = 200     # one bright 2x2 block
        half(f).shape, half(f)[0, 0, 0], half(f)[0, 1, 0]              # -> (540, 960, 3), 200, 0
    """
    return cv2.resize(frame, (HALF_W, HALF_H), interpolation=cv2.INTER_AREA)


def extract(video, wanted, out_dir, progress_every=20000):
    """Decode `video` once, keep the `wanted` frame indices at half resolution; returns out_dir.

    Writes out_dir/frames.bin (the JPEGs back to back) and out_dir/index.npz (frame, offset,
    length). Frames are counted from 0 in decode order, as everywhere in the pipeline (video
    frame i = submission row i); decoding stops after the last wanted frame.

        extract("BAR-ATH_tactical.mp4", [11185, 11186], "/tmp/x")      # -> "/tmp/x"
        FrameStore("/tmp/x")[11186].shape                              # -> (540, 960, 3)
    """
    import av
    wanted = np.unique(np.asarray(wanted, np.int64))
    keep = set(wanted.tolist())
    last = int(wanted[-1])
    os.makedirs(out_dir, exist_ok=True)
    offsets, lengths, got = [], [], []
    pos = 0
    with open(os.path.join(out_dir, "frames.bin"), "wb") as out:
        container = av.open(video)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, fr in enumerate(container.decode(stream)):
            if i in keep:
                rgb = half(fr.to_ndarray(format="rgb24"))
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                assert ok
                out.write(jpg.tobytes())
                offsets.append(pos)
                lengths.append(len(jpg))
                got.append(i)
                pos += len(jpg)
            if progress_every and i % progress_every == 0:
                print("  decoded frame {} of {}, kept {}".format(i, last, len(got)), flush=True)
            if i >= last:
                break
        container.close()
    if len(got) != len(wanted):
        raise RuntimeError("video ended after {} of {} wanted frames".format(len(got), len(wanted)))
    np.savez(os.path.join(out_dir, "index.npz"), frame=np.array(got), offset=np.array(offsets),
             length=np.array(lengths))
    return out_dir


class FrameStore:
    """Random access to the frames written by extract(): store[frame] -> (540, 960, 3) RGB uint8.

    The data file is memory-mapped lazily, so a store can be handed to DataLoader workers.

        store = FrameStore(path)
        len(store), 11186 in store                  # -> number of frames, True / False
    """

    def __init__(self, path):
        self.path = path
        with np.load(os.path.join(path, "index.npz")) as ix:
            self.pos = {int(f): (int(o), int(n)) for f, o, n in zip(ix["frame"], ix["offset"],
                                                                    ix["length"])}
        self._data = None

    def __len__(self):
        return len(self.pos)

    def __contains__(self, frame):
        return int(frame) in self.pos

    def __getitem__(self, frame):
        if self._data is None:
            self._data = np.memmap(os.path.join(self.path, "frames.bin"), np.uint8, "r")
        o, n = self.pos[int(frame)]
        bgr = cv2.imdecode(np.asarray(self._data[o:o + n]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
