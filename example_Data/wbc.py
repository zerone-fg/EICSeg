import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F

color_map = {
    0: (160, 232, 130),
    1: (226, 226, 231),
    2: (243, 213, 211),
    3: (209, 244, 224),
    4: (255, 242, 204),
    5: (206, 175, 95),
    6: (148, 72, 140),
    7: (110, 87, 193),
    8: (41, 147, 107),
    9: (25, 91, 141)
}

def save_colored_mask_1(mask, save_path):
    h, w = mask.shape
    save_mask = np.zeros((h, w, 3))
    print(np.unique(mask))
    for i in np.unique(mask):
        if i != 0:
            save_mask[mask == i] = color_map[i]
    cv2.imwrite(save_path, save_mask)


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)

    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)

    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    import os
    for file in os.listdir("/newdata3/xsa/ICUSeg_Data/wbc1/imgs"):
        img = process_img(os.path.join("/newdata3/xsa/ICUSeg_Data/wbc1/imgs",file), size=size)
        seg_file = file.replace(".jpg", ".png")
        seg = process_seg(os.path.join("/newdata3/xsa/ICUSeg_Data/wbc1/gts",seg_file), size=size)
        data.append((img / 255.0, seg, file))
    return data


def require_download_wbc():
    dest_folder = pathlib.Path("/newdata3/xsa/ICUSeg_Data/wbc1")

    if not dest_folder.exists():
        repo_url = "https://github.com/zxaoyou/segmentation_WBC.git"
        subprocess.run(
            ["git", "clone", repo_url, str(dest_folder),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class WBCDataset(Dataset):
    dataset: Literal["JTSC", "CV"]
    split: Literal["support", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    support_frac: float = 0.7
    size: Tuple[int, int] = (128, 128),
    mode: str = 'test'

    def __post_init__(self):
        root = require_download_wbc()
        path = root
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y), name) for x, y, name in load_folder(path, size=self.size)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
        if self.mode == 'test':
            self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def _train_split_indexes(self):
        N = len(self._data)
        p = [i for i in range(N)]
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        if self.split == "support":  # 仅对 support 数据动态打乱
            idx = np.random.randint(0, len(self._idxs))
        img, seg, name = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg, name
