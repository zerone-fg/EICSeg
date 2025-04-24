
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage.interpolation import zoom
from PIL import Image
import pathlib
join = os.path.join


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img_1024 = Image.open(path)
    img_1024 = img_1024.resize(size, resample=Image.BILINEAR)
    img_1024 = img_1024.convert('L')

    img = np.array(img_1024)
    img = img.astype(np.float32)
    return img

def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    gt_448 = Image.open(path)
    gt_448 = gt_448.resize(size, resample=Image.NEAREST)
    seg = np.array(gt_448)
    seg = np.stack([seg == 0, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in os.listdir(path):
        img = process_img(join(path, file), size=size)
        seg_file = join(path, file).replace('image', 'mask')
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data


def require_download_wbc():
    dest_folder = pathlib.Path("/share/xieshiao/segmentation_WBC-master/")

    if not dest_folder.exists():
        repo_url = "https://github.com/zxaoyou/segmentation_WBC.git"
        subprocess.run(
            ["git", "clone", repo_url, str(dest_folder),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class ToothDataset(Dataset):
    dataset: Literal["JTSC", "CV"]
    split: Literal["support", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    support_frac: float = 0.7
    size: Tuple[int, int] = (128, 128)
    mode: str = 'test'

    def __post_init__(self):
        root = '/data1/tooth/train/image/'
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(root, size=self.size)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "background": 0}[self.label]
        if self.mode == 'test':
            self._idxs = self._split_indexes()
        else:
            self._idxs = self._train_split_indexes()

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
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg
