import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import os
from scipy.ndimage.interpolation import zoom
from PIL import Image
import pathlib
join = os.path.join


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg_i_l = PIL.Image.open(path)
    seg_o_l = PIL.Image.open(path.replace('i_label', 'o_label'))

    seg_i_l = seg_i_l.resize(size, resample=PIL.Image.NEAREST)
    seg_o_l = seg_o_l.resize(size, resample=PIL.Image.NEAREST)

    seg_i_l = np.array(seg_i_l)
    seg_o_l = np.array(seg_o_l)

    seg_i = np.array(seg_i_l == 255, dtype=np.uint8)
    seg_o = np.array(seg_o_l == 255, dtype=np.uint8)

    seg = seg_o - seg_i
    seg = np.stack([seg == 0, seg == 1])
    seg = seg.astype(np.float32)
    return seg

def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in os.listdir(path):
        img = process_img(join(path, file), size=size)
        seg_file_i = join(path, 'i_labels', file).replace('imgs', 'gts')
        seg_file_o = join(path, 'o_labels', file).replace('imgs', 'gts')
        if os.path.isfile(seg_file_i) and os.path.isfile(seg_file_o):
            seg = process_seg(seg_file_i, size=size)
            data.append((img / 255.0, seg, file))
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
class SCDDataset(Dataset):
    dataset: Literal["JTSC", "CV"]
    split: Literal["support", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    support_frac: float = 0.7
    size: Tuple[int, int] = (128, 128)
    mode: str = 'test'

    def __post_init__(self):
        root = '/newdata3/xsa/ICUSeg_Data/SCD/imgs'
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y), name) for x, y, name in load_folder(root, size=self.size)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
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
        if self.split == "support":  # 仅对 support 数据动态打乱
            idx = np.random.randint(0, len(self._idxs))
        img, seg, name = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg, name
