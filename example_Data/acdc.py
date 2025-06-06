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
    img_1024 = np.load(
        join(path), "r", allow_pickle=True
    )
    x, y = img_1024.shape
    img_448 = zoom(img_1024, (size[0] / x, size[1] / y), order=0)
    img = img_448.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    gt = np.load(
        path, "r", allow_pickle=True
    )
    gt_448 = Image.fromarray(np.uint8(gt)).resize((size[0], size[1]), Image.NEAREST)
    seg = np.array(gt_448)
    seg = np.stack([seg == 0, seg == 1, seg == 2, seg == 3])
    seg = seg.astype(np.float32)
    return seg

def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in os.listdir(path):
        if 'acdc' in file:
            img = process_img(join(path, file), size=size)
            seg_file = join(path, file).replace('imgs', 'gts')
            seg = process_seg(seg_file, size=size)
            data.append((img, seg, file))
    return data


def require_download_wbc():
    dest_folder = pathlib.Path("/data1/segmentation_WBC-master/")

    if not dest_folder.exists():
        repo_url = "https://github.com/zxaoyou/segmentation_WBC.git"
        subprocess.run(
            ["git", "clone", repo_url, str(dest_folder),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class ACDCDataset(Dataset):
    dataset: Literal["JTSC", "CV"]
    split: Literal["support", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    support_frac: float = 0.7
    size: Tuple[int, int] = (128, 128)

    def __post_init__(self):
        root = '/newdata3/xsa/ICUSeg_Data/mri/npy/imgs'
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y), name) for x, y, name in load_folder(root, size=self.size)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
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
