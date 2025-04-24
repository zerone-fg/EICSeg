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

    # img_save = img
    # img_save.save("/newdata3/xsa/ICUSeg/mambamodel/eval/wbc_vis/{}".format(str(path).split("/")[-1]))

    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)

    # seg.save("/newdata3/xsa/ICUSeg/mambamodel/eval/wbc_vis/{}".format(str(path).split("/")[-1]))
    # h, w = seg.shape
    # save_mask = torch.zeros((1, 1, h, w), dtype=torch.uint8)

    # for i, id in enumerate([128, 255]):
    #     save_mask[0, 0, seg == id] = torch.tensor(i + 1 + 4, dtype=torch.uint8)
    # save_mask = F.interpolate(save_mask, (448, 448), mode='nearest')
    # save_mask = save_mask.squeeze(0).squeeze(0)
    # save_colored_mask_1(np.array(save_mask), "/newdata3/xsa/ICUSeg/mambamodel/eval/wbc_vis/{}".format(str(path).split("/")[-1]))

    seg = np.stack([seg == 0, seg == 128, seg == 255])
    # seg[seg == 128] = 1
    # seg[seg == 255] = 2
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    # for file in sorted(path.glob("*.bmp")):
    #     img = process_img(file, size=size)
    #     seg_file = file.with_suffix(".png")
    #     seg = process_seg(seg_file, size=size)
    #     data.append((img / 255.0, seg, file))
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
        # path = root / {"JTSC": "Dataset 1", "CV": "Dataset 2"}[self.dataset]
        path = root
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y), name) for x, y, name in load_folder(path, size=self.size)]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
        if self.mode == 'test':
            self._idxs = self._split_indexes()
        # else:
        #     self._idxs = self._train_split_indexes()

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

    # def __getitem__(self, idx):
    #     img, seg, name = self._data[self._idxs[idx]]
    #     if self.label is not None:
    #         seg = seg[self._ilabel][None]
    #     return img, seg, name
    def __getitem__(self, idx):
        if self.split == "support":  # 仅对 support 数据动态打乱
            idx = np.random.randint(0, len(self._idxs))
        img, seg, name = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = seg[self._ilabel][None]
        return img, seg, name