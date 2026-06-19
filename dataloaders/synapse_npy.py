"""Synapse npy 格式数据加载器（供 HPC 使用）。

HPC 上数据结构：
    <npy_dir>/{case_id}_image.npy  shape: (D, H, W) float32, CT HU 值
    <npy_dir>/{case_id}_label.npy  shape: (D, H, W) float32, 整数类别

预处理：clip(-125, 275) → normalize to [0, 1]
Padding：若某维 < patch_size，自动用 0 padding 到 patch_size
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_npy(npy_dir: str, case_id: str):
    """加载单个样本，预处理并返回 (image, label) numpy float32。

    npy 文件存储为 (D, H, W)，转置为 (H, W, D) 以匹配 h5 训练格式，
    保证训练 patch 轴向与验证一致（slice 轴在 dim 2）。
    """
    img = np.load(f"{npy_dir}/{case_id}_image.npy").astype(np.float32)
    lbl = np.load(f"{npy_dir}/{case_id}_label.npy").astype(np.float32)
    img = np.clip(img, -125.0, 275.0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # (D=80, H=160, W=160) → (H=160, W=160, D=80) 匹配 h5 轴向
    img = img.transpose(1, 2, 0)
    lbl = lbl.transpose(1, 2, 0)
    return img, lbl


def _pad_to(vol: np.ndarray, min_shape: tuple) -> np.ndarray:
    """若体积某维小于 min_shape，用 0 padding 到 min_shape（居中）。"""
    pads = []
    for s, m in zip(vol.shape, min_shape):
        if s < m:
            total = m - s
            before = total // 2
            after = total - before
            pads.append((before, after))
        else:
            pads.append((0, 0))
    if any(p != (0, 0) for p in pads):
        vol = np.pad(vol, pads, mode='constant', constant_values=0)
    return vol


class Synapse_fast_npy(Dataset):
    """与 Synapse_fast（h5 版）接口完全相同，但从 npy 文件读取。"""

    def __init__(self, labeled_list, unlabeled_list, base_dir=None, transform=None):
        self.transform = transform
        self.labeled_list = labeled_list
        self.unlabeled_list = unlabeled_list

        self.images_l, self.labels_l = [], []
        for i, cid in enumerate(labeled_list):
            img, lbl = _load_npy(base_dir, cid)
            self.images_l.append(img)
            self.labels_l.append(lbl)
            print(f"Loading {i:2d}-th labeled sample from {base_dir}/{cid}_image.npy")

        self.images_u, self.labels_u = [], []
        for i, cid in enumerate(unlabeled_list):
            img, lbl = _load_npy(base_dir, cid)
            self.images_u.append(img)
            self.labels_u.append(lbl)
            print(f"Loading {i:2d}-th unlabeled sample from {base_dir}/{cid}_image.npy")

        print(f"Loaded! Total {len(labeled_list) + len(unlabeled_list)} samples for training")
        print(len(self.images_l), len(self.labels_l), len(self.images_u), len(self.labels_u))

    def __len__(self):
        return len(self.unlabeled_list) * 4

    def __getitem__(self, idx):
        if idx < len(self.unlabeled_list) * 2:
            i = idx % len(self.labeled_list)
            image, label = self.images_l[i], self.labels_l[i]
        else:
            i = idx % len(self.unlabeled_list)
            image, label = self.images_u[i], self.labels_u[i]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
