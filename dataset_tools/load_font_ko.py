"""font_ko 로더 v2 (numpy 필수, torch 선택). 사전학습·파인튜닝 세트 공통.

    from load_font_ko import load_split, char_condition, FontKoDataset
    X, y, fonts, chars = load_split("font_ko_pretrain", "train")   # X: memmap (N,64,64) uint8
    # y[:,0] = font_id (meta/fonts.csv), y[:,1] = char_id (meta/chars.csv, 0..2448)
    cond = char_condition(chars)          # (2449, 4) int: cho, jung, jong, extra_id  (해당 없으면 -1)

    ds = FontKoDataset("font_ko_finetune_nanum", "train", kinds={"hangul", "punct"})
"""
import csv, os
import numpy as np

KINDS = ("hangul", "digit", "latin_upper", "latin_lower", "punct", "symbol")


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_split(root, split):
    X = np.load(os.path.join(root, split, "images.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, split, "labels.npy"))
    fonts = _csv(os.path.join(root, "meta", "fonts.csv"))
    chars = _csv(os.path.join(root, "meta", "chars.csv"))
    return X, y, fonts, chars


def char_condition(chars):
    """char_id → [cho, jung, jong, extra_id]. 한글은 extra_id=-1, 추가 문자는 cho/jung/jong=-1.
    모델에서는 -1을 '없음' 토큰(예: 각 임베딩의 마지막 인덱스)으로 바꿔 쓰면 됩니다."""
    return np.array([[int(c["cho"]), int(c["jung"]), int(c["jong"]), int(c["extra_id"])] for c in chars], np.int64)


try:
    import torch
    from torch.utils.data import Dataset

    class FontKoDataset(Dataset):
        """(x, font_id, char_id). x는 [-1,1] float (1,64,64).
        categories: fonts.csv의 category 중 일부만 (사전학습: handwriting/print/pixel)
        kinds: chars.csv의 kind 중 일부만 (hangul, digit, latin_upper, latin_lower, punct, symbol)
        exclude_fonts: 빼고 싶은 font_id 집합 (예: 사전학습에서 in_finetune_set==1인 폰트)"""

        def __init__(self, root, split, categories=None, kinds=None, exclude_fonts=None, transform=None):
            self.X, y, fonts, self.chars = load_split(root, split)
            keep = np.ones(len(y), bool)
            if categories is not None:
                ok = [int(f["font_id"]) for f in fonts if f["category"] in categories]
                keep &= np.isin(y[:, 0], ok)
            if kinds is not None:
                ok = [int(c["char_id"]) for c in self.chars if c["kind"] in kinds]
                keep &= np.isin(y[:, 1], ok)
            if exclude_fonts is not None:
                keep &= ~np.isin(y[:, 0], list(exclude_fonts))
            self.idx = np.nonzero(keep)[0]
            self.y = y
            self.transform = transform

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            k = self.idx[i]
            x = torch.from_numpy(np.array(self.X[k], dtype=np.float32) / 127.5 - 1.0)[None]
            if self.transform is not None:
                x = self.transform(x)
            return x, int(self.y[k, 0]), int(self.y[k, 1])
except ImportError:
    pass
