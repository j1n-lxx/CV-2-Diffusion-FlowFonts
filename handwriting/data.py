# -*- coding: utf-8 -*-
"""데이터. font_ko_v2.zip 을 푼 폴더를 읽습니다.

    font_ko_v2/
      font_ko_pretrain/         사전학습 (421개 스타일)
      font_ko_finetune_nanum/   나눔손글씨 109종
        train/ val/ test/ train_unseen_chars/ dev/   images.npy (N,64,64) uint8 + labels.npy (N,2) [font_id, char_id]
        meta/ fonts.csv chars.csv splits.json

Task 2 의 data.py 처럼 이미지는 [-1, 1] 로 돌려줍니다 (흰 글자 = +1, 배경 = -1).
"""
import csv
import os

import numpy as np
import torch

import hangul as H

SETS = {"pre": "font_ko_pretrain", "ft": "font_ko_finetune_nanum"}


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FontSet:
    """한 split 의 이미지 + (style, cond) 라벨.

    style : 스타일 임베딩 인덱스 (세트별 font_id, 혼합 학습이면 오프셋을 더함)
    cond  : (N, 4) [cho, jung, jong, extra]  — hangul.py 규칙
    """

    def __init__(self, x, style, char_id, cond_table, style_names, char_list):
        self.x, self.style, self.char_id = x, style, char_id
        self.cond_table = cond_table
        self.style_names = style_names     # 스타일 인덱스 → 폰트 이름
        self.chars = char_list             # char_id → 글자

    def __len__(self):
        return len(self.char_id)

    def batch(self, idx, device):
        """idx(LongTensor) → (x[-1,1] float (B,1,64,64), style (B,), cond (B,4)) on device."""
        i = idx.cpu().numpy()
        order = np.argsort(i)                      # memmap 은 정렬된 순서로 읽는 게 빠릅니다
        xb = np.empty((len(i), 64, 64), np.uint8)
        xb[order] = self.x[i[order]]
        x = torch.from_numpy(xb).to(device).float().div_(127.5).sub_(1.0)[:, None]
        st = self.style[idx.cpu()].to(device)
        cond = self.cond_table[self.char_id[idx.cpu()]].to(device)
        return x, st, cond


def load(root, which="pre", split="train", kinds=None, exclude_fonts=None, frac=1.0,
         in_memory=True, style_offset=0, seed=0):
    """root = font_ko_v2 폴더.

    kinds         : {'hangul','digit','latin_upper','latin_lower','punct','symbol'} 중 일부만
    exclude_fonts : 빼고 싶은 font_id 집합 (예: 사전학습에서 나눔손글씨 109종)
    frac          : 폰트마다 글자의 이 비율만 남김 (파인튜닝 데이터 100%·10%·1% 실험용)
    """
    d = os.path.join(root, SETS[which])
    X = np.load(os.path.join(d, split, "images.npy"), mmap_mode="r")
    L = np.load(os.path.join(d, split, "labels.npy"))
    fonts = _csv(os.path.join(d, "meta", "fonts.csv"))
    chars = _csv(os.path.join(d, "meta", "chars.csv"))

    keep = np.ones(len(L), bool)
    if kinds is not None:
        ok = [int(c["char_id"]) for c in chars if c["kind"] in kinds]
        keep &= np.isin(L[:, 1], ok)
    if exclude_fonts:
        keep &= ~np.isin(L[:, 0], list(exclude_fonts))
    if frac < 1.0:
        # 폰트마다 같은 글자 부분집합이 아니라, 폰트별로 다른 무작위 부분집합
        rng = np.random.default_rng(seed)
        sub = np.zeros(len(L), bool)
        for f in np.unique(L[keep, 0]):
            ids = np.nonzero(keep & (L[:, 0] == f))[0]
            k = max(1, int(round(len(ids) * frac)))
            sub[rng.choice(ids, k, replace=False)] = True
        keep &= sub
    idx = np.nonzero(keep)[0]
    if in_memory:
        x = np.ascontiguousarray(X[idx]) if len(idx) < len(L) else np.array(X)
        lab = L[idx]
    else:
        x, lab = _Indexed(X, idx), L[idx]

    style = torch.from_numpy(lab[:, 0].astype(np.int64)) + style_offset
    char_id = torch.from_numpy(lab[:, 1].astype(np.int64))
    names = [f["family"] + ("" if f.get("style", "Regular") in ("", "Regular") else " " + f["style"]) for f in fonts]
    return FontSet(x, style, char_id, H.cond_table(chars), names, [c["char"] for c in chars])


class _Indexed:
    """memmap 위에 인덱스를 씌운 얇은 래퍼 (in_memory=False 용)."""

    def __init__(self, X, idx):
        self.X, self.idx = X, idx

    def __getitem__(self, i):
        return self.X[self.idx[i]]

    def __len__(self):
        return len(self.idx)


def concat(a, b):
    """혼합 학습(c)용: 두 FontSet 을 합칩니다. b 의 style 은 이미 오프셋이 더해져 있어야 합니다."""
    x = np.concatenate([np.asarray(a.x), np.asarray(b.x)])
    return FontSet(x, torch.cat([a.style, b.style]), torch.cat([a.char_id, b.char_id]),
                   a.cond_table, a.style_names + b.style_names, a.chars)


def read_exclude(path):
    """exclude_fonts.txt → 사전학습 font_id 집합. 'none' 이거나 없으면 빈 집합.
    상대 경로는 현재 폴더, 없으면 이 파일(handwriting/) 옆에서 찾습니다."""
    if not path or str(path).lower() == "none":
        return set()
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        if not os.path.exists(alt):
            return set()
        path = alt
    ids = set()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(int(line.split("\t")[0]))
    return ids


def nanum_font_ids(root):
    """사전학습 세트에서 파인튜닝 세트와 같은 폰트(나눔손글씨 109종)의 font_id."""
    fonts = _csv(os.path.join(root, SETS["pre"], "meta", "fonts.csv"))
    return {int(f["font_id"]) for f in fonts if f["in_finetune_set"] == "1"}


def n_styles(root, which):
    return len(_csv(os.path.join(root, SETS[which], "meta", "fonts.csv")))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    root = sys.argv[1] if len(sys.argv) > 1 else "font_ko_v2"
    for w in ("pre", "ft"):
        s = load(root, w, "dev")
        x, st, c = s.batch(torch.arange(8), "cpu")
        print(f"{w} dev  {len(s):,}장  x {tuple(x.shape)} 범위 [{x.min():.2f}, {x.max():.2f}]  "
              f"style {st[:4].tolist()}  cond {c[0].tolist()} = {s.chars[int(s.char_id[0])]}")
