# -*- coding: utf-8 -*-
"""학습 중 증강. GPU 에서 배치 단위로 겁니다.

폰트 데이터는 (글자, 폰트)마다 이미지가 딱 한 장이라, 증강 없이 학습하면 모델이 시드와 무관하게
같은 모양만 내는 '폰트 복사기'가 됩니다. 약한 기하 변형과 획 굵기 변화를 섞어서
같은 글자도 쓸 때마다 조금씩 다른, 손글씨다운 분포를 만듭니다.

    strength=1 기준 (세기는 선형으로 곱해짐)
      회전 ±3°   기울임 ±5°   크기 0.9~1.05   이동 가로 ±2px 세로 ±1px
      탄성 왜곡 최대 약 1px (획이 미세하게 떨림)   획 굵기 ±1px (30% 확률)

  - 세로 이동을 작게 둔 이유: 마침표·쉼표·따옴표는 높이가 곧 의미라서 크게 움직이면 안 됩니다.
  - 좌우 반전·큰 회전·잘라내기는 넣지 않습니다 (글자가 바뀌거나 획이 잘림).
  - 각 샘플은 확률 p 로만 증강해서 원본 글자도 계속 봅니다.
"""
import math

import torch
import torch.nn.functional as F

S = 64


def augment(x, strength=1.0, p=0.5, generator=None):
    """x: (B,1,64,64) [-1,1] (흰 글자 +1). 같은 shape·범위로 돌려줍니다."""
    if strength <= 0 or p <= 0:
        return x
    B, dev = x.shape[0], x.device

    def U(*shape):          # [-1, 1) 균등
        return torch.rand(*shape, device=dev, generator=generator) * 2 - 1

    on = torch.rand(B, device=dev, generator=generator) < p
    if not on.any():
        return x
    s = strength
    x01 = (x + 1) / 2                               # 배경 0 으로 두고 변형 (grid_sample 의 바깥 = 0)

    # ---- 기하 변형 (affine + 탄성)
    rot = U(B) * math.radians(3) * s
    shear = U(B) * math.radians(5) * s
    scale = 1 + (torch.rand(B, device=dev, generator=generator) * 0.15 - 0.10) * s     # 0.9 ~ 1.05
    tx = U(B) * (2 * 2 / S) * s                     # ±2px  (정규화 좌표에서 1px = 2/64)
    ty = U(B) * (1 * 2 / S) * s                     # ±1px
    c, si, sh = torch.cos(rot), torch.sin(rot), torch.tan(shear)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = c / scale; th[:, 0, 1] = (-si + sh * c) / scale; th[:, 0, 2] = tx
    th[:, 1, 0] = si / scale; th[:, 1, 1] = (c + sh * si) / scale; th[:, 1, 2] = ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    # 탄성: 거친 4×4 무작위 변위를 부드럽게 키워서 더함 (진폭 약 1px)
    d = U(B, 2, 4, 4) * (1.0 * 2 / S) * s
    d = F.interpolate(d, size=(S, S), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    grid = grid + d
    y = F.grid_sample(x01, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # ---- 획 굵기 (약 ±0.5px). 침식(min-pool)은 가는 획을 회색으로 흐리게 만들어서 쓰지 않고,
    #      안티앨리어싱된 가장자리의 밝기를 옮겨서 획 중심은 그대로 흰색(1)로 둡니다.
    r = torch.rand(B, device=dev, generator=generator)
    k = min(1.0, s)
    thick = r < 0.15 * k                             # 15%: 굵게  — 살짝 번지게 한 뒤 대비를 올림
    thin = (r >= 0.15 * k) & (r < 0.30 * k)          # 15%: 가늘게 — 가장자리의 옅은 부분을 깎음
    if thick.any():
        w = torch.tensor([1.0, 2.0, 1.0], device=dev)
        ker = (w[:, None] * w[None, :] / 16).view(1, 1, 3, 3)
        y[thick] = (F.conv2d(y[thick], ker, padding=1) * 2.0).clamp(0, 1)
    if thin.any():
        y[thin] = ((y[thin] - 0.35) / 0.65).clamp(0, 1)

    # 보간 때문에 아주 작은 글자(마침표 등)의 획 중심이 옅어지는 것을 막기 위해 최대 밝기를 1로 다시 맞춤
    y = y / y.amax(dim=(1, 2, 3), keepdim=True).clamp(min=0.5)
    y = y.clamp(0, 1) * 2 - 1
    return torch.where(on[:, None, None, None], y, x)


if __name__ == "__main__":
    # 증강 미리보기: python augment.py ../font_ko_v2  → aug_preview.png
    import sys
    import numpy as np
    from PIL import Image
    import data as D
    sys.stdout.reconfigure(encoding="utf-8")
    root = sys.argv[1] if len(sys.argv) > 1 else "font_ko_v2"
    ds = D.load(root, "ft", "dev")
    x, _, _ = ds.batch(torch.arange(0, 5000, 625)[:8], "cpu")
    g = torch.Generator().manual_seed(0)
    rows = [x] + [augment(x, st, p=1.0, generator=g) for st in (0.5, 1.0, 1.0, 1.5)]
    img = torch.cat([torch.cat(list(r[:, 0]), 1) for r in rows], 0)
    Image.fromarray((255 - (img + 1) * 127.5).clamp(0, 255).byte().numpy()).save("aug_preview.png")
    print("aug_preview.png  (위부터: 원본, 세기 0.5, 1.0, 1.0, 1.5)")
