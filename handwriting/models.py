# -*- coding: utf-8 -*-
"""디노이저. Task 2 의 SmallUNet 을 64×64 와 '자모 + 스타일' 조건으로 키운 것입니다.

    64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64     (16·8 해상도에 self-attention)

조건은 Task 2 처럼 time embedding 에 더해집니다.
    temb + content(cho, jung, jong, extra) + style(font)
각 조건의 마지막 인덱스가 classifier-free guidance 용 '없음' 토큰입니다.

출력은 pred='eps' (Task 2 와 같음), pred='v', pred='rf' 중 하나를 학습합니다.
  - eps, v : 디퓨전 (곡선 경로 sqrt(ab) x0 + sqrt(1-ab) eps), DDIM 으로 샘플링
  - rf     : Rectified Flow (직선 경로 tau x0 + (1-tau) z), 출력 = 속도 x0 - z, 오일러로 샘플링
             시간은 디퓨전과 같은 정수 t (0..T-1) 를 쓰고 tau = 1 - t/(T-1) 로 바꿉니다 (t 가 클수록 노이즈).
1스텝 증류(distill.py)까지 갈 거면 v 를 쓰세요. eps 예측은 t≈T 에서 x0 를 복원할 때
1/sqrt(alpha_bar) 가 터져서 1스텝 학생 모델이 학습되지 않습니다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hangul as H


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    """Task 2 의 ResBlock 그대로 (+dropout)."""

    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(t))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    """공간 self-attention. 자모끼리의 위치 관계(초성-중성-종성 배치)를 전체적으로 보게 합니다."""

    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)
        self.h = heads

    def forward(self, x):
        B, C, Hh, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, Hh * W).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.o(a.transpose(-1, -2).reshape(B, C, Hh, W))


class HangulUNet(nn.Module):
    """base=64 이면 파라미터 약 550만 개, base=96 이면 약 1,160만 개.

    n_styles: 스타일(폰트) 개수. 마지막 인덱스 n_styles 가 '스타일 없음' 토큰입니다.
    """

    def __init__(self, n_styles, base=64, tdim=256, cdim=128, pred="v", dropout=0.1):
        super().__init__()
        assert pred in ("eps", "v", "rf")
        self.pred, self.tdim, self.n_styles = pred, tdim, n_styles
        self.cfg = dict(n_styles=n_styles, base=base, tdim=tdim, cdim=cdim, pred=pred, dropout=dropout)
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # 자모 칸마다 따로 임베딩한 뒤 이어 붙입니다 (더하면 'ㄱ이 초성인지 종성인지'가 섞입니다).
        self.e_cho = nn.Embedding(H.N_CHO + 1, cdim)
        self.e_jung = nn.Embedding(H.N_JUNG + 1, cdim)
        self.e_jong = nn.Embedding(H.N_JONG + 1, cdim)
        self.e_extra = nn.Embedding(H.N_EXTRA + 1, cdim)
        self.cmlp = nn.Sequential(nn.Linear(4 * cdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.e_style = nn.Embedding(n_styles + 1, tdim)

        c1, c2, c3, c4 = base, base * 2, base * 3, base * 4
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, tdim, dropout)                    # 64
        self.ds1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = ResBlock(c1, c2, tdim, dropout)                    # 32
        self.ds2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.d3 = ResBlock(c2, c3, tdim, dropout); self.a3 = Attn(c3)   # 16
        self.ds3 = nn.Conv2d(c3, c3, 3, stride=2, padding=1)
        self.m1 = ResBlock(c3, c4, tdim, dropout); self.ma = Attn(c4)   # 8
        self.m2 = ResBlock(c4, c3, tdim, dropout)
        self.u3 = ResBlock(c3 + c3, c2, tdim, dropout); self.ua3 = Attn(c2)  # 16
        self.u2 = ResBlock(c2 + c2, c1, tdim, dropout)               # 32
        self.u1 = ResBlock(c1 + c1, c1, tdim, dropout)               # 64
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    # ------------------------------------------------------------ '없음' 토큰
    @property
    def NULL_STYLE(self):
        return self.n_styles

    def null_cond(self, B, device):
        return torch.tensor(H.NULL_COND, device=device).expand(B, 4)

    def null_style(self, B, device):
        return torch.full((B,), self.NULL_STYLE, device=device, dtype=torch.long)

    # ------------------------------------------------------------
    def forward(self, x, t, cond=None, style=None):
        B = x.shape[0]
        if cond is None:
            cond = self.null_cond(B, x.device)
        if style is None:
            style = self.null_style(B, x.device)
        c = torch.cat([self.e_cho(cond[:, 0]), self.e_jung(cond[:, 1]),
                       self.e_jong(cond[:, 2]), self.e_extra(cond[:, 3])], -1)
        temb = self.tmlp(timestep_embedding(t, self.tdim)) + self.cmlp(c) + self.e_style(style)

        h1 = self.d1(self.inp(x), temb)
        h2 = self.d2(self.ds1(h1), temb)
        h3 = self.a3(self.d3(self.ds2(h2), temb))
        m = self.m2(self.ma(self.m1(self.ds3(h3), temb)), temb)
        u = F.interpolate(m, scale_factor=2, mode="nearest")
        u = self.ua3(self.u3(torch.cat([u, h3], 1), temb))
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), temb)
        return self.out(u)

    # ------------------------------------------------------------ 파인튜닝용
    def resize_styles(self, n_new):
        """스타일 임베딩을 n_new 개로 바꿉니다 (사전학습 → 나눔손글씨 파인튜닝).
        새 스타일은 기존 스타일 평균에서 시작하고, '없음' 토큰은 그대로 옮깁니다."""
        old = self.e_style.weight.data
        new = nn.Embedding(n_new + 1, old.shape[1]).to(old.device)
        new.weight.data[:n_new] = old[:-1].mean(0, keepdim=True) + 0.01 * torch.randn_like(new.weight.data[:n_new])
        new.weight.data[n_new] = old[-1]
        self.e_style, self.n_styles = new, n_new
        self.cfg["n_styles"] = n_new


# ---------------------------------------------------------------- 출력 변환
def split_pred(out, x, t, alpha_bar, pred):
    """모델 출력 → (eps, x0). diffusion.ddim_step 은 eps 를 받으므로 여기서 맞춰 줍니다.

    v = sqrt(ab) * eps - sqrt(1-ab) * x0    (Salimans & Ho, 2022)
    """
    a = alpha_bar[t].view(-1, 1, 1, 1) if torch.is_tensor(t) else alpha_bar[t]
    sa, sb = a.sqrt(), (1 - a).sqrt()
    if pred == "eps":
        eps = out
        x0 = (x - sb * eps) / sa
    else:
        x0 = sa * x - sb * out
        eps = sb * x + sa * out
    return eps, x0


def target(x0, eps, t, alpha_bar, pred):
    """학습 타깃. pred='eps' 면 Task 2 와 똑같이 eps 입니다."""
    if pred == "eps":
        return eps
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * eps - (1 - a).sqrt() * x0


# ---------------------------------------------------------------- Rectified Flow
def rf_tau(t, T):
    """정수 타임스텝 → RF 시간 tau (0 = 노이즈, 1 = 글자)."""
    return 1.0 - t / (T - 1)


def rf_steps(T, n):
    """오일러 n 스텝의 시작 타임스텝 목록. n=1 → [T-1] (노이즈에서 한 번에 글자로)."""
    return [round((T - 1) * (1 - i / n)) for i in range(n)]


def build(cfg):
    return HangulUNet(**cfg)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    m = HangulUNet(n_styles=421)
    print(f"HangulUNet {sum(p.numel() for p in m.parameters()):,} 파라미터")
    x = torch.randn(4, 1, 64, 64)
    t = torch.randint(0, 1000, (4,))
    cond = torch.tensor([[0, 0, 4, H.NULL_EXTRA]] * 4)
    print("shape:", m(x, t, cond, torch.tensor([0, 1, 2, 3])).shape, m(x, t).shape)
