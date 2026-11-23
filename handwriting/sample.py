# -*- coding: utf-8 -*-
"""샘플링. Task 2 의 sample_loop 를 '자모 + 스타일' 조건으로 바꾼 것입니다.

    python sample.py --ckpt runs/ft_v_seed0.pt --text "안녕하세요, 반가워요!" --style 0 --steps 50 --w 1.5
    python sample.py --ckpt runs/distill/student_1step.pt --text "오늘 날씨 좋다 :)" --style 3

증류된 1스텝 모델은 체크포인트에 자기 타임스텝 목록(steps)이 들어 있어서 --steps 를 무시합니다.
"""
import argparse
import os
import time

import numpy as np
import torch

import hangul as H
import models as M
from diffusion import make_schedule, ddim_step, timestep_schedule


def sample_loop(model, shape, alpha_bar, T, n_steps, cond=None, style=None, w=0.0,
                seed=0, device="cpu", steps=None, x_init=None):
    """노이즈에서 시작해 샘플을 만듭니다. Task 2 의 sample_loop 와 같은 구조입니다.

    cond  : (B, 4) [cho, jung, jong, extra]   None 이면 무조건부
    style : (B,)   스타일 인덱스              None 이면 '스타일 없음'
    w     : classifier-free guidance 세기. eps = ec + w * (ec - eu)
            eu 는 조건(자모·스타일)을 모두 '없음'으로 둔 예측입니다.
    steps : 타임스텝 목록을 직접 줄 때 (증류 모델). None 이면 기본 목록 (make_steps).

    pred='rf' 모델은 같은 루프에서 DDIM 대신 오일러 스텝을 씁니다:
        x <- x + (tau_next - tau) * u,   u = 모델이 예측한 속도 (guidance 도 같은 식으로 섞음)
    """
    with torch.no_grad():
        if x_init is None:
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(shape, device=device, generator=g)
        else:
            x = x_init
        steps = steps if steps is not None else make_steps(model, T, n_steps)
        B = len(x)
        rf = model.pred == "rf"
        for i, t in enumerate(steps):
            tt = torch.full((B,), t, device=device, dtype=torch.long)
            if cond is None and style is None:
                out = model(x, tt)
            elif w == 0:
                out = model(x, tt, cond, style)
            else:
                oc = model(x, tt, cond, style)
                ou = model(x, tt, model.null_cond(B, device), model.null_style(B, device))
                out = oc + w * (oc - ou)          # 출력이 v 여도 eps 와 선형 관계라 똑같이 섞으면 됩니다
            s = steps[i + 1] if i + 1 < len(steps) else None
            if rf:
                tau, tau_next = M.rf_tau(t, T), (M.rf_tau(s, T) if s is not None else 1.0)
                x = x + (tau_next - tau) * out
                if s is None:
                    x = x.clamp(-1, 1)
                continue
            eps, _ = M.split_pred(out, x, tt, alpha_bar, model.pred)
            x = ddim_step(x, eps, t, s, alpha_bar)
        return x


def make_steps(model, T, n):
    """n 스텝짜리 기본 타임스텝 목록. 디퓨전은 DDIM 목록, RF 는 오일러 목록."""
    return M.rf_steps(T, n) if model.pred == "rf" else timestep_schedule(T, n)


def method_of(c):
    """체크포인트 → 방법 이름: diffusion · pd(progressive distillation) · rf · reflow"""
    if c.get("method"):
        return c["method"]
    pred = c["cfg"].get("pred", "v")
    if c.get("steps") is not None:
        return "pd"
    return "rf" if pred == "rf" else "diffusion"


# ================================================================ 문장 → 이미지
def to_uint8(x):
    """[-1,1] (B,1,64,64) → uint8 (B,64,64), 흰 글자 / 검은 배경 (데이터셋과 같은 극성)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()


def compose_line(tiles, gap=3, space=22, thresh=30):
    """글자 이미지들을 한 줄로. 잉크 좌우 끝을 잘라 붙이고, 띄어쓰기는 빈 칸.
    세로는 자르지 않습니다 (문장부호의 높이가 데이터에 이미 들어 있습니다)."""
    parts = []
    for t in tiles:
        if t is None:
            parts.append(np.zeros((64, space), np.uint8)); continue
        cols = np.where(t.max(0) > thresh)[0]
        t = t[:, cols.min():cols.max() + 1] if len(cols) else t[:, :12]
        parts.append(np.pad(t, ((0, 0), (gap, gap))))
    return np.concatenate(parts, 1) if parts else np.zeros((64, 1), np.uint8)


def render_sentence(model, text, style, alpha_bar, T, n_steps=50, w=1.5, seed=0,
                    device="cpu", steps=None):
    """문장 하나를 한 번의 배치로 생성합니다. 모든 글자가 같은 스타일을 씁니다.

    반환: items = [(글자, uint8 이미지 또는 None)], line = 이어 붙인 uint8 이미지, sec = 생성 시간
    지원하지 않는 글자는 None (화면에는 빈 칸으로 나옵니다).
    """
    items = H.parse(text)
    conds = [c for _, c in items if c is not None]
    imgs = []
    t0 = time.time()
    if conds:
        cond = torch.tensor(conds, device=device)
        st = torch.full((len(conds),), int(style), device=device, dtype=torch.long)
        x = sample_loop(model, (len(conds), 1, 64, 64), alpha_bar, T, n_steps, cond, st, w,
                        seed=seed, device=device, steps=steps)
        imgs = list(to_uint8(x))
    if device == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t0
    out, k = [], 0
    for ch, c in items:
        if c is None:
            out.append((ch, None))
        else:
            out.append((ch, imgs[k])); k += 1
    line = compose_line([im for _, im in out])
    return out, line, sec


# ================================================================ 체크포인트
def load(ckpt, device):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    model = M.build(c["cfg"]).to(device)
    model.load_state_dict(c.get("ema") or c["model"])     # EMA 가중치가 있으면 그걸 씁니다
    model.eval()
    return model, c


def find_style(c, style):
    """--style 에 숫자나 폰트 이름 일부를 받습니다."""
    names = c.get("style_names", [])
    if str(style).isdigit():
        return int(style)
    hits = [i for i, n in enumerate(names) if str(style) in n]
    if not hits:
        raise SystemExit(f"'{style}' 이름이 들어간 스타일이 없습니다.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="안녕하세요, 반가워요! (Hi 2026) “좋아” ♡")
    ap.add_argument("--style", default="0", help="스타일 번호 또는 폰트 이름 일부")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/samples")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, c = load(a.ckpt, device)
    _, alpha_bar = make_schedule(c["T"], c["schedule"], device)
    steps = c.get("steps")                          # 증류 모델이면 고정된 타임스텝 목록
    w = 0.0 if steps else a.w                       # 증류 모델은 guidance 가 이미 구워져 있음 (c["w_baked"])
    sidx = find_style(c, a.style)
    render_sentence(model, a.text[:2], sidx, alpha_bar, c["T"], a.steps, w, a.seed, device, steps)  # 워밍업
    items, line, sec = render_sentence(model, a.text, sidx, alpha_bar, c["T"], a.steps, w, a.seed, device, steps)

    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    nstep = len(steps) if steps else a.steps
    path = os.path.join(a.out, f"line_style{sidx}_s{nstep}_w{w:g}_seed{a.seed}.png")
    Image.fromarray(255 - line).save(path)
    n = sum(im is not None for _, im in items)
    name = c.get("style_names", ["?"] * (sidx + 1))[sidx]
    wtxt = f"w={c['w_baked']} (증류 때 구워짐)" if steps else f"w={w}"
    print(f"스타일 {sidx} ({name}), {n}글자, {nstep}스텝, {wtxt}  ->  {path}")
    print(f"문장 하나 생성 {sec * 1000:.0f} ms  (글자당 {sec * 1000 / max(n, 1):.1f} ms, {device})")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
