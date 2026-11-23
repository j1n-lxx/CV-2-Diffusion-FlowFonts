# -*- coding: utf-8 -*-
"""학습. Task 2 의 train.py 와 같은 구조에 단계(stage) 세 가지를 더했습니다.

    # (b) 사전학습 → 파인튜닝  (기본 경로)
    python train.py --stage pre --data font_ko_v2 --epochs 30
    python train.py --stage ft  --data font_ko_v2 --epochs 60 --init runs/pre_v_seed0.pt

    # (a) 나눔손글씨만, 처음부터
    python train.py --stage ft  --data font_ko_v2 --epochs 60
    # (c) 혼합 학습
    python train.py --stage mix --data font_ko_v2 --epochs 30

    # Rectified Flow 로 처음부터 (디퓨전 대신 직선 경로). 1스텝은 이후 reflow.py
    python train.py --stage pre --objective rf --data font_ko_v2 --epochs 30
    python train.py --stage ft  --data font_ko_v2 --epochs 60 --init runs/pre_rf_seed0.pt

    # 파인튜닝 데이터 10%·1% 실험:  --frac 0.1 / --frac 0.01
    # 사전학습에서 나눔손글씨 109종 빼기:  --exclude_nanum

학습은 GPU 에서 하세요. 64×64 이고 사전학습 세트가 74만 장이라 CPU 로는 현실적이지 않습니다.
처음에는 --split dev 로 몇 분 돌려 파이프라인부터 확인하세요.

여러 GPU 병렬 (DDP):
    torchrun --standalone --nproc_per_node=4 train.py --stage pre --data font_ko_v2 --epochs 10 --amp
  --bs 는 GPU 한 장당 배치입니다 (전체 배치 = bs × GPU 수). 한 epoch 의 스텝 수는 GPU 수만큼 줄어듭니다.
  병렬일 때는 이미지를 RAM 에 복사하지 않고 memmap 으로 읽어서, 프로세스들이 OS 페이지 캐시를 같이 씁니다.
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import data as D
import models as M
from diffusion import make_schedule, q_sample, check_data
from dist_utils import Dist, unwrap
from augment import augment
import hangul as H


# ================================================================ 한 스텝 (Task 2 TODO 1 과 같은 다섯 줄)
def training_step(model, x0, cond, style, alpha_bar, T, p_uncond=0.1, p_style=0.1):
    """배치 하나의 손실.

      1. t ~ U[0, T)
      2. eps ~ N(0, I)
      3. x_t = q_sample(x0, t, eps, alpha_bar)
      4. 예측 = model(x_t, t, cond, style)        (조건을 일부 가린 뒤)
      5. MSE(예측, 타깃)                            타깃은 pred='eps' 면 eps, 'v' 면 v

    pred='rf' (Rectified Flow) 이면 3·5 번만 바뀝니다:
      3. tau = 1 - t/(T-1),  x_t = tau * x0 + (1 - tau) * eps     (노이즈와 글자를 잇는 직선 위의 점)
      5. MSE(예측, x0 - eps)                                       (직선의 방향 = 속도)

    가리기 (classifier-free guidance 준비):
      p_uncond 확률로 자모·스타일을 둘 다 '없음'으로  →  무조건부 예측 eu 를 배움
      p_style  확률로 스타일만 '없음'으로            →  '어떤 필체든 이 글자' 를 배움
                                                       (파인튜닝 때 새 스타일이 기존 글자 지식을 빌려 씀)
    """
    m = unwrap(model)              # 병렬(DDP)일 때 '없음' 토큰·pred 는 안쪽 모델에서 꺼냅니다
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=x0.device, dtype=torch.long)
    eps = torch.randn_like(x0)
    if m.pred == "rf":
        tau = M.rf_tau(t.float(), T).view(B, 1, 1, 1)
        x_t = tau * x0 + (1 - tau) * eps
    else:
        x_t = q_sample(x0, t, eps, alpha_bar)

    r = torch.rand(B, device=x0.device)
    drop_all = r < p_uncond
    drop_style = (r >= p_uncond) & (r < p_uncond + p_style)
    cond = torch.where(drop_all[:, None], m.null_cond(B, x0.device), cond)
    style = torch.where(drop_all | drop_style, m.null_style(B, x0.device), style)

    out = model(x_t, t, cond, style)
    if m.pred == "rf":
        return F.mse_loss(out, x0 - eps)
    return F.mse_loss(out, M.target(x0, eps, t, alpha_bar, m.pred))


# ================================================================ 학습 루프
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="pre", choices=["pre", "ft", "mix"])
    ap.add_argument("--data", default="font_ko_v2", help="font_ko_v2.zip 을 푼 폴더")
    ap.add_argument("--split", default="train", help="train (기본) 또는 dev (빠른 확인)")
    ap.add_argument("--init", default=None, help="ft: 사전학습 체크포인트에서 이어 학습")
    ap.add_argument("--exclude_nanum", action="store_true", help="pre/mix: 사전학습 세트에서 나눔손글씨 109종 제외")
    ap.add_argument("--frac", type=float, default=1.0, help="ft/mix: 나눔손글씨 글자 비율 (1.0, 0.1, 0.01)")
    ap.add_argument("--pred", default="v", choices=["v", "eps", "rf"])
    ap.add_argument("--objective", default=None, choices=["diffusion", "rf"],
                    help="rf = Rectified Flow 로 학습 (--pred rf 와 같음). 기본: 디퓨전 (pred v)")
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--epochs", type=float, default=10, help="소수도 됩니다 (0.2 = 데이터의 20%만큼)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="기본: 처음부터 2e-4, --init 이면 1e-4")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--p_style", type=float, default=0.1)
    ap.add_argument("--aug", type=float, default=None,
                    help="증강 세기 (0=끔). 기본: pre 0.5, ft 1.0, mix 0.75 — augment.py 참고")
    ap.add_argument("--aug_p", type=float, default=0.5, help="샘플마다 증강을 걸 확률")
    ap.add_argument("--extra_weight", type=float, default=4.0,
                    help="숫자·영문·문장부호를 한글보다 몇 배 자주 뽑을지 (1=데이터 비율 그대로)")
    ap.add_argument("--exclude_file", default="exclude_fonts.txt",
                    help="사전학습에서 뺄 폰트 목록 (픽셀·장식·split 누출 등). 끄려면 none")
    ap.add_argument("--grid_every", type=int, default=2000, help="몇 스텝마다 고정 시드 문장 그림을 저장할지")
    ap.add_argument("--no_memory", action="store_true", help="이미지를 RAM 에 올리지 않고 디스크(memmap)에서 읽기")
    ap.add_argument("--amp", action="store_true", help="CUDA bf16 mixed precision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    if a.objective == "rf":
        a.pred = "rf"
    if a.init:                                    # 이어 학습: 방식(pred)은 init 모델을 따름 (파일 이름도)
        init_pred = torch.load(a.init, map_location="cpu", weights_only=False)["cfg"].get("pred", "v")
        if a.objective and (a.objective == "rf") != (init_pred == "rf"):
            raise SystemExit(f"--objective {a.objective} 인데 --init 모델은 pred={init_pred} 입니다. 같은 방식으로 이어 학습하세요.")
        a.pred = init_pred
    dd = Dist()                                   # 병렬이면 torchrun 이 준 환경변수로 초기화
    device = dd.device
    say = print if dd.main else (lambda *k, **kw: None)   # 로그·저장은 0번 프로세스만
    torch.manual_seed(a.seed)                     # 모든 프로세스가 같은 초기 가중치·같은 순서로 시작
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.stage}_{a.pred}_seed{a.seed}"
    if a.frac < 1:
        tag += f"_frac{a.frac:g}"
    if a.stage == "ft" and not a.init:
        tag += "_scratch"
    if a.exclude_nanum:
        tag += "_nonanum"
    if a.split != "train":
        tag += f"_{a.split}"

    # ---- 데이터 (병렬이면 memmap: 프로세스마다 3GB 씩 복사하지 않고 OS 페이지 캐시를 공유)
    mem = not a.no_memory and not dd.enabled
    excl = D.read_exclude(a.exclude_file) | (D.nanum_font_ids(a.data) if a.exclude_nanum else set())
    if a.aug is None:
        a.aug = {"pre": 0.5, "ft": 1.0, "mix": 0.75}[a.stage]
    if a.stage == "pre":
        ds = D.load(a.data, "pre", a.split, exclude_fonts=excl, in_memory=mem, seed=a.seed)
        n_styles = D.n_styles(a.data, "pre")
    elif a.stage == "ft":
        ds = D.load(a.data, "ft", a.split, frac=a.frac, in_memory=mem, seed=a.seed)
        n_styles = D.n_styles(a.data, "ft")
    else:
        n_pre = D.n_styles(a.data, "pre")
        pre = D.load(a.data, "pre", a.split, exclude_fonts=excl, seed=a.seed)
        ft = D.load(a.data, "ft", a.split, frac=a.frac, style_offset=n_pre, seed=a.seed)
        ds = D.concat(pre, ft)
        n_styles = n_pre + D.n_styles(a.data, "ft")

    # ---- 모델
    if a.init:
        c = torch.load(a.init, map_location="cpu", weights_only=False)
        model = M.build(c["cfg"])
        model.load_state_dict(c.get("ema") or c["model"])
        assert c["T"] == a.T and c["schedule"] == a.schedule, "사전학습과 같은 T·schedule 을 쓰세요."
        if model.n_styles != n_styles:
            model.resize_styles(n_styles)
        a.pred = model.pred
        say(f"초기화: {a.init}  (pred={model.pred}, 스타일 {n_styles}개로 맞춤)")
    else:
        torch.manual_seed(a.seed)                 # resize 등의 난수와 무관하게 모든 프로세스가 같은 초기값
        model = M.HangulUNet(n_styles=n_styles, base=a.base, pred=a.pred)
    model = model.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    net = dd.wrap(model)                          # 학습 forward/backward 는 net, 저장·샘플링은 model/ema
    lr = a.lr or (1e-4 if a.init else 2e-4)

    xb, _, _ = ds.batch(torch.arange(min(256, len(ds))), device)
    check_data(xb, "학습 데이터")
    n_par = sum(p.numel() for p in model.parameters())
    say(f"device={device}  GPU/프로세스 {dd.world}개  stage={a.stage}  pred={model.pred}  T={a.T}  "
        f"schedule={a.schedule}  bs={a.bs}×{dd.world}={a.bs * dd.world}  lr={lr}  seed={a.seed}")
    say(f"데이터 {len(ds):,}장  스타일 {n_styles}개  파라미터 {n_par:,}")
    n_extra = int((ds.char_id >= H.N_HANGUL_KS).sum())
    say(f"증강 세기 {a.aug} (확률 {a.aug_p})  추가 문자 {n_extra:,}장 ×{a.extra_weight:g}  "
        f"제외 폰트 {len(excl) if a.stage != 'ft' else 0}개")

    _, alpha_bar = make_schedule(a.T, a.schedule, device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    gbs = a.bs * dd.world                         # 전체 배치
    per_epoch = max(1, len(ds) // gbs)
    total = max(1, int(a.epochs * per_epoch))
    warm = min(1000, total // 20)
    # warmup 후 cosine 감쇠
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    # 고정 시드 문장 그림에 쓸 스타일: 학습 데이터에 있는 스타일 4개
    shown = sorted(set(ds.style[torch.linspace(0, len(ds) - 1, 4).long()].tolist()))
    style_names = ds.style_names

    losses, log, t0, step = [], [], time.time(), 0
    g = torch.Generator().manual_seed(a.seed)     # 모든 프로세스가 같은 순열을 만들고 자기 몫만 가져감
    torch.manual_seed(a.seed + 1000 * dd.rank)    # t·노이즈·가리기 난수는 프로세스마다 다르게
    net.train()
    is_cuda = str(device).startswith("cuda")
    # 추가 문자(숫자·영문·문장부호)는 4% 정도라 데모 문장에 자주 나오는 쉼표·마침표가 약해집니다 → 더 자주 뽑기
    probs = None
    if a.extra_weight != 1.0:
        probs = torch.ones(len(ds)) + (a.extra_weight - 1.0) * (ds.char_id >= H.N_HANGUL_KS).float()
    while step < total:
        if probs is None:
            perm = torch.randperm(len(ds), generator=g)
        else:   # 복원 추출 (모든 프로세스가 같은 생성기라 같은 순서)
            perm = torch.multinomial(probs, len(ds), replacement=True, generator=g)
        for b in range(per_epoch):
            if step >= total:
                break
            lo = b * gbs + dd.rank * a.bs
            idx = perm[lo:lo + a.bs]
            x0, st, cond = ds.batch(idx, device)
            if a.aug > 0:
                x0 = augment(x0, a.aug, a.aug_p)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and is_cuda):
                loss = training_step(net, x0, cond, st, alpha_bar, a.T, a.p_uncond, a.p_style)
            opt.zero_grad(set_to_none=True)
            loss.backward()                       # 병렬이면 여기서 GPU 들의 gradient 가 평균됩니다
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():                 # 가중치가 모든 프로세스에서 같으므로 EMA 도 같음
                d = min(a.ema, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            losses.append(loss.item()); step += 1
            if step % 200 == 0 or step == total:
                m = dd.mean(sum(losses[-200:]) / len(losses[-200:]))
                log.append({"step": step, "loss": m, "sec": time.time() - t0})
                say(f"  step {step:>7}/{total}  ep {step / per_epoch:5.2f}  loss {m:.4f}  "
                    f"({time.time() - t0:.0f}s)", flush=True)
            if step % a.grid_every == 0 or step == total:
                if dd.main:
                    _save_grid(ema, alpha_bar, a, step, shown, device, tag)
                    _save_ckpt(a, model, ema, style_names, tag, step, log, n_par, t0, final=False)
                dd.barrier()

    if dd.main:
        path = _save_ckpt(a, model, ema, style_names, tag, step, log, n_par, t0, final=True)
        print(f"\n{(time.time() - t0) / 60:.1f}분  ->  {path}")
        print("손실만 보지 말고 runs/grids 의 문장 그림을 보세요.")
    dd.close()


def _save_ckpt(a, model, ema, style_names, tag, step, log, n_par, t0, final):
    path = os.path.join(a.out, f"{tag}.pt" if final else f"{tag}_last.pt")
    # GPU 개수와 무관하게 같은 형식 (DDP 래퍼가 아닌 안쪽 모델을 저장) → 1장에서 그대로 불러올 수 있음
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "cfg": model.cfg, "T": a.T,
                "schedule": a.schedule, "stage": a.stage, "seed": a.seed, "step": step,
                "style_names": style_names, "args": vars(a)}, path)
    if final:
        json.dump({"log": log, "train_minutes": (time.time() - t0) / 60, "n_params": n_par, "args": vars(a)},
                  open(os.path.join(a.out, f"{tag}_train.json"), "w"), indent=2, ensure_ascii=False)
        last = os.path.join(a.out, f"{tag}_last.pt")
        if os.path.exists(last):
            os.remove(last)
    return path


GRID_TEXT = "안녕하세요, 반가워요! (Hi 2026) “좋아” ♡ 똠방각하"


def _save_grid(ema, alpha_bar, a, step, shown, device, tag):
    """고정 시드로 같은 문장을 스타일 4개로. 이게 진짜 손실 곡선입니다 (Task 2 의 epoch 그리드와 같은 역할)."""
    try:
        import numpy as np
        from PIL import Image
        from sample import render_sentence
        lines = [render_sentence(ema, GRID_TEXT, s, alpha_bar, a.T, 50, 1.5, 0, device)[1] for s in shown]
        W = max(l.shape[1] for l in lines)
        img = np.concatenate([np.pad(l, ((2, 2), (0, W - l.shape[1]))) for l in lines], 0)
        d = os.path.join(a.out, "grids", tag)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(255 - img).save(os.path.join(d, f"step{step:07d}.png"))
    except Exception as e:  # 그림 저장 실패가 학습을 멈추게 하지는 않습니다
        print(f"    (그림 저장 건너뜀: {e})")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
