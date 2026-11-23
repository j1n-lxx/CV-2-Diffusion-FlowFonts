# -*- coding: utf-8 -*-
"""1스텝 증류: progressive distillation (Salimans & Ho, 2022).

teacher 의 DDIM 두 스텝을 student 의 한 스텝으로 맞추는 라운드를 반복해서
64 → 32 → 16 → 8 → 4 → 2 → 1 스텝으로 줄입니다.

    python distill.py --teacher runs/ft_v_seed0.pt --data font_ko_v2 --w 1.5 --iters 4000

  - 첫 라운드 teacher 는 guidance(w)를 섞어서 씁니다. student 는 조건부 한 번만 돌리므로
    guidance 가 student 안에 '구워져' 들어갑니다 → 데모 때 forward 1번 = 1스텝.
  - 모델은 pred='v' 여야 합니다 (train.py 기본값). eps 예측은 t≈T 에서 x0 를 못 만들어서
    마지막 1스텝 라운드가 실패합니다.
  - 라운드마다 student 체크포인트를 runs/distill/student_{N}step.pt 로 저장합니다.
    4스텝 모델은 1스텝이 흐릴 때를 대비한 예비용입니다.

데이터는 파인튜닝 세트(ft) train 을 씁니다. teacher 가 사전학습만 된 모델이면 --which pre.

여러 GPU 병렬:  torchrun --standalone --nproc_per_node=4 distill.py --teacher ... (--bs 는 GPU 한 장당)
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn.functional as F

import data as D
import models as M
from diffusion import make_schedule, q_sample, ddim_step
from dist_utils import Dist


def grid(T, N):
    """N 스텝짜리 타임스텝 목록. 절반으로 줄일 때 짝수 번째만 남으면 되도록 중첩되게 만듭니다.
    t_k = round((T-1) * k / N),  k = N, N-1, ..., 1   (마지막 스텝 다음은 x0)"""
    return [round((T - 1) * k / N) for k in range(N, 0, -1)]


def _eps(model, x, t, cond, style, alpha_bar, w):
    B = len(x)
    tt = torch.full((B,), t, device=x.device, dtype=torch.long) if isinstance(t, int) else t
    out = model(x, tt, cond, style)
    if w:
        ou = model(x, tt, model.null_cond(B, x.device), model.null_style(B, x.device))
        out = out + w * (out - ou)
    return M.split_pred(out, x, tt, alpha_bar, model.pred)[0]


def distill_target(teacher, z, k, N, T, cond, style, alpha_bar, w):
    """student 가 z(t_k) 에서 한 스텝으로 가야 할 곳을 teacher 두 스텝으로 만들고,
    그 결과를 student 기준 v 타깃으로 바꿔 돌려줍니다.

    student 의 스텝:  t_k → t_{k-2}   (k-2 == 0 이면 x0)
    teacher 의 스텝:  t_k → t_{k-1} → t_{k-2}
    """
    tk, t1 = round((T - 1) * k / N), round((T - 1) * (k - 1) / N)
    t2 = round((T - 1) * (k - 2) / N) if k - 2 > 0 else None
    with torch.no_grad():
        z1 = ddim_step(z, _eps(teacher, z, tk, cond, style, alpha_bar, w), tk, t1 if k - 1 > 0 else None, alpha_bar)
        if k - 1 == 0:                      # (k=1 은 쓰지 않지만 방어용)
            x_tgt = z1
        else:
            z2 = ddim_step(z1, _eps(teacher, z1, t1, cond, style, alpha_bar, w), t1, t2, alpha_bar)
            if t2 is None:
                x_tgt = z2                  # teacher 의 두 번째 스텝이 바로 x0
            else:
                a_t, a_2 = alpha_bar[tk], alpha_bar[t2]
                st, s2 = (1 - a_t).sqrt(), (1 - a_2).sqrt()
                # DDIM 한 스텝  z2 = a2^.5 x + s2 (z - a_t^.5 x)/st  을 x 에 대해 푼 것
                x_tgt = (z2 - (s2 / st) * z) / (a_2.sqrt() - (s2 / st) * a_t.sqrt())
        a_t = alpha_bar[tk]
        eps_tgt = (z - a_t.sqrt() * x_tgt) / (1 - a_t).sqrt()
        v_tgt = a_t.sqrt() * eps_tgt - (1 - a_t).sqrt() * x_tgt
    return tk, v_tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", default="font_ko_v2")
    ap.add_argument("--which", default="ft", choices=["ft", "pre"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--start", type=int, default=64, help="첫 teacher 스텝 수 (2의 거듭제곱)")
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--w", type=float, default=1.5, help="첫 라운드 teacher 의 guidance (student 에 구워짐)")
    ap.add_argument("--iters", type=int, default=5000,
                    help="라운드당 학습 스텝. 기본 5000 × 6라운드 = 30,000 (reflow 20,000 + 1스텝 증류 10,000 과 같은 학생 학습량)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true", help="CUDA bf16 mixed precision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/distill")
    a = ap.parse_args()

    dd = Dist()
    device = dd.device
    say = print if dd.main else (lambda *k, **kw: None)
    torch.manual_seed(a.seed + 1000 * dd.rank)    # 프로세스마다 다른 노이즈
    os.makedirs(a.out, exist_ok=True)
    c = torch.load(a.teacher, map_location="cpu", weights_only=False)
    teacher = M.build(c["cfg"]); teacher.load_state_dict(c.get("ema") or c["model"])
    teacher = teacher.to(device).eval().requires_grad_(False)
    assert teacher.pred == "v", ("증류하려면 pred='v' 로 학습한 모델이 필요합니다 (train.py --pred v). "
                                 "RF 모델(pred='rf')은 reflow.py 로 1스텝을 만드세요.")
    T = c["T"]
    _, alpha_bar = make_schedule(T, c["schedule"], device)
    ds = D.load(a.data, a.which, a.split, seed=a.seed, in_memory=not dd.enabled)
    assert teacher.n_styles == D.n_styles(a.data, a.which), "teacher 와 데이터 세트(--which)의 스타일 수가 다릅니다."
    say(f"device={device}  GPU/프로세스 {dd.world}개  bs={a.bs}×{dd.world}  teacher={a.teacher}  "
        f"데이터 {len(ds):,}장  w={a.w}")

    N, log, w = a.start, [], a.w
    is_cuda = str(device).startswith("cuda")
    cost = {"student_iters": 0, "teacher_nfe": 0, "train_minutes": 0.0, "teacher_steps": a.start}   # 실험 비교용 계산량
    g = torch.Generator().manual_seed(a.seed + dd.rank)   # 프로세스마다 다른 배치·다른 시작 스텝
    while N > a.end:
        Ns = N // 2
        student = copy.deepcopy(teacher).train().requires_grad_(True)
        ema = copy.deepcopy(teacher).eval()
        net = dd.wrap(student)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
        t0, run = time.time(), []
        for it in range(a.iters):
            idx = torch.randint(0, len(ds), (a.bs,), generator=g)
            x0, st, cond = ds.batch(idx, device)
            # student 의 스텝 시작점: 짝수 k (N 스텝 grid 기준) 중 무작위
            k = 2 * int(torch.randint(1, Ns + 1, (1,), generator=g))
            tk = round((T - 1) * k / N)
            z = q_sample(x0, torch.full((a.bs,), tk, device=device, dtype=torch.long), torch.randn_like(x0), alpha_bar)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and is_cuda):
                _, v_tgt = distill_target(teacher, z, k, N, T, cond, st, alpha_bar, w)
                out = net(z, torch.full((a.bs,), tk, device=device, dtype=torch.long), cond, st)
                loss = F.mse_loss(out.float(), v_tgt.float())
            cost["teacher_nfe"] += a.bs * dd.world * 2 * (2 if w > 0 else 1)     # 교사 DDIM 2스텝 (+ guidance)
            opt.zero_grad(set_to_none=True); loss.backward()     # 병렬이면 gradient 평균
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            with torch.no_grad():
                d = min(a.ema, (1 + it) / (10 + it))
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.mul_(d).add_(ps.detach(), alpha=1 - d)
            run.append(loss.item())
            if (it + 1) % 500 == 0 or it + 1 == a.iters:
                m = dd.mean(sum(run[-500:]) / len(run[-500:]))
                say(f"  [{N}→{Ns}스텝] iter {it + 1:>6}/{a.iters}  loss {m:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        log.append({"teacher_steps": N, "student_steps": Ns, "final_loss": m, "minutes": (time.time() - t0) / 60})
        cost["student_iters"] += a.iters
        cost["train_minutes"] += (time.time() - t0) / 60
        path = os.path.join(a.out, f"student_{Ns}step.pt")
        if dd.main:
            torch.save({"model": student.state_dict(), "ema": ema.state_dict(), "cfg": teacher.cfg, "T": T,
                        "schedule": c["schedule"], "steps": grid(T, Ns), "w_baked": a.w, "method": "pd",
                        "cost": dict(cost), "stage": c.get("stage"), "teacher_method": "diffusion",
                        "seed": c.get("seed", a.seed),
                        "style_names": c.get("style_names", []), "teacher": a.teacher, "args": vars(a)}, path)
            print(f"  -> {path}")
        dd.barrier()
        # 다음 라운드 teacher = 이번 student 의 EMA (가중치가 모든 프로세스에서 같으므로 EMA 도 같음)
        teacher, N, w = ema.requires_grad_(False), Ns, 0.0     # 다음 라운드부터 guidance 는 이미 구워져 있음
    if dd.main:
        json.dump(log, open(os.path.join(a.out, "distill_log.json"), "w"), indent=2)
        print("\n완료. 1스텝·4스텝을 eval.py --ckpt runs/distill/student_1step.pt 로 비교하세요.")
    dd.close()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
