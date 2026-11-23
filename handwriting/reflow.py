# -*- coding: utf-8 -*-
"""1스텝 만들기: Reflow (Rectified Flow, Liu et al. 2023) + 선택적 1스텝 증류 (InstaFlow 방식).

    # RF 로 처음부터 학습한 모델이 교사 (주 경로)
    python reflow.py --teacher runs/ft_rf_seed0.pt --data font_ko_v2 --w 1.5
    # 디퓨전 교사로도 가능 (비교 실험: 같은 쌍 생성 방식, 교사만 다름)
    python reflow.py --teacher runs/ft_v_seed0.pt --data font_ko_v2 --w 1.5 --init scratch
    # reflow 뒤에 1스텝 증류까지
    python reflow.py --teacher runs/ft_rf_seed0.pt --data font_ko_v2 --distill_iters 10000

하는 일
  1) 쌍 만들기: 교사가 노이즈 z 에서 글자 x' 를 끝까지 생성  (z, 자모, 필체) → x'
     - 교사 샘플러: RF 교사는 오일러, 디퓨전 교사는 DDIM. --gen_steps 스텝, guidance --w
     - guidance 는 이 단계에서 결과에 '구워져' 학생은 조건부로 한 번만 돌면 됩니다.
     - 쌍은 runs/reflow/pairs_<교사>_s<스텝>_w<w>/ 에 저장해 두고 다음 실행에서 재사용합니다.
  2) Reflow: 같은 RF 손실로 학생 학습. 단, 노이즈와 글자의 짝이 무작위가 아니라 교사가 정한 짝.
         x_t = tau x' + (1 - tau) z,   정답 속도 = x' - z
     짝이 서로 교차하지 않으므로 학생의 경로가 직선에 가까워지고 → 1스텝 오일러 (x = z + u(z)) 가 됩니다.
  3) (선택) 1스텝 증류: z + u(z, t=T-1) 가 바로 x' 가 되도록 회귀.

결과: runs/reflow/<교사>_reflow.pt  (와 _reflow_distill.pt). 1스텝이 기본이고,
eval.py --steps 1 2 4 로 같은 학생을 2·4스텝 오일러로도 잴 수 있습니다.

여러 GPU: torchrun --standalone --nproc_per_node=4 reflow.py ...  (쌍 생성도 나눠서 합니다)
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data as D
import models as M
from diffusion import make_schedule
from dist_utils import Dist
from sample import load, make_steps, method_of, render_sentence, sample_loop

GRID_TEXT = "안녕하세요, 반가워요! (Hi 2026) “좋아” ♡ 똠방각하"


# ================================================================ 1) 쌍 만들기
def pair_dir(a, teacher_tag):
    return a.pairs or os.path.join(a.out, f"pairs_{teacher_tag}_s{a.gen_steps}_w{a.w:g}_n{a.n_pairs or 'all'}")


def make_pairs(teacher, c, ds, a, dd, device, path, say):
    """교사로 (z, x') 쌍을 만들어 memmap 으로 저장. 여러 GPU 면 배치 묶음을 나눠 맡습니다."""
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path) and json.load(open(meta_path)).get("complete"):
        say(f"쌍 재사용: {path}")
        return
    T = c["T"]
    _, alpha_bar = make_schedule(T, c["schedule"], device)
    fixed = c.get("steps")                                   # 교사가 증류 모델이면 고정 스텝 + guidance 구워짐
    w = 0.0 if fixed else a.w
    steps = fixed or make_steps(teacher, T, a.gen_steps)
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(ds), generator=g)
    N = min(a.n_pairs or len(ds), len(ds))
    order = order[:N]
    if dd.main:
        os.makedirs(path, exist_ok=True)
        np.lib.format.open_memmap(os.path.join(path, "z.npy"), "w+", np.float16, (N, 1, 64, 64))
        np.lib.format.open_memmap(os.path.join(path, "x.npy"), "w+", np.uint8, (N, 64, 64))
        np.lib.format.open_memmap(os.path.join(path, "cond.npy"), "w+", np.int64, (N, 4))
        np.lib.format.open_memmap(os.path.join(path, "style.npy"), "w+", np.int64, (N,))
    dd.barrier()
    Z = np.load(os.path.join(path, "z.npy"), mmap_mode="r+")
    X = np.load(os.path.join(path, "x.npy"), mmap_mode="r+")
    Cc = np.load(os.path.join(path, "cond.npy"), mmap_mode="r+")
    S = np.load(os.path.join(path, "style.npy"), mmap_mode="r+")
    chunks = list(range(0, N, a.gen_bs))
    t0 = time.time()
    mine = chunks[dd.rank::dd.world]
    for j, lo in enumerate(mine):
        idx = order[lo:lo + a.gen_bs]
        _, st, cond = ds.batch(idx, device)
        # 노이즈는 묶음 번호로 정해서 (CPU 생성기) 장치·GPU 수와 무관하게 같은 쌍
        z = torch.randn(len(idx), 1, 64, 64, generator=torch.Generator().manual_seed(a.seed * 1_000_003 + lo))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and str(device).startswith("cuda")):
            x = sample_loop(teacher, z.shape, alpha_bar, T, len(steps), cond, st, w, device=device,
                            steps=steps, x_init=z.to(device))
        x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(1).cpu().numpy()
        sl = slice(lo, lo + len(idx))
        Z[sl] = z.numpy().astype(np.float16); X[sl] = x
        Cc[sl] = cond.cpu().numpy(); S[sl] = st.cpu().numpy()
        if dd.main and (j % 20 == 0 or j == len(mine) - 1):
            done = min(N, (j + 1) * a.gen_bs * dd.world)
            say(f"  쌍 생성 {done:>8,}/{N:,}  ({time.time() - t0:.0f}s)", flush=True)
    for arr in (Z, X, Cc, S):
        arr.flush()
    dd.barrier()
    if dd.main:
        json.dump({"complete": True, "teacher": a.teacher, "teacher_method": method_of(c), "gen_steps": len(steps),
                   "w": w if not fixed else c.get("w_baked"), "n": N, "minutes": (time.time() - t0) / 60,
                   "teacher_forwards_per_pair": len(steps) * (2 if w > 0 else 1)},
                  open(meta_path, "w"), indent=2, ensure_ascii=False)
        say(f"쌍 {N:,}개 완료 ({(time.time() - t0) / 60:.1f}분) → {path}")
    dd.barrier()


class Pairs:
    def __init__(self, path, in_memory):
        ld = (lambda f: np.load(os.path.join(path, f))) if in_memory else \
             (lambda f: np.load(os.path.join(path, f), mmap_mode="r"))
        self.z, self.x, self.cond, self.style = ld("z.npy"), ld("x.npy"), ld("cond.npy"), ld("style.npy")

    def __len__(self):
        return len(self.x)

    def batch(self, idx, device):
        i = np.sort(idx.numpy())
        z = torch.from_numpy(np.asarray(self.z[i], np.float32)).to(device)
        x = torch.from_numpy(np.asarray(self.x[i])).float().div(127.5).sub(1).unsqueeze(1).to(device)
        return z, x, torch.from_numpy(np.asarray(self.cond[i])).to(device), torch.from_numpy(np.asarray(self.style[i])).to(device)


# ================================================================ 2·3) 학생 학습
def reflow_loss(net, m, z, x, cond, style, T):
    B = len(x)
    t = torch.randint(0, T, (B,), device=x.device)
    tau = M.rf_tau(t.float(), T).view(B, 1, 1, 1)
    out = net(tau * x + (1 - tau) * z, t, cond, style)
    return F.mse_loss(out, x - z)


def onestep_loss(net, m, z, x, cond, style, T):
    t = torch.full((len(x),), T - 1, device=x.device, dtype=torch.long)
    return F.mse_loss(z + net(z, t, cond, style), x)


def train_phase(name, loss_fn, student, pairs, a, iters, lr, dd, device, T, say):
    ema = copy.deepcopy(student).eval().requires_grad_(False)
    net = dd.wrap(student.train().requires_grad_(True))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    warm = min(500, iters // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)))
    g = torch.Generator().manual_seed(a.seed + 7 + dd.rank)
    run, log, t0 = [], [], time.time()
    is_cuda = str(device).startswith("cuda")
    for it in range(iters):
        idx = torch.randint(0, len(pairs), (a.bs,), generator=g)
        z, x, cond, st = pairs.batch(idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and is_cuda):
            loss = loss_fn(net, student, z, x, cond, st, T)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            d = min(a.ema, (1 + it) / (10 + it))
            for pe, ps in zip(ema.parameters(), student.parameters()):
                pe.mul_(d).add_(ps.detach(), alpha=1 - d)
        run.append(loss.item())
        if (it + 1) % 200 == 0 or it + 1 == iters:
            mloss = dd.mean(sum(run[-200:]) / len(run[-200:]))
            log.append({"step": it + 1, "loss": mloss, "sec": time.time() - t0})
            say(f"  [{name}] iter {it + 1:>6}/{iters}  loss {mloss:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    return ema, log, (time.time() - t0) / 60


def pair_cost(pairs_dir):
    m = json.load(open(os.path.join(pairs_dir, "meta.json")))
    return {"teacher_nfe": int(m["n"] * m["teacher_forwards_per_pair"]), "pair_minutes": m.get("minutes"),
            "teacher_steps": m["gen_steps"], "n_pairs": m["n"]}


def save(path, student, ema, c, a, method, log, minutes, pairs_dir, say, student_iters=0):
    T = c["T"]
    cost = dict(pair_cost(pairs_dir), student_iters=student_iters, train_minutes=minutes)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"model": student.state_dict(), "ema": ema.state_dict(), "cfg": student.cfg, "T": T,
                "schedule": c["schedule"], "stage": c.get("stage"), "seed": a.seed,
                "steps": M.rf_steps(T, 1), "w_baked": c.get("w_baked") if c.get("steps") else a.w,
                "method": method, "cost": cost, "init": a.init, "teacher": a.teacher, "teacher_method": method_of(c), "pairs": pairs_dir,
                "style_names": c.get("style_names", []), "args": vars(a)}, path)
    tag = os.path.splitext(os.path.basename(path))[0]
    json.dump({"log": log, "train_minutes": minutes, "args": vars(a), "method": method},
              open(os.path.join(a.out, f"{tag}_train.json"), "w"), indent=2, ensure_ascii=False)
    # 1스텝 고정 시드 문장 그림 (train.py 의 grids 와 같은 위치)
    try:
        from PIL import Image
        names = c.get("style_names", [])
        shown = sorted(set(np.linspace(0, max(0, len(names) - 1), 4).round().astype(int).tolist())) or [0]
        dev = next(ema.parameters()).device
        _, ab = make_schedule(T, c["schedule"], dev)
        lines = [render_sentence(ema, GRID_TEXT, s, ab, T, 1, 0.0, 0, dev, M.rf_steps(T, 1))[1] for s in shown]
        W = max(l.shape[1] for l in lines)
        img = np.concatenate([np.pad(l, ((2, 2), (0, W - l.shape[1]))) for l in lines], 0)
        d = os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "grids", tag)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(255 - img).save(os.path.join(d, "step0000001.png"))
    except Exception as e:                       # 그림은 부가 기능 — 실패해도 체크포인트는 저장됨
        say(f"  (문장 그림 실패: {e})")
    say(f"  -> {path}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="교사 체크포인트 (RF 또는 디퓨전)")
    ap.add_argument("--data", default="font_ko_v2")
    ap.add_argument("--which", default=None, choices=["ft", "pre"], help="기본: 교사 stage 로 결정")
    ap.add_argument("--split", default="train")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--n_pairs", type=int, default=0, help="쌍 개수 (0 = 데이터 전체, 글자·필체 한 번씩)")
    ap.add_argument("--gen_steps", type=int, default=64, help="교사 샘플링 스텝 (distill.py 첫 교사와 같은 64)")
    ap.add_argument("--w", type=float, default=1.5, help="교사 guidance (학생에 구워짐)")
    ap.add_argument("--gen_bs", type=int, default=512)
    ap.add_argument("--pairs", default=None, help="쌍 폴더 (기본: runs/reflow/pairs_...)")
    ap.add_argument("--init", default="teacher", choices=["teacher", "scratch"],
                    help="학생 시작 가중치. teacher = 교사 복사 (교사가 디퓨전이면 출력 의미가 달라 scratch 권장)")
    ap.add_argument("--iters", type=int, default=20000, help="reflow 학습 스텝")
    ap.add_argument("--distill_iters", type=int, default=10000,
                    help="reflow 뒤 1스텝 증류 (0 이면 생략). 기본 20,000 + 10,000 = distill.py 30,000 과 같은 학생 학습량")
    ap.add_argument("--bs", type=int, default=128, help="GPU 한 장당 배치")
    ap.add_argument("--lr", type=float, default=None, help="기본: teacher 초기화 1e-4, scratch 2e-4")
    ap.add_argument("--distill_lr", type=float, default=5e-5)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/reflow")
    a = ap.parse_args()

    dd = Dist()
    device = dd.device
    say = print if dd.main else (lambda *k, **kw: None)
    os.makedirs(a.out, exist_ok=True)
    teacher, c = load(a.teacher, device)
    teacher.requires_grad_(False)
    T = c["T"]
    tmethod = method_of(c)
    which = a.which or ("pre" if c.get("stage") == "pre" else "ft")
    ds = D.load(a.data, which, a.split, frac=a.frac, seed=a.seed, in_memory=not dd.enabled)
    assert teacher.n_styles == D.n_styles(a.data, which), "교사와 데이터 세트(--which)의 스타일 수가 다릅니다."
    ttag = os.path.splitext(os.path.basename(a.teacher))[0]
    pdir = pair_dir(a, ttag)
    name = ttag + "_reflow" + ("_scratch" if a.init == "scratch" else "")      # 초기화가 다른 비교군은 이름으로 구분
    say(f"device={device}  GPU/프로세스 {dd.world}개  교사={a.teacher} ({tmethod})  데이터 {len(ds):,}장  "
        f"교사 {a.gen_steps}스텝 w={a.w}")

    # 1) 쌍
    make_pairs(teacher, c, ds, a, dd, device, pdir, say)
    pairs = Pairs(pdir, in_memory=not dd.enabled)
    say(f"쌍 {len(pairs):,}개로 학습")

    # 2) reflow 학생
    cfg = dict(teacher.cfg, pred="rf")
    torch.manual_seed(a.seed)
    student = M.build(cfg)
    if a.init == "teacher":
        student.load_state_dict(teacher.state_dict())
        if teacher.pred != "rf":
            say(f"  주의: 교사 pred={teacher.pred} → 학생 pred=rf. 가중치만 빌리고 출력 의미는 새로 배웁니다.")
    student = student.to(device)
    del teacher
    torch.manual_seed(a.seed + 1000 * dd.rank)
    lr = a.lr or (1e-4 if a.init == "teacher" else 2e-4)
    ema, log, minutes = train_phase("reflow", reflow_loss, student, pairs, a, a.iters, lr, dd, device, T, say)
    if dd.main:
        save(os.path.join(a.out, f"{name}.pt"), student, ema, c, a, "reflow", log, minutes, pdir, say,
             student_iters=a.iters)
    dd.barrier()

    # 3) 1스텝 증류 (선택): reflow 학생의 EMA 에서 시작
    if a.distill_iters > 0:
        student2 = copy.deepcopy(ema)
        ema2, log2, min2 = train_phase("1스텝 증류", onestep_loss, student2, pairs, a, a.distill_iters,
                                       a.distill_lr, dd, device, T, say)
        if dd.main:
            save(os.path.join(a.out, f"{name}_distill.pt"), student2, ema2, c, a, "reflow+distill",
                 log + [dict(e, step=e["step"] + a.iters) for e in log2], minutes + min2, pdir, say,
                 student_iters=a.iters + a.distill_iters)
        dd.barrier()
    if dd.main:
        print(f"\n완료. 평가: python eval.py --ckpt {os.path.join(a.out, name + '.pt')} --data {a.data} --steps 1 2 4")
    dd.close()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
