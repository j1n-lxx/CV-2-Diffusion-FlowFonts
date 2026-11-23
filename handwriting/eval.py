# -*- coding: utf-8 -*-
"""평가 축을 전부 재서 results.json 으로. 보고서 숫자는 여기서 나옵니다 (Task 2 eval.py 와 같은 역할).

    # 일반 모델: 스텝 수 × guidance
    python eval.py --ckpt runs/ft_v_seed0.pt --data font_ko_v2 --steps 4 10 50 --w 0 1.5 3
    # 증류 모델: 체크포인트에 스텝·guidance 가 고정돼 있어서 --steps/--w 무시
    python eval.py --ckpt runs/distill/student_1step.pt --data font_ko_v2
    # 노트북·학교 PC 같은 CPU 환경 속도만 (분류기·데이터 필요 없음)
    python eval.py --ckpt runs/distill/student_1step.pt --device cpu --speed_only
    python eval.py --ckpt runs/ft_v_seed0.pt --device cpu --speed_only --steps 1 4 10 50 --w 1.5
    # RF / reflow: 같은 명령. reflow 학생은 --steps 1 2 4 (오일러 스텝, guidance 는 구워짐)
    python eval.py --ckpt runs/reflow/ft_rf_seed0_reflow.pt --data font_ko_v2 --steps 1 2 4

결과(runs/eval/results_*.json)는 dashboard.py 로 한 화면에서 볼 수 있습니다.

필요한 분류기 (classifier.py 로 먼저 학습):  runs/clf/jamo_clf.pt, runs/clf/style_clf.pt
CLIPScore: open_clip_torch + transformers, torch 버전 고정해서 설치 (README Setup 참고)  (첫 실행 때 다국어 CLIP 가중치를 Hugging Face 에서 받음. 없으면 그 지표만 건너뜀)

재는 것 (정의는 metrics.py 맨 위)
  주 지표   seen/unseen 마다  fid_clf (FID ↓), is_clf (IS ↑), recall·precision (↑), clip_score·clip_rprec (↑)
            속도              char_ms (글자 하나 지연), sentence_ms (20자 문장), forwards_per_char (= NFE)
  보조      match, style_acc, (unseen) nn_to_train,
            repeat: 같은 글자를 같은 필체로 K 번 → repeat_variety(손글씨다움), repeat_match, 그림 grid_*_repeat_*.png
  real      진짜 폰트 글자(= 고정 글리프 기준선)의 같은 지표, 진짜끼리의 FID 바닥값·precision/recall
  --extra   예전 지표도: fidelity, ssim, lpips, fid(Inception), l2_to_font, seed_variety

두 가지 글자 묶음으로 잽니다.
  seen    : 학습한 필체 × 학습한 글자   (train split)
  unseen  : 학습한 필체 × 처음 보는 글자 (train_unseen_chars split, 2,350자 중 235자)
처음 보는 필체(val/test 폰트)는 스타일 임베딩이 없어 생성할 수 없으므로, 분류기 검증과
진짜 데이터 기준선에만 씁니다.
"""
import argparse
import json
import os
import platform
import time

import torch

import data as D
import metrics
from classifier import load_clf
from diffusion import make_schedule
import models as M
from sample import load, method_of, sample_loop, render_sentence

SENTENCE = "오늘 날씨가 정말 좋네요! 같이 산책 갈래?"
REPEAT_CHARS = "이는가의에하요고"     # 한 문장에 자주 되풀이되는 글자 (holdout 235자에 든 글자는 고정 글리프가 없어 빠짐: 예 '다')


def pick(ds, n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(len(ds), generator=g)[:n]


GEN_BS = 500      # 한 번에 생성하는 글자 수. CPU 에서는 main 에서 100 으로 줄임 (메모리)


def gen(model, cond, style, alpha_bar, T, ns, w, seed, device, steps, bs=None):
    bs = bs or GEN_BS
    out = []
    for i in range(0, len(cond), bs):
        out.append(sample_loop(model, (len(cond[i:i + bs]), 1, 64, 64), alpha_bar, T, ns,
                               cond[i:i + bs], style[i:i + bs], w, seed=seed + i, device=device,
                               steps=steps).cpu())
    return torch.cat(out)


def pick_device(want):
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA GPU 가 없습니다.")
    return want


def device_info(device):
    """(종류, 이름). 대시보드에서 CPU/GPU 를 구분하고 기기 이름을 보여줄 때 씁니다."""
    if str(device).startswith("cuda"):
        return "GPU", torch.cuda.get_device_name(0)
    if device == "mps":
        return "GPU", "Apple MPS"
    name = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return "CPU", f"{name} ({torch.get_num_threads()}스레드)"


def forwards_per_char(distilled, ns, w):
    """글자 하나에 모델을 몇 번 돌리는가. guidance(w>0) 는 스텝마다 조건부·무조건부 2번."""
    return ns if distilled else ns * (2 if w > 0 else 1)


def plan(c, a_steps, a_w):
    """체크포인트 종류별 평가 설정.
    반환: (method, baked, grid=[(스텝, w)], steps_for(ns) → 타임스텝 목록 또는 None)
      - diffusion / rf : --steps × --w 전부
      - pd (progressive distillation) : 저장된 고정 스텝 하나, w 는 구워짐
      - reflow        : --steps (기본 1) 의 오일러 스텝, w 는 구워짐
    """
    method = method_of(c)
    T = c["T"]
    if method.startswith("reflow"):
        return method, True, [(s, 0.0) for s in (a_steps or [1])], (lambda ns: M.rf_steps(T, ns))
    if c.get("steps") is not None:
        return method, True, [(len(c["steps"]), 0.0)], (lambda ns: c["steps"])
    return method, False, [(s, w) for s in (a_steps or [50]) for w in a_w], (lambda ns: None)


def time_sentence(model, c, alpha_bar, ns, w, device, reps=5, steps=None):
    render_sentence(model, SENTENCE[:3], 0, alpha_bar, c["T"], ns, w, 0, device, steps)   # 워밍업
    secs = [render_sentence(model, SENTENCE, 0, alpha_bar, c["T"], ns, w, k, device, steps)[2] for k in range(reps)]
    secs.sort()
    return 1000 * secs[len(secs) // 2]          # 중앙값 (ms)


def time_char(model, c, alpha_bar, ns, w, device, reps=20, steps=None):
    """새 글자 하나를 만드는 데 걸리는 시간 (배치 1, 중앙값 ms). 데모에서 키를 쳐 음절이 완성될 때의 지연."""
    for k in range(2):
        render_sentence(model, "가", 0, alpha_bar, c["T"], ns, w, k, device, steps)          # 워밍업
    secs = sorted(render_sentence(model, "갉", 0, alpha_bar, c["T"], ns, w, k, device, steps)[2] for k in range(reps))
    return 1000 * secs[len(secs) // 2]


def speed_only(model, c, alpha_bar, a, device, dev_kind, dev_name, tag):
    method, distilled, grid, steps_for = plan(c, a.steps, a.w)
    n_chars = sum(1 for ch in SENTENCE if not ch.isspace())
    print(f"속도만 측정: {dev_kind} {dev_name}  ckpt={tag}  문장 {n_chars}자\n")
    print(f"{'steps':>5} {'w':>4} {'모델 실행/글자':>12} {'ms/문장':>9} {'ms/글자':>8} {'글자 하나 ms':>11}")
    runs = []
    for ns, w in grid:
        ms = time_sentence(model, c, alpha_bar, ns, w, device, steps=steps_for(ns))
        cms = time_char(model, c, alpha_bar, ns, w, device, steps=steps_for(ns))
        f = forwards_per_char(distilled, ns, w)
        runs.append({"steps": ns, "w": w, "forwards_per_char": f, "sentence_ms": ms, "sentence_chars": n_chars,
                     "char_ms": cms})
        print(f"{ns:>5} {w:>4.1f} {f:>12} {ms:>9.0f} {ms / n_chars:>8.1f} {cms:>11.1f}")
    res = {"ckpt": tag, "method": method, "cost": c.get("cost"), "init": c.get("init"), "teacher_method": c.get("teacher_method"),
           "train_seed": c.get("seed"), "distilled": distilled, "device": dev_kind, "device_name": dev_name, "speed_only": True,
           "w_baked": c.get("w_baked"), "runs": runs}
    path = os.path.join(a.out, f"results_{tag}_{dev_kind.lower()}_speed.json")
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="font_ko_v2")
    ap.add_argument("--which", default=None, help="ft/pre. 기본: 체크포인트 stage 로 결정 (mix 는 ft)")
    ap.add_argument("--jamo_clf", default="runs/clf/jamo_clf.pt")
    ap.add_argument("--style_clf", default="runs/clf/style_clf.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=None, help="기본: 50 (reflow 학생은 1)")
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 1.5, 3.0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="GPU 가 있어도 --device cpu 로 노트북·학교 PC 같은 CPU 환경 속도를 잴 수 있습니다")
    ap.add_argument("--threads", type=int, default=None, help="CPU 스레드 수 고정 (기기 간 비교용)")
    ap.add_argument("--repeat_chars", default=REPEAT_CHARS, help="반복 글자 평가에 쓸 글자들")
    ap.add_argument("--repeat_k", type=int, default=8, help="글자마다 몇 번 반복해 쓸지")
    ap.add_argument("--repeat_styles", type=int, default=16, help="반복 글자 평가에 쓸 필체 수")
    ap.add_argument("--bs", type=int, default=None, help="한 번에 생성하는 글자 수 (기본 GPU 500, CPU·MPS 100)")
    ap.add_argument("--no_clip", action="store_true", help="CLIPScore 건너뛰기")
    ap.add_argument("--clip_model", default=metrics.CLIP_MODEL, help="open_clip '모델:가중치'")
    ap.add_argument("--clip_prompt", default=metrics.CLIP_PROMPT, help="CLIPScore 텍스트 틀 ({} = 글자)")
    ap.add_argument("--extra", action="store_true",
                    help="보조 지표도: fidelity, SSIM, LPIPS, Inception FID, L2font, seed_variety")
    ap.add_argument("--no_lpips", action="store_true", help="(--extra) LPIPS 건너뛰기")
    ap.add_argument("--no_inception", action="store_true", help="(--extra) Inception FID 건너뛰기")
    ap.add_argument("--speed_only", action="store_true",
                    help="분류기·데이터 없이 문장 생성 시간만 잽니다 (노트북에서 빠르게)")
    a = ap.parse_args()

    device = pick_device(a.device)
    global GEN_BS
    GEN_BS = a.bs or (500 if device == "cuda" else 100)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev_kind, dev_name = device_info(device)
    os.makedirs(a.out, exist_ok=True)
    model, c = load(a.ckpt, device)
    T = c["T"]
    _, alpha_bar = make_schedule(T, c["schedule"], device)
    tag = os.path.splitext(os.path.basename(a.ckpt))[0]
    if tag.startswith("student_") and c.get("teacher"):      # PD 학생은 이름에 교사(시드)를 붙여 결과 파일이 겹치지 않게
        tag = f"{os.path.splitext(os.path.basename(c['teacher']))[0]}_pd_{tag[len('student_'):]}"
    if a.speed_only:
        return speed_only(model, c, alpha_bar, a, device, dev_kind, dev_name, tag)
    stage = c.get("stage") or (torch.load(c["teacher"], map_location="cpu", weights_only=False).get("stage")
                               if c.get("teacher") and os.path.exists(c["teacher"]) else "ft")
    which = a.which or ("pre" if stage == "pre" else "ft")
    offset = D.n_styles(a.data, "pre") if (stage == "mix" and which == "ft") else 0
    method, distilled, grid, steps_for = plan(c, a.steps, a.w)
    print(f"device={device}  ckpt={os.path.basename(a.ckpt)}  stage={stage}  which={which}  방법={method}  "
          f"{'(w=' + str(c.get('w_baked')) + ' 구워짐)' if distilled else ''}\n")

    jclf, jc = load_clf(a.jamo_clf, device)
    sclf = load_clf(a.style_clf, device)[0] if (which == "ft" and os.path.exists(a.style_clf)) else None

    seen = D.load(a.data, which, "train", in_memory=False, style_offset=offset)
    unseen = D.load(a.data, which, "train_unseen_chars", in_memory=False, style_offset=offset)
    S = {}
    for name, ds in (("seen", seen), ("unseen", unseen)):
        idx = pick(ds, a.n, a.seed)
        x, st, cond = ds.batch(idx, "cpu")
        S[name] = dict(x=x, st=st, cond=cond, chars=[ds.chars[int(ds.char_id[int(i)])] for i in idx],
                       classes=torch.unique(cond, dim=0))
        S[name]["feat"] = metrics.features(jclf, x, device)            # 진짜 글자 특징 (FID·precision/recall)

    # ---- 진짜 데이터 기준선
    res = {"ckpt": tag, "method": method, "cost": c.get("cost"), "init": c.get("init"),
           "teacher_method": c.get("teacher_method"), "train_seed": c.get("seed"), "stage": stage, "which": which, "distilled": distilled, "w_baked": c.get("w_baked"),
           "device": dev_kind, "device_name": dev_name, "jamo_clf": jc.get("results"), "real": {}, "runs": []}
    def main_metrics(xg, name):
        """주 지표: FID(분류기 특징), IS, precision/recall, CLIPScore — 같은 (필체, 글자)의 진짜 묶음과 비교."""
        fg = metrics.features(jclf, xg, device)
        out = {"fid_clf": metrics.frechet(fg, S[name]["feat"]),
               "is_clf": metrics.classifier_is(jclf, xg, S[name]["classes"], device)}
        out["precision"], out["recall"] = metrics.precision_recall(S[name]["feat"], fg)
        if not a.no_clip:
            out["clip_score"], out["clip_rprec"] = metrics.clip_score(xg, S[name]["chars"], device, a.clip_model, a.clip_prompt)
        return out

    def extra_metrics(xg, name):
        """--extra: 예전 보조 지표 (선행연구 표와 나란히 놓을 때)."""
        xr = S[name]["x"]
        out = {"ssim": metrics.ssim(xg, xr), "fidelity": metrics.jamo_scores(jclf, xg, S[name]["cond"], device)[2]}
        if not a.no_lpips:
            out["lpips"] = metrics.lpips_score(xg, xr, device)
        if not a.no_inception:
            out["fid"] = metrics.fid_inception(xg, xr, device)
        return out

    for name, ds in (("seen", seen), ("unseen", unseen)):
        m, heads, fid = metrics.jamo_scores(jclf, S[name]["x"], S[name]["cond"], device)
        R0 = res["real"][name] = {"match": m, "fidelity": fid}
        # 진짜 글자(= 고정 글리프 기준선)의 IS·CLIPScore. FID·precision/recall 은 생성 묶음을 바로 이 진짜 묶음
        # (같은 필체·같은 글자)과 비교하므로 진짜 글자 자신은 FID 0, precision·recall 1 입니다.
        R0["is_clf"] = metrics.classifier_is(jclf, S[name]["x"], S[name]["classes"], device)
        R0["is_max"] = len(S[name]["classes"])
        if not a.no_clip:
            R0["clip_score"], R0["clip_rprec"] = metrics.clip_score(S[name]["x"], S[name]["chars"], device, a.clip_model, a.clip_prompt)
        if sclf is not None:
            R0["style_acc"] = metrics.style_scores(sclf, S[name]["x"], S[name]["st"] - offset, None, device)[0]
    # 처음 보는 글자의 진짜 이미지가, 같은 필체의 학습 글자들과 얼마나 가까운가 (암기 기준선)
    tr_by_style = {}
    def train_imgs(s):
        if s not in tr_by_style:
            ids = torch.nonzero(seen.style == s).flatten()
            tr_by_style[s] = seen.batch(ids, "cpu")[0]
        return tr_by_style[s]
    styles_u = S["unseen"]["st"][:200]
    real_nn = sum(metrics.nn_to_train(S["unseen"]["x"][i:i + 1], train_imgs(int(s)), device)
                  for i, s in enumerate(styles_u)) / len(styles_u)
    res["real"]["unseen"]["nn_to_train"] = real_nn
    for name in ("seen", "unseen"):
        R0 = res["real"][name]
        clip = "-" if R0.get("clip_score") is None else f"{R0['clip_score']:.3f} (R-prec {R0['clip_rprec']:.3f})"
        print(f"[기준] 진짜 글자 {name:>6}: IS {R0['is_clf']:.1f} (최대 {R0['is_max']})  CLIPScore {clip}  match {R0['match']:.3f}"
              f"   (FID·precision/recall 은 이 진짜 묶음과 비교 → 진짜 자신은 0·1)")
    print(f"[기준] 진짜 글자 NN-train(처음 보는 글자) {real_nn:.2f}\n")

    # ---- 반복 글자: 같은 필체로 같은 글자를 K 번. 고정 글리프 = 그 필체의 진짜 폰트 글자 (늘 같은 그림)
    rep_styles = [int(s) for s in torch.unique(seen.style)[:a.repeat_styles]]
    rc, rs, rfix, rlab = [], [], [], []
    for s_ in rep_styles:
        for ch in a.repeat_chars:
            if ch not in seen.chars:
                continue
            cid = seen.chars.index(ch)
            hit = torch.nonzero((seen.style == s_) & (seen.char_id == cid)).flatten()
            if len(hit) == 0:
                continue
            x1, st1, c1 = seen.batch(hit[:1], "cpu")
            rfix.append(x1[0]); rc.append(c1[0]); rs.append(st1[0]); rlab.append(ch)
    miss = [ch for ch in a.repeat_chars if ch not in set(rlab)]
    if miss:
        print(f"[반복 글자] 학습 글자에 없어 뺀 글자: {''.join(miss)} (holdout 이거나 데이터에 없음)")
    R = None
    if rc:
        K = a.repeat_k
        R = dict(fixed=torch.stack(rfix), cond=torch.stack(rc).repeat_interleave(K, 0),
                 st=torch.stack(rs).repeat_interleave(K, 0), G=len(rc), K=K, n_chars=len(set(rlab)), labels=rlab)
        res["real"]["repeat"] = {"repeat_variety_feat": 0.0, "repeat_variety_pix": 0.0,
                                 "repeat_match": metrics.jamo_scores(jclf, R["fixed"], torch.stack(rc), device)[0],
                                 "groups": R["G"], "k": K, "chars": a.repeat_chars, "note": "고정 글리프: 늘 같은 그림이라 0"}
        print(f"[반복 글자] 필체 {len(rep_styles)}개 × 글자 {R['n_chars']}개 = {R['G']}묶음, 묶음마다 {K}번\n")

    hdr = (f"{'steps':>5} {'w':>4} {'NFE':>4} | {'FID':>6} {'IS':>6} {'Prec':>5} {'Rec':>5} {'CLIP':>5} | "
           f"{'FID':>6} {'IS':>6} {'Prec':>5} {'Rec':>5} {'CLIP':>5} | {'match':>6} {'style':>6} {'rep_f':>6} | {'ms/1자':>7}")
    print(f"{'':>16}| {'── seen (본 글자) ──':^31}| {'── unseen (처음 보는 글자) ──':^29}| {'보조 (unseen)':^21}|")
    print(hdr)
    f3 = lambda v, d=3, wd=5: f"{'-':>{wd}}" if v is None else f"{v:>{wd}.{d}f}"
    for ns, w in grid:
        r = {"steps": ns, "w": w}
        for name in S:
            xg = gen(model, S[name]["cond"].to(device), S[name]["st"].to(device), alpha_bar, T, ns, w, a.seed, device,
                     steps_for(ns))
            m, heads, _ = metrics.jamo_scores(jclf, xg, S[name]["cond"], device)
            r[name] = {"match": m, "match_heads": heads}
            r[name].update(main_metrics(xg, name))
            if sclf is not None:
                r[name]["style_acc"] = metrics.style_scores(sclf, xg, S[name]["st"] - offset, None, device)[0]
            if a.extra:
                r[name].update(extra_metrics(xg, name))
            if name == "seen" and a.extra:
                r[name]["l2_to_font"] = metrics.l2_to_target(xg, S[name]["x"])
                xg2 = gen(model, S[name]["cond"][:300].to(device), S[name]["st"][:300].to(device), alpha_bar, T, ns, w,
                          a.seed + 1000, device, steps_for(ns))
                r[name]["seed_variety_feat"], r[name]["seed_variety_pix"] = metrics.seed_variety(jclf, xg[:300], xg2, device)
            if name == "unseen":
                r[name]["nn_to_train"] = sum(metrics.nn_to_train(xg[i:i + 1], train_imgs(int(s)), device)
                                             for i, s in enumerate(styles_u)) / len(styles_u)
            metrics.save_grid(xg, os.path.join(a.out, f"grid_{tag}_{name}_s{ns}_w{w:g}.png"))
        # 반복 글자 (손글씨다움): 같은 조건을 K 번, 노이즈만 다름
        if R is not None:
            xr = gen(model, R["cond"].to(device), R["st"].to(device), alpha_bar, T, ns, w, a.seed + 2000, device,
                     steps_for(ns)).reshape(R["G"], R["K"], 1, 64, 64)
            vf, vp = metrics.repeat_variety(jclf, xr, device)
            r["repeat"] = {"repeat_variety_feat": vf, "repeat_variety_pix": vp,
                           "repeat_match": metrics.jamo_scores(jclf, xr.reshape(-1, 1, 64, 64), R["cond"], device)[0],
                           "groups": R["G"], "k": R["K"]}
            n1 = min(R["G"], len(a.repeat_chars))                     # 그림은 첫 필체 한 줄씩
            metrics.save_repeat_grid(R["fixed"][:n1], xr[:n1], os.path.join(a.out, f"grid_{tag}_repeat_s{ns}_w{w:g}.png"))
        # 속도: 문장 하나 (같은 필체, 한 배치), 5번 잰 중앙값 / 새 글자 하나 (배치 1), 20번 잰 중앙값 / NFE
        r["sentence_ms"] = time_sentence(model, c, alpha_bar, ns, w, device, steps=steps_for(ns))
        r["char_ms"] = time_char(model, c, alpha_bar, ns, w, device, steps=steps_for(ns))
        r["sentence_chars"] = sum(1 for ch in SENTENCE if not ch.isspace())
        r["forwards_per_char"] = forwards_per_char(distilled, ns, w)
        res["runs"].append(r)
        sv, uv = r["seen"], r["unseen"]
        rv = r.get("repeat", {}).get("repeat_variety_feat")
        print(f"{ns:>5} {w:>4.1f} {r['forwards_per_char']:>4} | "
              f"{sv['fid_clf']:>6.2f} {sv['is_clf']:>6.1f} {f3(sv['precision'], 2)} {f3(sv['recall'], 2)} {f3(sv.get('clip_score'), 2)} | "
              f"{uv['fid_clf']:>6.2f} {uv['is_clf']:>6.1f} {f3(uv['precision'], 2)} {f3(uv['recall'], 2)} {f3(uv.get('clip_score'), 2)} | "
              f"{uv['match']:>6.3f} {f3(uv.get('style_acc'), 3, 6)} {f3(rv, 2, 6)} | {r['char_ms']:>7.1f}", flush=True)

    if any(r["unseen"]["nn_to_train"] < real_nn for r in res["runs"]):
        print("\n경고: 처음 보는 글자를 생성한 결과가 진짜 글자보다 학습 글자에 더 가깝습니다 → 다른 글자를 베끼는지 의심")
    if a.extra and any(r["seen"]["seed_variety_feat"] < 0.05 * max(1e-6, max(q["seen"]["seed_variety_feat"] for q in res["runs"]))
           for r in res["runs"]):
        print("참고: seed_variety 가 거의 0 인 설정은 시드와 무관하게 같은 모양만 냅니다 (1스텝 증류에서 흔함).")
    path = os.path.join(a.out, f"results_{tag}_{dev_kind.lower()}_seed{a.seed}.json")
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
