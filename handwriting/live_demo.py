# -*- coding: utf-8 -*-
"""실시간 손글씨 데모 (발표용). 타자를 치는 동안 글자가 바로바로 손글씨로 나타납니다.

    python live_demo.py --ckpt runs/distill/student_1step.pt          # http://127.0.0.1:7862
    python live_demo.py --ckpt runs/distill/student_1step.pt --host 0.0.0.0   # 같은 망의 다른 기기에서
    python live_demo.py --ckpt runs/ft_v_seed0.pt --steps 10          # 증류 전 모델 (느림, 비교용)
    python live_demo.py --ckpt ... --device cpu                       # GPU 없이 (노트북 발표)

demo.py(Gradio, 버튼을 눌러 문장 단위 생성)와 달리 이 페이지는
  - 키를 누를 때마다 **새로 생긴 글자만** 생성합니다. 이미 쓴 글자는 캐시에서 꺼냅니다.
  - 글자마다 노이즈를 (시드, 글자, 그 글자가 문장에서 몇 번째로 나왔는지) 로 정합니다.
    그래서 뒤에 글자를 더 쳐도 앞 글자 모양이 바뀌지 않고, 같은 글자가 두 번 나오면 모양이 조금 다릅니다.
    노이즈는 필체와 무관하므로 필체만 바꾸면 '같은 손놀림, 다른 필체' 가 됩니다.
  - 한글 조합 중인 낱자(ㅎ, ㅏ …)는 흐리게 표시하고, 음절이 완성되는 순간 생성합니다.
  - 과정 창 http://127.0.0.1:7862/flow 를 따로 열면, 데모 페이지에서 친 키가 실제로 거친 과정을 흐름도로 보여 줍니다
    (발표 때 다른 창·다른 화면에 띄우기 좋게 분리).

필요한 패키지: torch, numpy, pillow, fastapi, uvicorn (gradio 를 설치했다면 이미 들어 있음).
원격 서버(VESSL)에서 돌릴 때는 포트 포워딩: ssh -L 7862:localhost:7862 <서버>
"""
import argparse
import base64
import io
import os
import threading
import time
import zlib
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

import hangul as H
from diffusion import make_schedule
from sample import load, make_steps, method_of, sample_loop, to_uint8

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CHARS = 200          # 한 번에 받는 문장 길이 상한
INK_THRESH = 30          # 잉크 좌우 끝을 찾을 때 기준 (compose_line 과 같음)


def is_jamo(ch):
    return 0x3131 <= ord(ch) <= 0x318E      # 호환용 자모: 조합 중인 낱자


def gray_png(a, lo, hi):
    """실수 배열 (64,64) → 회색조 PNG base64 (흐름 그래프의 작은 그림용)."""
    a = np.clip((np.asarray(a, np.float32) - lo) / (hi - lo), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((a * 255).astype(np.uint8), "L").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def jamo_names(cond):
    """[cho, jung, jong, extra] → 사람이 읽는 자모 (흐름 그래프 표시용)."""
    cho, jung, jong, extra = cond
    if extra != H.NULL_EXTRA:
        return {"extra": H.EXTRA[extra]}
    return {"cho": CHO[cho], "jung": JUNG[jung], "jong": JONG[jong] if jong else ""}


CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
TRACE_SHOW = 3           # 흐름 그래프에 그림까지 보내는 새 글자 수 (최근 것부터)


def glyph_png(img):
    """uint8 (64,64) 흰 글자 → 투명 배경 PNG (알파 = 잉크). 페이지에서 잉크 색을 입힙니다."""
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 3] = img
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    cols = np.where(img.max(0) > INK_THRESH)[0]
    left, right = (int(cols.min()), int(cols.max())) if len(cols) else (26, 37)
    return {"png": base64.b64encode(buf.getvalue()).decode(), "l": left, "r": right}


def style_check(model):
    """필체 임베딩마다 가장 가까운 다른 필체까지의 거리와, 학습되지 않은 필체 번호.

    필체 번호 방식에서는 학습에 쓰지 않은 폰트(나눔손글씨 검증 11 · 테스트 10)의 임베딩이 초기값 근처에
    서로 거의 같은 값으로 남습니다. 그래서 '가장 가까운 필체까지의 거리' 가 다른 필체들(중앙값)의 30% 보다
    작으면 학습 안 된 필체로 봅니다. (rf-v1: 학습 안 됨 ≤ 0.22, 학습됨 ≥ 0.92)"""
    e = model.e_style.weight[:-1].detach().float().cpu()          # 마지막 줄은 '없음' 토큰
    if len(e) < 3:
        return [], [0.0] * len(e)
    d = torch.cdist(e, e)
    d.fill_diagonal_(float("inf"))
    nn_d = d.min(1).values
    bad = [i for i in range(len(e)) if nn_d[i] < 0.3 * nn_d.median()]
    return (bad if len(bad) >= 2 else []), [round(float(v), 3) for v in nn_d]


class Writer:
    """모델 + 글자 캐시. 요청마다 캐시에 없는 글자만 한 배치로 생성합니다."""

    def __init__(self, ckpt, device, style=None, cache_size=20000):
        self.device = device
        self.model, self.c = load(ckpt, device)
        _, self.alpha_bar = make_schedule(self.c["T"], self.c["schedule"], device)
        self.fixed = self.c.get("steps")                     # 증류 모델이면 타임스텝 목록이 고정
        self.names = self.c.get("style_names") or [str(i) for i in range(self.model.n_styles)]
        self.untrained, self.nn_dist = style_check(self.model)
        trained = [i for i in range(len(self.names)) if i not in self.untrained]
        self.default_style = style if style is not None else (trained[0] if trained else 0)
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.lock = threading.Lock()
        # 과정 창(/flow)으로 보낼 최근 요청 기록. 과정 창이 열려 있을 때만 만듭니다.
        self.tcond = threading.Condition()
        self.tseq, self.tlast, self.tpoll = 0, None, 0.0

    # ---------------------------------------------------------------- 정보
    def info(self):
        dev = "GPU" if self.device == "cuda" else self.device.upper()
        if self.device == "cuda":
            name = torch.cuda.get_device_name(0)
        else:
            name = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")),
                        "") if os.path.exists("/proc/cpuinfo") else ""
        return {"styles": [f"{i}: {n}" for i, n in enumerate(self.names)], "default_style": self.default_style,
                "untrained": self.untrained,
                "distilled": bool(self.fixed), "steps": len(self.fixed) if self.fixed else None,
                "method": method_of(self.c),
                "w_baked": self.c.get("w_baked"), "device": dev, "device_name": name}

    # ---------------------------------------------------------------- 생성
    def _noise(self, seed, ch, k):
        g = torch.Generator().manual_seed(zlib.crc32(f"{seed}|{ch}|{k}".encode()))
        return torch.randn(1, 64, 64, generator=g)          # CPU 에서 뽑아 장치와 무관하게 같은 노이즈

    def _sync(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def _generate(self, todo, style, seed, ns, w, tr=None):
        """todo: [(key, ch, cond, k)] → {key: uint8 (64,64)}. tr 가 dict 면 단계별 시간과 그림을 담습니다."""
        t0 = time.perf_counter()
        z = torch.stack([self._noise(seed, ch, k) for _, ch, _, k in todo]).to(self.device)
        cond = torch.tensor([c for _, _, c, _ in todo], device=self.device)
        st = torch.full((len(todo),), style, device=self.device, dtype=torch.long)
        steps = self.fixed or make_steps(self.model, self.c["T"], ns)
        self._sync()
        t1 = time.perf_counter()
        x = sample_loop(self.model, z.shape, self.alpha_bar, self.c["T"], len(steps), cond, st,
                        0.0 if self.fixed else w, device=self.device, steps=steps, x_init=z)
        self._sync()
        t2 = time.perf_counter()
        if tr is not None:
            tr.update(noise_ms=round((t1 - t0) * 1000, 2), model_ms=round((t2 - t1) * 1000, 2),
                      t_list=[int(t) for t in steps])
            tr["_raw"] = (todo[-TRACE_SHOW:], seed, z[-TRACE_SHOW:, 0], x[-TRACE_SHOW:, 0])   # 그림은 시간 재기 밖에서
        return dict(zip([t[0] for t in todo], to_uint8(x)))

    def publish(self, tr):
        with self.tcond:
            self.tseq += 1
            tr["seq"] = self.tseq
            self.tlast = tr
            self.tcond.notify_all()

    def wait_trace(self, after, timeout=20.0):
        """과정 창의 롱폴링: after 보다 새 기록이 생길 때까지 최대 timeout 초 기다립니다."""
        self.tpoll = time.time()
        with self.tcond:
            if after >= 0:                                      # after < 0: 처음 연결 → 기다리지 않고 지금 상태
                self.tcond.wait_for(lambda: self.tseq > after and self.tlast is not None, timeout)
            self.tpoll = time.time()
            return self.tlast if self.tlast is not None and self.tseq > after else {"seq": self.tseq}

    def render(self, text, style, seed, ns, w, have=(), trace=False, single=False):
        """single=True ('글자별 ms' 켬): 새 글자를 한 글자씩 따로 생성해 글자마다 실제 시간을 잽니다.
        전에 여러 글자를 한 배치로 만들어 둔 글자(시간이 배치 평균)도 같은 노이즈로 다시 만들어 잽니다 (모양은 같음)."""
        trace = trace and time.time() - self.tpoll < 60          # 과정 창이 최근 1분 안에 연결돼 있었을 때만
        style = int(min(max(style, 0), len(self.names) - 1))
        ns = len(self.fixed) if self.fixed else int(min(max(ns, 1), 1000))
        w = 0.0 if self.fixed else float(w)
        t_start = time.perf_counter()
        items, todo, seen = [], [], {}
        hits = 0
        for ch, cond in H.parse(text[:MAX_CHARS]):
            if ch == " ":
                items.append({"ch": " ", "kind": "space"}); continue
            if ch == "\n":
                items.append({"ch": "\n", "kind": "newline"}); continue
            if cond is None:
                items.append({"ch": ch, "kind": "jamo" if is_jamo(ch) else "unsupported"}); continue
            k = seen.get(ch, 0); seen[ch] = k + 1
            key = f"{style}|{seed}|{ns}|{w:g}|{ch}|{k}"
            items.append({"ch": ch, "kind": "glyph", "key": key})
            if key in self.cache and not (single and self.cache[key].get("n", 1) > 1):
                hits += 1
            elif all(t[0] != key for t in todo):
                todo.append((key, ch, cond, k))
        t_parsed = time.perf_counter()
        tr = {} if trace else None
        ms = png_ms = 0.0
        out = {}
        if todo:
            groups = [[t] for t in todo] if single else [todo]     # 한 글자씩 / 한 배치로
            out, each = {}, {}
            with self.lock, torch.inference_mode():
                for gi, grp in enumerate(groups):
                    sub = {} if tr is not None else None
                    t0 = time.perf_counter()
                    o = self._generate(grp, style, seed, ns, w, sub)
                    dt = (time.perf_counter() - t0) * 1000
                    ms += dt
                    out.update(o)
                    for key in o:
                        each[key] = (dt / len(grp), len(grp), dt)
                    if sub is not None:                     # 과정 창: 시간은 합치고 그림은 마지막 묶음
                        tr["noise_ms"] = round(tr.get("noise_ms", 0) + sub["noise_ms"], 2)
                        tr["model_ms"] = round(tr.get("model_ms", 0) + sub["model_ms"], 2)
                        tr["t_list"], tr["_raw"] = sub["t_list"], sub["_raw"]
            t0 = time.perf_counter()
            for key, img in out.items():
                g = glyph_png(img)
                per, n, bt = each[key]
                g["ms"] = round(per, 1)          # 이 글자의 생성 시간 (한 배치로 n자를 만들었으면 배치 시간 ÷ n)
                g["n"], g["batch_ms"] = n, round(bt, 1)
                self.cache[key] = g
            png_ms = (time.perf_counter() - t0) * 1000
        have = set(have or ())
        glyphs = {}
        for it in items:
            if it.get("key"):
                self.cache.move_to_end(it["key"])
                if it["key"] not in have or it["key"] in out:      # 다시 잰 글자는 새 시간으로 보냄
                    glyphs[it["key"]] = self.cache[it["key"]]
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        res = {"items": items, "glyphs": glyphs, "new": len(todo), "ms": round(ms, 2), "steps": ns,
               "forwards_per_char": ns if self.fixed else ns * (2 if w > 0 else 1)}
        if trace:
            raw = tr.pop("_raw", None)
            if raw:
                sub, sd, zc, xc = raw
                zc, xc = zc.float().cpu().numpy(), xc.float().cpu().numpy()
                tr["gen"] = [{"ch": ch, "k": k, "cond": list(c), "jamo": jamo_names(c),
                              "seed": zlib.crc32(f"{sd}|{ch}|{k}".encode()),
                              "z": gray_png(zz, -2.5, 2.5), "u": gray_png(xx - zz, -3, 3), "x": gray_png(xx, -1, 1)}
                             for (_, ch, c, k), zz, xx in zip(sub, zc, xc)]
            last = next((it for it in reversed(items) if it["kind"] not in ("space", "newline")), None)
            tr.update(parse_ms=round((t_parsed - t_start) * 1000, 3), png_ms=round(png_ms, 2),
                      total_ms=round((time.perf_counter() - t_start) * 1000, 2),
                      n_glyph=sum(it["kind"] == "glyph" for it in items), hits=hits, misses=len(todo),
                      last=last and {"ch": last["ch"], "kind": last["kind"],
                                     **({"cond": H.char_cond(last["ch"]), "jamo": jamo_names(H.char_cond(last["ch"]))}
                                        if last["kind"] == "glyph" else {})},
                      method=method_of(self.c), pred=self.model.pred, w=w,
                      guided=not self.fixed and w > 0, device=self.device, steps=ns,
                      text=text[-60:], style=style, style_name=self.names[style])
            self.publish(tr)
        return res


def build_app(writer):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class Req(BaseModel):
        text: str = ""
        style: int = 0
        seed: int = 0
        steps: int = 4
        w: float = 1.5
        have: list[str] = []
        trace: bool = False                  # True 면 이 요청의 과정(단계별 시간, 노이즈 그림 …)을 과정 창(/flow)으로 보냄
        single: bool = False                 # True 면 새 글자를 한 글자씩 생성해 글자별 시간을 잼 ('글자별 ms')

    app = FastAPI(title="손글씨 실시간 데모")
    page = open(os.path.join(HERE, "live_demo.html"), encoding="utf-8").read()
    flow_page = open(os.path.join(HERE, "live_flow.html"), encoding="utf-8").read()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page

    @app.get("/flow", response_class=HTMLResponse)
    def flow():                               # 과정 창: 데모 페이지에서 친 키가 거친 과정을 따로 보여 줌
        return flow_page

    @app.get("/api/trace")
    def trace(after: int = -1):               # 롱폴링 (새 기록이 생기면 바로 응답, 없으면 20초 뒤 빈 응답)
        return writer.wait_trace(after)

    @app.get("/api/info")
    def info():
        return writer.info()

    @app.post("/api/render")
    def render(r: Req):                      # 동기 함수 → FastAPI 가 스레드풀에서 실행, 모델은 lock 으로 보호
        return writer.render(r.text, r.style, r.seed, r.steps, r.w, r.have, r.trace, r.single)

    return app


def pick_device(want):
    if want != "auto":
        return want
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--threads", type=int, default=0, help="CPU 스레드 수 (0 = torch 기본값)")
    ap.add_argument("--style", type=int, default=0, help="처음에 보여줄 필체 번호")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 이면 같은 망의 다른 기기에서 접속 가능")
    ap.add_argument("--port", type=int, default=7862)
    ap.add_argument("--styles", action="store_true", help="필체 목록과 학습 여부만 출력하고 끝냄")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    import uvicorn
    writer = Writer(a.ckpt, pick_device(a.device), a.style)
    nu = len(writer.untrained)
    if a.styles:
        print(f"필체 {len(writer.names)}개: 학습됨 {len(writer.names) - nu} · 학습 안 됨 {nu}\n"
              f"(가장 가까운 다른 필체까지의 거리. 학습 안 된 필체끼리는 거의 같은 값이라 이 거리가 아주 작음)\n")
        for i, n in enumerate(writer.names):
            print(f"{i:>4}  {'학습 안 됨' if i in writer.untrained else '학습됨    '}  {writer.nn_dist[i]:>6.3f}  {n}")
        return
    if a.style in writer.untrained:
        print(f"주의: 필체 {a.style} 은(는) 학습되지 않은 필체입니다 (python live_demo.py --ckpt ... --styles 로 목록 확인)")
    writer.render("가나다 Ab1!", a.style, 12345, 4, 1.5)        # 워밍업 (첫 요청이 느리지 않게)
    writer.cache.clear()
    i = writer.info()
    print(f"{'증류 ' + str(i['steps']) + '스텝' if i['distilled'] else '증류 전 모델'} · {i['device']} {i['device_name']}"
          f" · 필체 {len(i['styles'])}개 (학습 {len(i['styles']) - nu} · 학습 안 됨 {nu})  →  http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}")
    uvicorn.run(build_app(writer), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
