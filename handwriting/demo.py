# -*- coding: utf-8 -*-
"""실시간 데모: 문장을 입력하면 글자마다 손글씨 이미지를 띄웁니다 (띄어쓰기 포함).

    pip install gradio
    python demo.py --ckpt runs/distill/student_1step.pt            # 1스텝 (데모 기본)
    python demo.py --ckpt runs/distill/student_4step.pt            # 1스텝이 흐릴 때 예비
    python demo.py --ckpt runs/ft_v_seed0.pt --steps 50            # 증류 전 원본 (비교용)

브라우저에서 http://127.0.0.1:7860 을 엽니다. 발표장에서 다른 기기로 보려면 --share.
문장 한 개를 한 번의 배치로 생성하고, 모든 글자에 같은 필체(스타일)를 씁니다.
"""
import argparse

import numpy as np
import torch

from diffusion import make_schedule
from sample import load, render_sentence


def build_app(ckpt, steps_default=50, w_default=1.5, device=None):
    import gradio as gr

    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, c = load(ckpt, device)
    _, alpha_bar = make_schedule(c["T"], c["schedule"], device)
    fixed = c.get("steps")                       # 증류 모델이면 고정 스텝
    names = c.get("style_names") or [str(i) for i in range(model.n_styles)]
    choices = [f"{i}: {n}" for i, n in enumerate(names)]
    render_sentence(model, "가나", 0, alpha_bar, c["T"], 4, 0.0, 0, device, fixed)   # 워밍업

    def run(text, style, steps, w, seed):
        sidx = int(str(style).split(":")[0])
        ns = len(fixed) if fixed else int(steps)
        items, line, sec = render_sentence(model, text, sidx, alpha_bar, c["T"], ns, 0.0 if fixed else float(w),
                                           int(seed), device, fixed)
        gallery = []
        for ch, im in items:
            if im is None:
                im = np.zeros((64, 64), np.uint8)
                ch = "␣" if ch == " " else f"{ch} (지원 안 함)"
            gallery.append((255 - im, ch))
        n = sum(im is not None for _, im in items)
        info = f"{n}글자 · {ns}스텝 · {sec * 1000:.0f} ms ({device})"
        return 255 - line, gallery, info

    with gr.Blocks(title="한글 손글씨 생성") as app:
        gr.Markdown("## 한글 손글씨 생성 데모\n문장을 입력하고 필체를 고르면 글자마다 손글씨 이미지를 만듭니다.")
        with gr.Row():
            text = gr.Textbox(label="문장", value="안녕하세요, 반가워요! (Hi 2026) “좋아” ♡", scale=4)
            btn = gr.Button("생성", variant="primary", scale=1)
        with gr.Row():
            style = gr.Dropdown(choices, value=choices[0], label="필체")
            steps = gr.Slider(1, 100, value=steps_default, step=1, label="스텝 수", visible=not fixed)
            w = gr.Slider(0, 5, value=w_default, step=0.5, label="guidance w", visible=not fixed)
            seed = gr.Number(value=0, precision=0, label="시드 (바꾸면 다시 생성)")
        line = gr.Image(label="한 줄", type="numpy", image_mode="L")
        info = gr.Markdown()
        gallery = gr.Gallery(label="글자별", columns=12, height="auto")
        for ev in (btn.click, text.submit):
            ev(run, [text, style, steps, w, seed], [line, gallery, info])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50, help="증류 모델이 아닐 때 기본 스텝 수")
    ap.add_argument("--w", type=float, default=1.5)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_app(a.ckpt, a.steps, a.w).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
