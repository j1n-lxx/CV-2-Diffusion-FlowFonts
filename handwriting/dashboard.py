# -*- coding: utf-8 -*-
"""실행 결과 대시보드. 데모(demo.py)와 따로 띄우는 '결과 보기' 화면입니다.

    pip install gradio altair
    python dashboard.py --runs runs                 # http://127.0.0.1:7861
    python dashboard.py --runs runs --share         # 다른 기기에서 보기

읽는 파일 (없으면 그 탭만 비어 있음)
  runs/eval/results_*.json        eval.py 결과 (품질 + 속도, 또는 --speed_only 속도만)
  runs/*_train.json               train.py 학습 기록 (손실 곡선)
  runs/grids/<tag>/step*.png      학습 중 고정 시드 문장 그림
  runs/eval/grid_*.png            평가 그리드

탭: 요약 · 속도와 품질 · 학습 곡선 · 그림.  학습·평가가 끝날 때마다 [새로고침] 을 누르세요.
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

# 색 (시각화 지침의 기본 팔레트, 밝은 화면 기준). 장치는 고정 색: CPU = 1번 파랑, GPU = 2번 주황
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DEVICE_COLOR = {"CPU": SERIES[0], "GPU": SERIES[1]}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
LOG_TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]


# ================================================================ 데이터 읽기
def setting_label(r):
    return f"{r['arm']} {r['steps']}스텝" if r["distilled"] else f"{r['arm']} {r['steps']}스텝 w{r['w']:g}"


def arm_of(method, ckpt, teacher_method=None, init=None):
    """비교군 이름 (experiments.sh 의 A–F). 예전 결과 파일은 체크포인트 이름으로 추정."""
    if method == "diffusion":
        return "A 디퓨전"
    if method == "rf":
        return "B RF"
    if method == "pd":
        return "F 디퓨전→PD"
    if method.startswith("reflow"):
        tm = teacher_method or ("diffusion" if re.search(r"(^|_)(ft|pre|mix)_v_", ckpt) else "rf")
        init = init or ("scratch" if "_scratch" in ckpt else "teacher")
        if tm == "rf":
            base = "C RF→Reflow" if init == "teacher" else "D RF→Reflow(무작위)"
        else:
            base = "E 디퓨전→Reflow" + ("" if init == "scratch" else "(교사 시작)")
        return base + ("+증류" if method == "reflow+distill" else "")
    return method


ARM_ORDER = ["A 디퓨전", "B RF", "C RF→Reflow", "C RF→Reflow+증류", "D RF→Reflow(무작위)", "D RF→Reflow(무작위)+증류",
             "E 디퓨전→Reflow", "E 디퓨전→Reflow+증류", "F 디퓨전→PD"]


# 방법 이름 (eval.py 의 method). 1스텝 학생(PD·Reflow)은 guidance 가 구워져 있어 w 표시 없음
METHOD_LABEL = {"diffusion": "디퓨전", "rf": "RF", "pd": "PD", "reflow": "Reflow", "reflow+distill": "Reflow+증류"}
METHOD_SHAPE = {"디퓨전": "circle", "RF": "square", "PD": "diamond", "Reflow": "triangle-up", "Reflow+증류": "cross"}


def load_eval(runs):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "eval", "results_*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        distilled = bool(d.get("distilled"))
        for r in d.get("runs", []):
            ns, w = int(r["steps"]), float(r.get("w", 0.0))
            f = r.get("forwards_per_char") or (ns if distilled else ns * (2 if w > 0 else 1))
            n = r.get("sentence_chars") or 18
            method = d.get("method") or ("pd" if distilled else "diffusion")
            row = dict(file=os.path.basename(p), ckpt=d.get("ckpt", "?"), distilled=distilled, method=method,
                       device=d.get("device", "?"), device_name=d.get("device_name", ""),
                       speed_only=bool(d.get("speed_only")), steps=ns, w=w, forwards=int(f),
                       ms=float(r["sentence_ms"]), ms_char=float(r.get("char_ms") or float(r["sentence_ms"]) / n))
            for name in ("seen", "unseen"):
                if name in r:
                    row[f"match_{name}"] = r[name].get("match")
                    row[f"style_{name}"] = r[name].get("style_acc")
                    for k, col in (("fid_clf", "fid"), ("is_clf", "is"), ("precision", "prec"), ("recall", "rec"),
                                   ("clip_score", "clip")):
                        row[f"{col}_{name}"] = r[name].get(k)
            if "seen" in r:
                row["fidelity"] = r["seen"].get("fidelity")
                row["seed_variety"] = r["seen"].get("seed_variety_feat")
                row["l2_font"] = r["seen"].get("l2_to_font")
            if "unseen" in r:
                row["nn_unseen"] = r["unseen"].get("nn_to_train")
            if "real" in d:
                row["real_match_unseen"] = d["real"].get("unseen", {}).get("match")
                row["real_nn_unseen"] = d["real"].get("unseen", {}).get("nn_to_train")
            cost = d.get("cost") or {}
            row["student_iters"] = cost.get("student_iters")
            row["teacher_nfe_m"] = (cost["teacher_nfe"] / 1e6) if cost.get("teacher_nfe") else None
            row["train_min"] = (cost.get("train_minutes") or 0) + (cost.get("pair_minutes") or 0) or None
            row["arm"] = arm_of(method, row["ckpt"], d.get("teacher_method"), d.get("init"))
            sd = d.get("train_seed")
            if sd is None:
                mt = re.search(r"seed(\d+)", row["ckpt"])
                sd = int(mt.group(1)) if mt else 0
            row["seed"] = int(sd)
            row["setting"] = setting_label(row)
            rows.append(row)
    return pd.DataFrame(rows)


NUM = ["ms", "ms_char", "match_seen", "match_unseen", "style_seen", "style_unseen", "fidelity", "seed_variety", "l2_font",
       "fid_seen", "fid_unseen", "is_seen", "is_unseen", "prec_seen", "prec_unseen", "rec_seen", "rec_unseen",
       "clip_seen", "clip_unseen",
       "nn_unseen", "real_match_unseen", "real_nn_unseen", "student_iters", "teacher_nfe_m", "train_min"]
KEY = ["arm", "setting", "method", "device", "device_name", "steps", "w", "distilled", "forwards", "speed_only"]


def aggregate(ev):
    """시드끼리 묶어 평균·표준편차 (열 이름 뒤 _sd), 시드 수 n_seeds."""
    if ev.empty:
        return ev
    ev = ev.copy()
    for c in NUM:
        if c not in ev:
            ev[c] = float("nan")
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev["device_name"] = ev["device_name"].fillna("")
    g = ev.groupby(KEY, sort=False)
    m = g[NUM].mean()
    s = g[NUM].std()
    s.columns = [c + "_sd" for c in NUM]
    out = pd.concat([m, s], axis=1)
    out["n_seeds"] = g["seed"].nunique()
    out["ckpt"] = g["ckpt"].agg(lambda x: ", ".join(sorted(set(x))))
    out["file"] = g["file"].agg(lambda x: ", ".join(sorted(set(x))))
    return out.reset_index()


def load_train(runs):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "**", "*_train.json"), recursive=True)):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        tag = os.path.basename(p)[:-len("_train.json")]
        for e in d.get("log", []):
            rows.append(dict(run=tag, step=e["step"], loss=e["loss"], minutes=e.get("sec", 0) / 60))
    return pd.DataFrame(rows)


def grid_dirs(runs):
    return sorted(d for d in glob.glob(os.path.join(runs, "grids", "*")) if os.path.isdir(d))


# ================================================================ 차트 (Altair)
def _style(ch):
    return (ch.configure(background=SURFACE, padding=12)
              .configure_view(stroke=None)
              .configure_axis(labelColor=INK2, titleColor=INK2, gridColor=GRID, domainColor=GRID, tickColor=GRID,
                              labelFontSize=12, titleFontSize=12)
              .configure_legend(labelColor=INK, titleColor=INK2, labelFontSize=12, titleFontSize=12, orient="top",
                                symbolOpacity=1)
              .configure_title(color=INK, fontSize=14, anchor="start", subtitleColor=INK2, subtitleFontSize=12)
              .configure_header(labelColor=INK, titleColor=INK2, labelFontSize=12))


def _dev_scale(df):
    import altair as alt
    devs = [d for d in ("CPU", "GPU") if d in set(df["device"])] + sorted(set(df["device"]) - {"CPU", "GPU"})
    return alt.Scale(domain=devs, range=[DEVICE_COLOR.get(d, "#8a8985") for d in devs])


def _note(text):
    import altair as alt
    return _style(alt.Chart(pd.DataFrame({"t": [text]})).mark_text(color=INK2, fontSize=13)
                  .encode(text="t:N").properties(width=560, height=80))


def _log_axis():
    import altair as alt
    return alt.Axis(values=LOG_TICKS, format=",.0f")


def chart_speed(df):
    """설정별 문장 생성 시간 (로그 축). 같은 설정을 CPU·GPU 로 나란히."""
    import altair as alt
    if df.empty:
        return _note("속도 결과가 없습니다. eval.py 를 실행하세요 (--speed_only 면 빠름).")
    d = (df.sort_values("speed_only", ascending=False)
           .drop_duplicates(["ckpt", "device", "steps", "w"]).copy())
    order = d.sort_values(["forwards", "distilled"], ascending=[True, False])["setting"].drop_duplicates().tolist()
    tip = [alt.Tooltip("setting:N", title="설정"), alt.Tooltip("device:N", title="장치"),
           alt.Tooltip("device_name:N", title="기기"), alt.Tooltip("ms:Q", title="ms/문장", format=",.0f"),
           alt.Tooltip("ms_char:Q", title="ms/글자", format=",.1f"),
           alt.Tooltip("forwards:Q", title="모델 실행/글자"), alt.Tooltip("ckpt:N", title="체크포인트")]
    base = alt.Chart(d).encode(
        y=alt.Y("setting:N", sort=order, title=None),
        x=alt.X("ms:Q", scale=alt.Scale(type="log"), title="문장 하나 생성 시간 (ms, 로그 축)", axis=_log_axis()),
        color=alt.Color("device:N", scale=_dev_scale(d), title="장치"),
        yOffset=alt.YOffset("device:N"), tooltip=tip)
    rule = base.mark_rule(strokeWidth=2, opacity=0.35).encode(x2=alt.datum(1))
    pts = base.mark_point(filled=True, size=110, stroke=SURFACE, strokeWidth=2, opacity=1)
    lab = base.mark_text(align="left", dx=10, fontSize=11, color=INK2).encode(text=alt.Text("ms:Q", format=",.0f"))
    ch = (rule + pts + lab).properties(width=560, height=max(140, 26 * len(order) * max(1, d["device"].nunique())),
                                       title=alt.TitleParams("스텝을 줄일수록 문장 생성 시간이 줄어듭니다",
                                                             subtitle="20자 안팎 문장 하나, 5번 잰 중앙값"))
    return _style(ch)


def chart_forwards(df):
    """글자 하나에 모델을 몇 번 돌리는가 (선형 축)."""
    import altair as alt
    if df.empty:
        return _note("결과가 없습니다.")
    d = df.drop_duplicates("setting").copy()
    order = d.sort_values(["forwards", "distilled"], ascending=[True, False])["setting"].tolist()
    ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, color=SERIES[0], size=14).encode(
        y=alt.Y("setting:N", sort=order, title=None),
        x=alt.X("forwards:Q", title="글자당 모델 실행 횟수"),
        tooltip=[alt.Tooltip("setting:N", title="설정"), alt.Tooltip("forwards:Q", title="모델 실행/글자")])
    lab = alt.Chart(d).mark_text(align="left", dx=6, fontSize=11, color=INK2).encode(
        y=alt.Y("setting:N", sort=order), x="forwards:Q", text="forwards:Q")
    return _style((ch + lab).properties(width=560, height=max(120, 26 * len(order)),
                                        title=alt.TitleParams("1스텝 학생(PD·Reflow)은 글자당 한 번만 실행합니다",
                                                              subtitle="guidance(w>0)는 스텝마다 2번 (조건부 + 무조건부)")))


def chart_tradeoff(df, which="unseen"):
    """속도(가로, 로그)와 글자 정확도(세로)를 한 평면에. 왼쪽 위가 좋은 쪽."""
    import altair as alt
    col = f"match_{which}"
    if df.empty or col not in df or df[col].dropna().empty:
        return _note("품질 결과가 없습니다. eval.py 를 분류기와 함께 실행하세요.")
    d = df.dropna(subset=[col]).copy()
    d["종류"] = d["method"].map(lambda m: METHOD_LABEL.get(m, m))
    d["steplab"] = d["arm"].str.split().str[0] + "·" + d["steps"].astype(str)
    tip = [alt.Tooltip("setting:N", title="설정"), alt.Tooltip("device:N", title="장치"),
           alt.Tooltip("ms:Q", title="ms/문장", format=",.0f"), alt.Tooltip(f"{col}:Q", title="match", format=".3f"),
           alt.Tooltip("style_seen:Q", title="style(본 글자)", format=".3f"),
           alt.Tooltip("fid_unseen:Q", title="FID(처음 보는 글자)", format=".2f"),
           alt.Tooltip("rec_unseen:Q", title="Recall(처음 보는 글자)", format=".3f"), alt.Tooltip("ckpt:N", title="체크포인트")]
    base = alt.Chart(d).encode(
        x=alt.X("ms:Q", scale=alt.Scale(type="log"), title="문장 하나 생성 시간 (ms, 로그 축) ← 빠름", axis=_log_axis()),
        y=alt.Y(f"{col}:Q", title=f"글자 정확도 match ({'처음 보는 글자' if which == 'unseen' else '본 글자'})",
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("device:N", scale=_dev_scale(d), title="장치"),
        shape=alt.Shape("종류:N", scale=alt.Scale(domain=[k for k in METHOD_SHAPE if k in set(d["종류"])],
                                                  range=[v for k, v in METHOD_SHAPE.items() if k in set(d["종류"])]),
                        title="모델"),
        tooltip=tip)
    pts = base.mark_point(filled=True, size=130, stroke=SURFACE, strokeWidth=2, opacity=1)
    lab = base.mark_text(align="left", dx=9, dy=-9, fontSize=11, color=INK2).encode(text="steplab:N")
    layers = [pts, lab]
    real = d["real_match_unseen"].dropna() if (which == "unseen" and "real_match_unseen" in d) else pd.Series(dtype=float)
    if len(real):
        ref = pd.DataFrame({"y": [float(real.iloc[0])], "t": ["진짜 글자 기준"]})
        layers.insert(0, alt.Chart(ref).mark_rule(color=INK2, strokeDash=[4, 4], strokeWidth=1).encode(y="y:Q"))
        layers.append(alt.Chart(ref).mark_text(align="right", dy=-6, fontSize=11, color=INK2)
                      .encode(y="y:Q", x=alt.value(560), text="t:N"))
    return _style(alt.layer(*layers).properties(
        width=560, height=320,
        title=alt.TitleParams("왼쪽 위일수록 빠르고 정확합니다", subtitle="점 옆 = 비교군·스텝 (예: C·1), 모양 = 방법, 시드 평균 · 점선 = 진짜 글자의 match")))


METRICS = [("fid_unseen", "FID ↓ (처음 보는 글자)", False), ("is_unseen", "IS ↑ (처음 보는 글자)", False),
           ("rec_unseen", "Recall ↑ (처음 보는 글자)", True), ("clip_unseen", "CLIPScore ↑ (처음 보는 글자)", False)]


def chart_quality(df):
    """설정별 품질 지표 네 개를 작은 차트로 나란히 (각자 자기 축)."""
    import altair as alt
    have = [m for m in METRICS if m[0] in df and df[m[0]].notna().any()] if not df.empty else []
    if not have:
        return _note("품질 결과가 없습니다.")
    d = df[df["speed_only"] == False].drop_duplicates(["ckpt", "device", "steps", "w"])   # noqa: E712
    long = d.melt(id_vars=["setting", "device", "forwards", "distilled", "ckpt"], value_vars=[m[0] for m in have],
                  var_name="metric", value_name="value").dropna(subset=["value"])
    long["지표"] = long["metric"].map({m[0]: m[1] for m in have})
    order = d.sort_values(["forwards", "distilled"], ascending=[True, False])["setting"].drop_duplicates().tolist()
    charts = []
    for key, title, unit in have:
        sub = long[long["metric"] == key]
        y = alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]) if unit else alt.Scale(zero=True))
        c = alt.Chart(sub).mark_point(filled=True, size=90, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
            x=alt.X("setting:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)), y=y,
            color=alt.Color("device:N", scale=_dev_scale(d), title="장치"),
            xOffset=alt.XOffset("device:N"),
            tooltip=[alt.Tooltip("지표:N"), alt.Tooltip("setting:N", title="설정"), alt.Tooltip("device:N", title="장치"),
                     alt.Tooltip("value:Q", title="값", format=".3f"), alt.Tooltip("ckpt:N", title="체크포인트")]
        ).properties(width=250, height=170, title=title)
        charts.append(c)
    rows = [alt.hconcat(*charts[i:i + 2]) for i in range(0, len(charts), 2)]
    return _style(alt.vconcat(*rows).properties(
        title=alt.TitleParams("설정별 품질", subtitle="1스텝에서 FID·IS·CLIPScore 가 유지되는지, Recall(다양성)이 줄어드는지 보세요")))


def chart_loss(tr, picked):
    import altair as alt
    if tr.empty:
        return _note("학습 기록(*_train.json)이 없습니다.")
    d = tr[tr["run"].isin(picked)] if picked else tr
    runs = sorted(d["run"].unique())[:8]                      # 색은 고정 순서 8개까지
    d = d[d["run"].isin(runs)]
    color = alt.Color("run:N", scale=alt.Scale(domain=runs, range=SERIES[:len(runs)]), title="실행")
    hover = alt.selection_point(fields=["step"], nearest=True, on="pointermove", empty=False)
    line = alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("step:Q", title="학습 스텝"), y=alt.Y("loss:Q", title="손실 (200스텝 평균)", scale=alt.Scale(zero=False)),
        color=color)
    pts = line.mark_point(filled=True, size=70, stroke=SURFACE, strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("run:N", title="실행"), alt.Tooltip("step:Q", title="스텝"),
                 alt.Tooltip("loss:Q", title="손실", format=".4f"), alt.Tooltip("minutes:Q", title="경과(분)", format=".1f")]
    ).add_params(hover)
    rule = alt.Chart(d).mark_rule(color=GRID).encode(x="step:Q").transform_filter(hover)
    return _style((line + rule + pts).properties(
        width=620, height=320, title=alt.TitleParams("학습 손실", subtitle="손실만 보지 말고 '그림' 탭의 문장 그림을 같이 보세요")))


# ================================================================ 요약
def summary_md(ev):
    if ev.empty:
        return "### 아직 결과가 없습니다\n`eval.py` 를 실행한 뒤 [새로고침] 을 누르세요."
    lines = ["| 장치 | 1스텝 학생 (PD·Reflow) | 50스텝 교사 | 빨라진 정도 | 1스텝 match (처음 보는 글자) | 증류 전 match |", "|---|---|---|---|---|---|"]
    for dev in [x for x in ("CPU", "GPU") if x in set(ev["device"])]:
        e = ev[ev["device"] == dev]
        one = e[(e["distilled"]) & (e["steps"] == 1)]
        fifty = e[(~e["distilled"]) & (e["steps"] == 50)]
        if fifty.empty:
            fifty = e[~e["distilled"]].sort_values("steps").tail(1)
        if one.empty and fifty.empty:
            continue
        o_ms = one["ms"].min() if len(one) else None
        f_ms = fifty["ms"].min() if len(fifty) else None
        o_m = one["match_unseen"].dropna().max() if "match_unseen" in one and len(one) else None
        tq = e[(~e["distilled"])].dropna(subset=["match_unseen"]) if "match_unseen" in e else e.iloc[:0]
        tq = tq.sort_values("steps").tail(1)
        f_m = tq["match_unseen"].iloc[0] if len(tq) else None
        q_lab = f" ({int(tq['steps'].iloc[0])}스텝)" if len(tq) else ""
        fs = f"{int(fifty['steps'].iloc[0])}스텝" if len(fifty) else "-"
        fmt = lambda v, u="": "-" if v is None or pd.isna(v) else f"{v:,.0f}{u}"
        acc = lambda v: "-" if v is None or pd.isna(v) else f"{v:.3f}"
        lines.append(f"| {dev} | {fmt(o_ms, ' ms')} | {fmt(f_ms, ' ms')} ({fs}) | "
                     f"{'-' if not (o_ms and f_ms) else f'{f_ms / o_ms:,.0f}배'} | {acc(o_m)} | {acc(f_m)}{q_lab} |")
    names = ev.drop_duplicates("device")[["device", "device_name"]].values.tolist()
    dev_txt = " · ".join(f"{a}: {b}" for a, b in names if b)
    return "### 1스텝 vs 50스텝\n" + "\n".join(lines) + (f"\n\n측정 기기 — {dev_txt}" if dev_txt else "")


def pm(m, sd, nd=3):
    if m is None or pd.isna(m):
        return ""
    return f"{m:.{nd}f}" + ("" if sd is None or pd.isna(sd) else f" ± {sd:.{nd}f}")


TABLE_COLS = [("setting", "설정"), ("device", "장치"), ("n_seeds", "시드 수"), ("forwards", "NFE"),
              ("ms_char", "ms/글자"), ("fid_unseen", "FID"), ("is_unseen", "IS"), ("rec_unseen", "Recall"),
              ("prec_unseen", "Precision"), ("clip_unseen", "CLIPScore"), ("match_unseen", "match"),
              ("style_seen", "필체 정확도"), ("nn_unseen", "NN-train"),
              ("student_iters", "학생 반복"), ("teacher_nfe_m", "교사 실행(백만)"), ("train_min", "1스텝 만들기(분)"),
              ("ckpt", "체크포인트")]


def _arm_rank(a):
    return ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER)


def table(ev):
    """시드 평균 ± 표준편차. 속도만 잰 결과는 품질 칸이 비어 있습니다."""
    if ev.empty:
        return pd.DataFrame(columns=[c[1] for c in TABLE_COLS])
    t = ev.copy()
    t["_r"] = t["arm"].map(_arm_rank)
    t = t.sort_values(["_r", "device", "steps", "w", "speed_only"])
    out = pd.DataFrame({
        "설정": t["setting"], "장치": t["device"] + t["speed_only"].map({True: " (속도만)", False: ""}),
        "시드 수": t["n_seeds"], "NFE": t["forwards"],
        "ms/글자": [pm(m, sd, 1) for m, sd in zip(t["ms_char"], t["ms_char_sd"])],
        **{lab: [pm(m, sd) for m, sd in zip(t[k], t[k + "_sd"])] for k, lab in
           [("fid_unseen", "FID"), ("is_unseen", "IS"), ("rec_unseen", "Recall"), ("prec_unseen", "Precision"),
            ("clip_unseen", "CLIPScore"), ("match_unseen", "match"), ("style_seen", "필체 정확도"), ("nn_unseen", "NN-train")]},
        "학생 반복": t["student_iters"].map(lambda v: "" if pd.isna(v) else f"{int(v):,}"),
        "교사 실행(백만)": t["teacher_nfe_m"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
        "1스텝 만들기(분)": t["train_min"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}"),
        "체크포인트": t["ckpt"]})
    return out


# ================================================================ 비교표 (보고서용)
CMP_METRICS = [("fid_unseen", "FID ↓", False), ("is_unseen", "IS ↑", False), ("rec_unseen", "Recall ↑", True),
               ("clip_unseen", "CLIPScore ↑", False), ("ms_char", "ms / 글자 (CPU) ↓", False)]


def compare_rows(ev, w_teacher=1.5):
    """비교군마다 대표 설정 한 줄: 학생은 1스텝, 교사(A·B)는 가장 많은 스텝과 1스텝 (guidance w_teacher)."""
    if ev.empty:
        return ev
    q = ev[~ev["speed_only"]].copy()
    sp = ev[ev["speed_only"] & (ev["device"] == "CPU")].set_index(["setting"])["ms_char"].to_dict()
    sp_sd = ev[ev["speed_only"] & (ev["device"] == "CPU")].set_index(["setting"])["ms_char_sd"].to_dict()
    keep = []
    for arm, d in q.groupby("arm"):
        if d["distilled"].all():
            keep.append(d[d["steps"] == d["steps"].min()])
        else:
            dw = d[d["w"] == w_teacher] if (d["w"] == w_teacher).any() else d
            keep.append(dw[dw["steps"].isin([dw["steps"].max(), dw["steps"].min()])])
    r = pd.concat(keep) if keep else q.iloc[:0]
    r = r[r["device"] == "CPU"] if (r["device"] == "CPU").any() else r
    r = r.drop_duplicates("setting").copy()
    r["ms_char"] = r["setting"].map(sp).fillna(r["ms_char"])
    r["ms_char_sd"] = r["setting"].map(sp_sd).fillna(r["ms_char_sd"])
    r["종류"] = r.apply(lambda x: "1스텝 학생" if x["distilled"] else ("다스텝 교사" if x["steps"] > 1 else "교사를 1스텝으로"), axis=1)
    r["_r"] = r["arm"].map(_arm_rank)
    return r.sort_values(["_r", "steps"], ascending=[True, False])


def chart_compare(ev):
    import altair as alt
    r = compare_rows(ev)
    if r.empty or r[[m[0] for m in CMP_METRICS]].isna().all().all():
        return _note("품질 결과가 없습니다. experiments.sh 의 eval 단계까지 돌리세요.")
    order = r["setting"].tolist()
    kinds = ["다스텝 교사", "교사를 1스텝으로", "1스텝 학생"]
    color = alt.Color("종류:N", scale=alt.Scale(domain=kinds, range=["#8a8985", "#eda100", SERIES[0]]), title=None)
    charts = []
    for key, title, unit in CMP_METRICS:
        d = r[["setting", "종류", "n_seeds", key, key + "_sd"]].rename(columns={key: "m", key + "_sd": "sd"}).dropna(subset=["m"])
        if d.empty:
            continue
        d["lo"] = d["m"] - d["sd"].fillna(0); d["hi"] = d["m"] + d["sd"].fillna(0)
        x = alt.X("m:Q", title=None, scale=alt.Scale(domain=[0, 1]) if unit else alt.Scale(zero=True))
        y = alt.Y("setting:N", sort=order, title=None, axis=alt.Axis(labels=not charts, labelLimit=220))
        tip = [alt.Tooltip("setting:N", title="설정"), alt.Tooltip("m:Q", title="평균", format=".3f"),
               alt.Tooltip("sd:Q", title="표준편차", format=".3f"), alt.Tooltip("n_seeds:Q", title="시드 수")]
        err = alt.Chart(d).mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q", y=y, color=color)
        pt = alt.Chart(d).mark_point(filled=True, size=90, stroke=SURFACE, strokeWidth=2, opacity=1).encode(x=x, y=y, color=color, tooltip=tip)
        charts.append((err + pt).properties(width=150 if charts else 150, height=26 * len(order), title=title))
    return _style(alt.hconcat(*charts, spacing=14).properties(
        title=alt.TitleParams("비교군별 대표 설정 (시드 평균 ± 표준편차)",
                              subtitle="학생은 1스텝, 교사는 가장 많은 스텝과 1스텝 (w=1.5). 막대 = ±1 표준편차")))


def compare_table(ev):
    r = compare_rows(ev)
    if r.empty:
        return "결과가 없습니다."
    head = "| 비교군 · 설정 | 시드 | NFE | " + " | ".join(m[1] for m in CMP_METRICS) + " | 학생 반복 | 교사 실행(백만) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for _, x in r.iterrows():
        vals = [pm(x[k], x[k + "_sd"], 1 if k == "ms_char" else 3) or "-" for k, _, _ in CMP_METRICS]
        si = "" if pd.isna(x["student_iters"]) else f"{int(x['student_iters']):,}"
        tn = "" if pd.isna(x["teacher_nfe_m"]) else f"{x['teacher_nfe_m']:.1f}"
        lines.append(f"| {x['setting']} | {int(x['n_seeds'])} | {int(x['forwards'])} | " + " | ".join(vals) + f" | {si} | {tn} |")
    return "\n".join(lines) + "\n\n표를 그대로 복사해 보고서에 붙일 수 있습니다 (마크다운). 학생 반복·교사 실행은 1스텝을 만드는 데 든 계산량입니다."


# ================================================================ 화면
def build_app(runs):
    import gradio as gr

    def refresh(which, picked):
        ev, tr = aggregate(load_eval(runs)), load_train(runs)
        tags = sorted(tr["run"].unique()) if not tr.empty else []
        picked = [p for p in (picked or []) if p in tags] or tags[:8]
        dirs = [os.path.basename(d) for d in grid_dirs(runs)]
        evals = sorted(glob.glob(os.path.join(runs, "eval", "grid_*.png")))
        return (compare_table(ev), chart_compare(ev), summary_md(ev), table(ev), chart_speed(ev), chart_forwards(ev), chart_tradeoff(ev, which),
                chart_quality(ev), gr.update(choices=tags, value=picked), chart_loss(tr, picked),
                gr.update(choices=dirs, value=dirs[0] if dirs else None),
                [(p, os.path.basename(p)[5:-4]) for p in evals])

    def show_grid(tag):
        if not tag:
            return gr.update(maximum=1, value=0), None, ""
        files = sorted(glob.glob(os.path.join(runs, "grids", tag, "step*.png")))
        if not files:
            return gr.update(maximum=1, value=0), None, "그림 없음"
        cap = pick_grid(tag, len(files) - 1)[1]
        return gr.update(maximum=max(len(files) - 1, 1), value=len(files) - 1), files[-1], cap

    def pick_grid(tag, i):
        files = sorted(glob.glob(os.path.join(runs, "grids", tag or "", "step*.png")))
        if not files:
            return None, ""
        i = int(min(max(i, 0), len(files) - 1))
        step = re.sub(r"\D", "", os.path.basename(files[i])).lstrip("0") or "0"
        return files[i], f"학습 스텝 {int(step):,}  ({i + 1}/{len(files)})"

    with gr.Blocks(title="실행 결과 대시보드") as app:
        gr.Markdown(f"## 실행 결과 대시보드\n`{os.path.abspath(runs)}` 의 평가·학습 기록을 보여줍니다. "
                    "학습이나 평가가 끝나면 **새로고침**을 누르세요.")
        btn = gr.Button("새로고침", variant="primary")
        with gr.Tab("비교표"):
            p_cmp = gr.Plot(label="비교군별 대표 설정", show_label=False)
            cmp_md = gr.Markdown()
        with gr.Tab("요약"):
            summ = gr.Markdown()
            tbl = gr.Dataframe(label="모든 평가 결과", wrap=True)
        with gr.Tab("속도와 품질"):
            which = gr.Radio([("처음 보는 글자", "unseen"), ("본 글자", "seen")], value="unseen",
                             label="세로축 match 기준")
            p_trade = gr.Plot(label="속도-정확도")
            p_speed = gr.Plot(label="생성 시간")
            p_fw = gr.Plot(label="모델 실행 횟수")
            p_q = gr.Plot(label="품질 지표")
        with gr.Tab("학습 곡선"):
            picked = gr.CheckboxGroup(label="보여줄 실행 (최대 8개)", choices=[])
            p_loss = gr.Plot(label="손실")
        with gr.Tab("그림"):
            with gr.Row():
                tag = gr.Dropdown(label="학습 실행", choices=[])
                idx = gr.Slider(0, 1, value=0, step=1, label="학습 중 시점 (왼쪽 = 처음, 오른쪽 = 마지막)")
            cap = gr.Markdown()
            img = gr.Image(label="고정 시드 문장 그림 (같은 문장·같은 시드)", type="filepath")
            gal = gr.Gallery(label="평가 그리드 (큐레이션 없음)", columns=2, height=560, object_fit="contain")

        outs = [cmp_md, p_cmp, summ, tbl, p_speed, p_fw, p_trade, p_q, picked, p_loss, tag, gal]
        btn.click(refresh, [which, picked], outs)
        app.load(refresh, [which, picked], outs)
        which.change(lambda w: chart_tradeoff(aggregate(load_eval(runs)), w), which, p_trade)
        picked.change(lambda p: chart_loss(load_train(runs), p), picked, p_loss)
        tag.change(show_grid, tag, [idx, img, cap])
        idx.release(pick_grid, [tag, idx], [img, cap])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="train.py·eval.py 의 --out 폴더")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_app(a.runs).launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
