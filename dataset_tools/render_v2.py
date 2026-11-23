"""렌더링 v2.
사용:
  python3 -I render_v2.py ft <ft_src_dir> <out_dir>        # 나눔손글씨: 한글 + 추가 문자
  python3 -I render_v2.py pre <harvest_work> <out_dir>     # 사전학습: 추가 문자만 (한글은 기존 faces 재사용)
"""
import glob, hashlib, json, os, sys
from multiprocessing import Pool
import numpy as np
from fontTools.ttLib import TTFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontds as F

MODE, SRC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def ft_job(args):
    i, path = args
    base = os.path.join(OUT, f"{i:04d}")
    if os.path.exists(base + ".json"):
        return
    try:
        size = F.font_scale(path)
        font = F.load_font(path, size)
        X, st, _ = F.render_hangul(font)
        cmap = set(TTFont(path, lazy=True).getBestCmap())
        XE, av, st2 = F.render_extra(font, cmap, F.hangul_center_y(font))
        st.update(st2)
        st.update(size_px=size, ink=round(float(X.mean()) / 255, 4), usable=bool(st["blank"] == 0 and st["dup"] <= 12))
        np.savez_compressed(base + ".npz", x=X, xe=XE, avail=av)
    except Exception as e:
        st = dict(usable=False, error=repr(e)[:200])
    st["file"] = path
    json.dump(st, open(base + ".json", "w"))


def pre_job(args):
    i, spec, size = args
    base = os.path.join(OUT, f"{i:04d}")
    if os.path.exists(base + ".json"):
        return
    path = os.path.join(SRC, "keep", spec["kept_as"])
    try:
        font = F.load_font(path, size, spec.get("face", 0), spec.get("instance"))
        tt = TTFont(path, lazy=True, fontNumber=spec.get("face", 0)) if path.lower().endswith((".ttc", ".otc")) else TTFont(path, lazy=True)
        cmap = set(tt.getBestCmap())
        XE, av, st = F.render_extra(font, cmap, F.hangul_center_y(font))
        np.savez_compressed(base + ".npz", xe=XE, avail=av)
    except Exception as e:
        st = dict(error=repr(e)[:200])
    json.dump(st, open(base + ".json", "w"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if MODE == "ft":
        paths = sorted(glob.glob(os.path.join(SRC, "**", "*.ttf"), recursive=True))
        jobs = list(enumerate(paths))
        fn = ft_job
    else:
        sel = json.load(open(os.path.join(SRC, "selected.json")))
        jobs = []
        for i, s in enumerate(sel):
            st = json.load(open(os.path.join(SRC, "faces", f"{i:04d}.json")))
            if st.get("usable"):
                jobs.append((i, s, st["size_px"]))
        fn = pre_job
    with Pool(2) as p:
        for _ in p.imap_unordered(fn, jobs, chunksize=2):
            pass
    print(MODE, "done", len(jobs))
