"""FreeType 이 힌팅 코드를 읽지 못하는 폰트('too many function definitions')를 힌팅 없이 다시 렌더링합니다.
글자 모양(윤곽선)은 그대로이고 힌팅 명령만 지웁니다. 원본 폰트 파일은 건드리지 않고 복사본을 만듭니다.
(나눔손글씨 동화또박·하나손글씨 2종이 여기에 걸립니다.)

사용:
  python3 -I dehint.py pre <work>/harvest_work   # 다음에 render_faces.py 를 한 번 더 돌리세요
  python3 -I dehint.py ft  <work>/ft_faces       # 바로 다시 렌더링까지 합니다
"""
import glob
import json
import os
import sys

import numpy as np
from fontTools.ttLib import TTFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontds as F

MODE, DIR = sys.argv[1], sys.argv[2]


def dehint(src, dst):
    t = TTFont(src)
    for tag in ("fpgm", "prep", "cvt "):
        if tag in t:
            del t[tag]
    if "glyf" in t:
        for g in t["glyf"].glyphs.values():
            if hasattr(g, "removeHinting"):
                g.removeHinting()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t.save(dst)


if MODE == "pre":
    sel_path = os.path.join(DIR, "selected.json")
    sel = json.load(open(sel_path))
    for i, f in enumerate(sel):
        p = os.path.join(DIR, "faces", f"{i:04d}.json")
        if not os.path.exists(p) or "function definitions" not in json.load(open(p)).get("error", ""):
            continue
        src = os.path.join(DIR, "keep", f["kept_as"])
        dst = src[:-4] + "_dehinted.ttf"
        dehint(src, dst)
        f["kept_as"] = os.path.relpath(dst, os.path.join(DIR, "keep"))
        f["note"] = "hinting removed (FreeType could not load hinting bytecode)"
        os.remove(p)
        print("dehinted:", f["family"])
    json.dump(sel, open(sel_path, "w"), ensure_ascii=False, indent=0)
    print("이제 render_faces.py 를 다시 돌리세요.")
else:
    for fn in sorted(glob.glob(os.path.join(DIR, "*.json"))):
        st = json.load(open(fn))
        if "function definitions" not in st.get("error", ""):
            continue
        p = st["file"]
        q = p.replace("/ft_src/", "/ft_dehinted/")
        dehint(p, q)
        size = F.font_scale(q); font = F.load_font(q, size)
        X, s, _ = F.render_hangul(font)
        XE, av, s2 = F.render_extra(font, set(TTFont(q, lazy=True).getBestCmap()), F.hangul_center_y(font))
        s.update(s2)
        s.update(size_px=size, ink=round(float(X.mean()) / 255, 4), usable=bool(s["blank"] == 0 and s["dup"] <= 12),
                 file=p, render_file=q, note="hinting removed (FreeType could not load hinting bytecode)")
        np.savez_compressed(fn[:-5] + ".npz", x=X, xe=XE, avail=av)
        json.dump(s, open(fn, "w"))
        print("dehinted + rendered:", os.path.basename(p))
