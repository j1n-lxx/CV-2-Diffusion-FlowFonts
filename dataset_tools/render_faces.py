"""selected.json의 각 face를 KS X 1001 한글 2,350자 × 64×64 uint8(흰 글자/검은 배경)로 렌더링한다.

사용: python3 -I render_faces.py <work_dir> [--procs 2]
결과: <work_dir>/faces/NNNN.npz (x: (2350,64,64)) + NNNN.json (품질 통계). 재실행 시 이미 만든 것은 건너뜀.
렌더링 규칙은 1차 데이터셋(build_font_dataset.py)과 동일:
  폰트별 스케일 1개(샘플 글자 ink 크기 98퍼센타일 = 54/64px), 글자별 ink bbox 중심 정렬, 3배 슈퍼샘플링 후 BOX 축소.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORK = sys.argv[1]
PROCS = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 2
KEEP = os.path.join(WORK, "keep"); OUT = os.path.join(WORK, "faces")
S, SS = 64, 3
CANVAS = S * SS; TARGET = 54 * SS; WORK_PX = CANVAS * 2; ORG = WORK_PX // 4
MAX_DUP = 12


def chars():
    out = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            try:
                out.append(bytes([hi, lo]).decode("euc-kr"))
            except UnicodeDecodeError:
                pass
    return out


CHARS = chars()


def load_font(spec, size):
    f = ImageFont.truetype(os.path.join(KEEP, spec["kept_as"]), size, index=spec.get("face", 0))
    if spec.get("instance"):
        f.set_variation_by_name(spec["instance"])
    return f


def draw(font, ch):
    im = Image.new("L", (WORK_PX, WORK_PX), 0)
    ImageDraw.Draw(im).text((ORG, ORG), ch, font=font, fill=255)
    return im, im.getbbox()


def render(args):
    i, spec = args
    base = os.path.join(OUT, f"{i:04d}")
    if os.path.exists(base + ".json"):
        return
    st = {}
    try:
        f0 = load_font(spec, 128)
        ext = []
        for ch in CHARS[::10]:
            _, bb = draw(f0, ch)
            if bb:
                ext.append(max(bb[2] - bb[0], bb[3] - bb[1]))
        if not ext:
            raise RuntimeError("all sample glyphs empty")
        m = float(np.percentile(ext, 98))
        size = max(8, int(round(128 * TARGET / m)))
        font = load_font(spec, size)
        X = np.zeros((len(CHARS), S, S), np.uint8)
        blank = shrunk = 0
        lim = CANVAS - 2 * SS
        for j, ch in enumerate(CHARS):
            im, bb = draw(font, ch)
            if bb is None:
                blank += 1
                continue
            g = im.crop(bb); w, h = g.size
            if max(w, h) > lim:
                k = lim / max(w, h)
                g = g.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS); w, h = g.size
                shrunk += 1
            c = Image.new("L", (CANVAS, CANVAS), 0)
            c.paste(g, ((CANVAS - w) // 2, (CANVAS - h) // 2))
            X[j] = np.asarray(c.resize((S, S), Image.BOX))
        dup = len(CHARS) - len({x.tobytes() for x in X})
        st = dict(size_px=size, blank=blank, shrunk=shrunk, dup=dup, ink=round(float(X.mean()) / 255, 4),
                  usable=bool(blank == 0 and dup <= MAX_DUP))
        np.savez_compressed(base + ".npz", x=X)
    except Exception as e:
        st = dict(usable=False, error=repr(e)[:200])
    json.dump(st, open(base + ".json", "w"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(os.path.join(WORK, "selected.json")))
    with Pool(PROCS) as p:
        for _ in p.imap_unordered(render, list(enumerate(sel)), chunksize=2):
            pass
    st = [json.load(open(os.path.join(OUT, f"{i:04d}.json"))) for i in range(len(sel))]
    print("rendered", len(st), "usable", sum(s.get("usable", False) for s in st))
