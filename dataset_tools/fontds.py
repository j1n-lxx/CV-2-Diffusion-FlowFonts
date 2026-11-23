"""공통 렌더링 모듈: 한글 2,350자 + 추가 문자(숫자·영문·문장부호·기호).

한글 규칙(1차와 동일): 폰트별 크기 1개(샘플 한글 잉크 크기 98퍼센타일 = 54/64px),
  글자마다 잉크 중심을 캔버스 중심에 맞춤, 3배 슈퍼샘플링 후 BOX 축소.
추가 문자 규칙: 한글과 같은 크기·같은 기준선. 세로 위치는 '한글을 가운데 맞출 때 쓰는 평균 세로 이동량'을
  그대로 적용해 원래 높이를 보존(마침표는 아래, 따옴표는 위). 가로는 잉크 중심을 가운데로.
  캔버스를 벗어나면 안쪽으로 밀고(shifted), 그래도 크면 축소(shrunk).
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S, SS = 64, 3
CANVAS = S * SS
TARGET = 54 * SS
WORK = CANVAS * 2
ORG = WORK // 4
MARGIN = 2 * SS


def ks_chars():
    out = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            try:
                out.append(bytes([hi, lo]).decode("euc-kr"))
            except UnicodeDecodeError:
                pass
    assert len(out) == 2350
    return out


HANGUL = ks_chars()
DIGITS = [chr(c) for c in range(0x30, 0x3A)]
UPPER = [chr(c) for c in range(0x41, 0x5B)]
LOWER = [chr(c) for c in range(0x61, 0x7B)]
ASCII_PUNCT = [chr(c) for c in range(0x21, 0x7F) if not chr(c).isalnum()]
CURLY = ["‘", "’", "“", "”"]
SYMBOL = ["♡"]  # ♡
EXTRA = DIGITS + UPPER + LOWER + ASCII_PUNCT + CURLY + SYMBOL
EXTRA_KIND = (["digit"] * 10 + ["latin_upper"] * 26 + ["latin_lower"] * 26 + ["punct"] * len(ASCII_PUNCT)
              + ["punct"] * 4 + ["symbol"])
assert len(EXTRA) == 99 == len(EXTRA_KIND)
ALL_CHARS = HANGUL + EXTRA


def load_font(path, size, index=0, instance=None):
    f = ImageFont.truetype(path, size, index=index)
    if instance:
        f.set_variation_by_name(instance)
    return f


def draw(font, ch):
    im = Image.new("L", (WORK, WORK), 0)
    ImageDraw.Draw(im).text((ORG, ORG), ch, font=font, fill=255)
    return im, im.getbbox()


def font_scale(path, index=0, instance=None):
    f0 = load_font(path, 128, index, instance)
    ext = []
    for ch in HANGUL[::10]:
        _, bb = draw(f0, ch)
        if bb:
            ext.append(max(bb[2] - bb[0], bb[3] - bb[1]))
    m = float(np.percentile(ext, 98))
    return max(8, int(round(128 * TARGET / m)))


def place_center(im, bb):
    g = im.crop(bb); w, h = g.size
    shrunk = 0
    lim = CANVAS - 2 * SS
    if max(w, h) > lim:
        k = lim / max(w, h)
        g = g.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS); w, h = g.size
        shrunk = 1
    c = Image.new("L", (CANVAS, CANVAS), 0)
    c.paste(g, ((CANVAS - w) // 2, (CANVAS - h) // 2))
    return np.asarray(c.resize((S, S), Image.BOX)), shrunk


def render_hangul(font):
    X = np.zeros((len(HANGUL), S, S), np.uint8)
    blank = shrunk = 0
    dys = []
    for j, ch in enumerate(HANGUL):
        im, bb = draw(font, ch)
        if bb is None:
            blank += 1
            continue
        X[j], s = place_center(im, bb)
        shrunk += s
        dys.append((bb[1] + bb[3]) / 2)
    return X, dict(blank=blank, shrunk=shrunk, dup=len(HANGUL) - len({x.tobytes() for x in X})), dys


def hangul_center_y(font):
    """한글 잉크 세로 중심의 중앙값 (작업 캔버스 좌표). 추가 문자의 기준선 보존에 사용."""
    ys = []
    for ch in HANGUL[::7]:
        _, bb = draw(font, ch)
        if bb:
            ys.append((bb[1] + bb[3]) / 2)
    return float(np.median(ys))


def render_extra(font, cmap, cy):
    """cmap: 폰트가 지원하는 코드포인트 집합. 지원 안 하면 avail=0 (두부 방지)."""
    X = np.zeros((len(EXTRA), S, S), np.uint8)
    avail = np.zeros(len(EXTRA), bool)
    shifted = shrunk = 0
    for j, ch in enumerate(EXTRA):
        if ord(ch) not in cmap:
            continue
        im, bb = draw(font, ch)
        if bb is None:
            continue
        g = im.crop(bb); w, h = g.size
        lim = CANVAS - 2 * MARGIN
        if max(w, h) > lim:
            k = lim / max(w, h)
            g = g.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS); w, h = g.size
            shrunk += 1
        # 세로: 한글 중심(cy)이 캔버스 중심에 오도록 한 것과 같은 이동량
        y0 = int(round(bb[1] + (bb[3] - bb[1] - h) / 2 - cy + CANVAS / 2))
        if y0 < MARGIN or y0 + h > CANVAS - MARGIN:
            y0 = min(max(y0, MARGIN), CANVAS - MARGIN - h)
            shifted += 1
        c = Image.new("L", (CANVAS, CANVAS), 0)
        c.paste(g, ((CANVAS - w) // 2, y0))
        X[j] = np.asarray(c.resize((S, S), Image.BOX))
        avail[j] = True
    return X, avail, dict(extra_avail=int(avail.sum()), extra_shifted=shifted, extra_shrunk=shrunk)
