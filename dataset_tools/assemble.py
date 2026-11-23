"""렌더링된 face들을 패밀리 단위 train/val/test + 글자 holdout으로 나눠 최종 데이터셋 폴더를 만든다.

사용: python3 -I assemble.py <work_dir> <out_dir> <scratch_dir>
"""
import csv, hashlib, json, os, random, re, shutil, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

WORK, OUT, SCR = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = 42
S = 64
FACES = os.path.join(WORK, "faces"); KEEP = os.path.join(WORK, "keep")

sel = json.load(open(os.path.join(WORK, "selected.json")))
gf_api = json.load(open(os.path.join(SCR, "gfm/package/data/api-response.json")))
gf_cat = {f["family"].lower(): f["category"] for f in gf_api}
npm_desc = {}
for fn in ("npm_manifests.json", "scope_manifests.json"):
    for k, v in json.load(open(os.path.join(SCR, fn))).items():
        if v:
            npm_desc[k] = v.get("desc") or ""

HAND_GF = ("gaegu", "hi melody", "single day", "east sea dokdo", "gamja flower", "poor story", "yeon sung",
           "cute font", "dokdo", "kirang haerang", "black and white picture", "dongle", "nanum pen", "nanum brush script")


def category(f):
    fam = f["family"].lower(); pkg = (f.get("pkg") or "").lower(); desc = npm_desc.get(f.get("pkg") or "", "")
    if re.search(r"galmuri|neodgm|dunggeunmo|둥근모|neo둥근모", fam + pkg) or re.search(r"픽셀|도트|비트맵|bitmap|pixel", desc, re.I):
        return "pixel"
    if f["src"].startswith("gf/"):
        if gf_cat.get(fam) == "handwriting" or fam.startswith(HAND_GF):
            return "handwriting"
        return "print"
    if re.search(r"handwrit|nanum-pen|nanum-brush|barun-pen", pkg) or "손글씨" in desc or "손글씨" in fam:
        return "handwriting"
    return "print"


def fam_key(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def ks_chars():
    out = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            try:
                out.append(bytes([hi, lo]).decode("euc-kr"))
            except UnicodeDecodeError:
                pass
    return out


CHARS = ks_chars()

# ---- 사용 가능한 face만 ----
faces = []
for i, f in enumerate(sel):
    st = json.load(open(os.path.join(FACES, f"{i:04d}.json")))
    if st.get("usable"):
        faces.append(dict(f, render_idx=i, **{k: st[k] for k in ("size_px", "ink", "dup")}, category=category(f)))
for fid, f in enumerate(faces):
    f["font_id"] = fid
    f["family_key"] = fam_key(f["family"])

# ---- 패밀리 단위 층화 분할 ----
rng = random.Random(SEED)
fam_cat = {}
for f in faces:
    fam_cat.setdefault(f["family_key"], f["category"])
fam_split = {}
for cat in sorted(set(fam_cat.values())):
    fams = sorted(k for k, c in fam_cat.items() if c == cat)
    rng.shuffle(fams)
    n = len(fams)
    n_test = max(1, round(n * 0.1)) if n >= 3 else 0
    n_val = max(1, round(n * 0.1)) if n >= 3 else 0
    for k in fams[:n_test]:
        fam_split[k] = "test"
    for k in fams[n_test:n_test + n_val]:
        fam_split[k] = "val"
    for k in fams[n_test + n_val:]:
        fam_split[k] = "train"
for f in faces:
    f["split"] = fam_split[f["family_key"]]

# ---- 글자 holdout (10%) ----
crng = random.Random(SEED + 1)
held = set(crng.sample(range(len(CHARS)), round(len(CHARS) * 0.1)))
seen_ids = [j for j in range(len(CHARS)) if j not in held]
held_ids = sorted(held)
all_ids = list(range(len(CHARS)))

# dev subset: train 패밀리 중 카테고리 비율대로 20개 패밀리의 첫 face × seen 글자 500자
drng = random.Random(SEED + 2)
train_fams = sorted({f["family_key"] for f in faces if f["split"] == "train"})
by_cat = defaultdict(list)
for k in train_fams:
    by_cat[fam_cat[k]].append(k)
dev_fams = []
for cat, ks in sorted(by_cat.items()):
    take = max(1, round(20 * len(ks) / len(train_fams)))
    dev_fams += drng.sample(ks, min(take, len(ks)))
dev_fams = set(dev_fams[:20])
dev_faces = []
for k in sorted(dev_fams):
    cand = [f for f in faces if f["family_key"] == k]
    reg = [f for f in cand if re.sub(r"[^a-z]", "", f["style"].lower()) in ("regular", "")]
    dev_faces.append((reg or cand)[0])
dev_chars = sorted(drng.sample(seen_ids, 500))

PLAN = {
    "train": ([f for f in faces if f["split"] == "train"], seen_ids),
    "train_unseen_chars": ([f for f in faces if f["split"] == "train"], held_ids),
    "val": ([f for f in faces if f["split"] == "val"], all_ids),
    "test": ([f for f in faces if f["split"] == "test"], all_ids),
    "dev": (dev_faces, dev_chars),
}

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
cache = {}


def face_x(f):
    if f["font_id"] not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[f["font_id"]] = np.load(os.path.join(FACES, f"{f['render_idx']:04d}.npz"))["x"]
    return cache[f["font_id"]]


counts = {}
for name, (fs, cids) in PLAN.items():
    d = os.path.join(OUT, name); os.makedirs(d)
    n = len(fs) * len(cids)
    X = np.lib.format.open_memmap(os.path.join(d, "images.npy"), mode="w+", dtype=np.uint8, shape=(n, S, S))
    L = np.zeros((n, 2), np.int32)
    o = 0
    cid_arr = np.array(cids)
    for f in fs:
        X[o:o + len(cids)] = face_x(f)[cid_arr]
        L[o:o + len(cids), 0] = f["font_id"]; L[o:o + len(cids), 1] = cid_arr
        o += len(cids)
    X.flush(); del X
    np.save(os.path.join(d, "labels.npy"), L)
    counts[name] = dict(fonts=len(fs), families=len({f["family_key"] for f in fs}), chars=len(cids), images=n)

# ---- 메타데이터 ----
md = os.path.join(OUT, "meta"); os.makedirs(md)
with open(os.path.join(md, "fonts.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["font_id", "family", "style", "family_key", "category", "split", "in_dev", "license", "source", "source_url",
                "package_version", "file", "sha256", "size_px", "ink_mean"])
    for f in faces:
        w.writerow([f["font_id"], f["family"], f["style"], f["family_key"], f["category"], f["split"], int(f in dev_faces),
                    f["lic_final"], f["src"], f["url"], f.get("version") or "", os.path.basename(f["file"]) +
                    (f"#{f['instance']}" if f.get("instance") else ""), f["sha256"], f["size_px"], f["ink"]])
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
with open(os.path.join(md, "chars.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["char_id", "char", "codepoint", "cho", "jung", "jong", "cho_char", "jung_char", "jong_char", "heldout", "in_dev"])
    dset = set(dev_chars)
    for j, ch in enumerate(CHARS):
        s = ord(ch) - 0xAC00
        c, v, t = s // 588, (s % 588) // 28, s % 28
        w.writerow([j, ch, f"U+{ord(ch):04X}", c, v, t, CHO[c], JUNG[v], JONG[t], int(j in held), int(j in dset)])
open(os.path.join(md, "chars.txt"), "w", encoding="utf-8").write("".join(CHARS))
json.dump(dict(seed=SEED, family_split={k: fam_split[k] for k in sorted(fam_split)},
               family_category={k: fam_cat[k] for k in sorted(fam_cat)},
               heldout_char_ids=held_ids, dev_font_ids=[f["font_id"] for f in dev_faces], dev_char_ids=dev_chars,
               counts=counts), open(os.path.join(md, "splits.json"), "w"), ensure_ascii=False, indent=1)

# ---- 출처·라이선스 ----
pv = os.path.join(OUT, "provenance"); os.makedirs(os.path.join(pv, "licenses"))
lic_map = {}
with open(os.path.join(pv, "sources.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["font_id", "family", "style", "license", "license_basis", "license_name_table", "license_url", "copyright",
                "source", "source_url", "package_version", "file", "sha256", "license_file"])
    for f in faces:
        kdir = os.path.join(KEEP, os.path.dirname(f["kept_as"]))
        lf = ""
        cands = sorted(p for p in os.listdir(kdir) if p.startswith("LICENSE__"))
        pick = [p for p in cands if re.search(r"ofl|licen", p, re.I)] or cands
        if pick:
            txt = open(os.path.join(kdir, pick[0]), encoding="utf-8", errors="replace").read()
            h = hashlib.sha256(txt.encode()).hexdigest()[:12]
            lf = f"licenses/{re.sub(r'[^A-Za-z0-9]+', '_', f['family'])}_{h}.txt"
            if h not in lic_map:
                open(os.path.join(pv, lf), "w", encoding="utf-8").write(txt); lic_map[h] = lf
            lf = lic_map[h]
        w.writerow([f["font_id"], f["family"], f["style"], f["lic_final"], f["lic_basis"], f["lic_desc"][:300], f["lic_url"], f["copyright"],
                    f["src"], f["url"], f.get("version") or "", f["file"], f["sha256"], lf])
shutil.copy(os.path.join(WORK, "excluded.csv"), os.path.join(pv, "excluded.csv"))

# ---- 미리보기 ----
pr = os.path.join(OUT, "previews"); os.makedirs(pr)
for name in ("train", "val", "test"):
    fs = [f for f in faces if f["split"] == name]
    rr = random.Random(SEED + 3)
    pick = rr.sample(fs, min(40, len(fs)))
    cols = [0, 300, 700, 1100, 1500, 1900, 2349]
    img = Image.new("L", (len(cols) * 66, len(pick) * 66), 255)
    for r, f in enumerate(pick):
        x = face_x(f)
        for c, j in enumerate(cols):
            img.paste(Image.fromarray(255 - x[j]), (c * 66, r * 66))
    img.save(os.path.join(pr, f"{name}_sample.png"))

stats = dict(
    n_fonts=len(faces), n_families=len(fam_cat), counts=counts,
    cat_fonts=dict(Counter(f["category"] for f in faces)), cat_families=dict(Counter(fam_cat.values())),
    lic=dict(Counter(f["lic_final"] for f in faces)),
    src=dict(Counter("google-fonts" if f["src"].startswith("gf/") else ("@kfonts" if "@kfonts/" in f["src"] else ("@noonnu" if "@noonnu/" in f["src"] else "npm-other")) for f in faces)),
    split_cat={s: dict(Counter(fam_cat[k] for k, v in fam_split.items() if v == s)) for s in ("train", "val", "test")},
    bytes=sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns),
)
json.dump(stats, open(os.path.join(WORK, "assemble_stats.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
