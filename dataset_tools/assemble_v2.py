"""v2 데이터셋 조립 (한글 2,350 + 추가 문자 99 = char_id 0..2448).

사용 (<work> = 작업 폴더. 순서는 README 의 Dataset 참고):
  python3 -I assemble_v2.py pre <work>/harvest_work <work>/build_v2/font_ko_pretrain <work>       # 사전학습 (v1과 같은 분할)
  python3 -I assemble_v2.py ft  <work>/ft_faces     <work>/build_v2/font_ko_finetune_nanum <work> # 나눔손글씨 (pre 다음에)
"""
import csv, hashlib, json, os, random, re, shutil, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

MODE, SRC, OUT, SCR = sys.argv[1:5]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fontds as F

SEED = 42
S = 64
NH, NE = len(F.HANGUL), len(F.EXTRA)
NCH = NH + NE


def fam_key(s):
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


# 글자 holdout·dev 글자: v1과 같은 시드·같은 코드 → 같은 목록
crng = random.Random(SEED + 1)
held = set(crng.sample(range(NH), round(NH * 0.1)))
seen_ids = [j for j in range(NH) if j not in held]
held_ids = sorted(held)
EXTRA_IDS = list(range(NH, NCH))

faces = []
if MODE == "pre":
    WORK = SRC
    sel = json.load(open(os.path.join(WORK, "selected.json")))
    v1 = {}
    with open(os.path.join(SCR, "build/font_ko_pretrain/meta/fonts.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v1[(r["family"], r["style"], r["sha256"])] = r
    for i, f in enumerate(sel):
        st = json.load(open(os.path.join(WORK, "faces", f"{i:04d}.json")))
        if not st.get("usable"):
            continue
        r = v1[(f["family"], f["style"], f["sha256"])]
        faces.append(dict(f, idx=i, size_px=st["size_px"], ink=st["ink"], category=r["category"], split=r["split"],
                          family_key=r["family_key"], v1_id=int(r["font_id"]), in_dev=r["in_dev"] == "1"))
    faces.sort(key=lambda f: f["v1_id"])
    for fid, f in enumerate(faces):
        assert fid == f["v1_id"], "font_id가 v1과 달라짐"
        f["font_id"] = fid

    def load_face(f):
        x = np.load(os.path.join(WORK, "faces", f"{f['idx']:04d}.npz"))["x"]
        e = np.load(os.path.join(WORK, "extras", f"{f['idx']:04d}.npz"))
        return np.concatenate([x, e["xe"]]), np.concatenate([np.ones(NH, bool), e["avail"]])
    # 파인튜닝 세트와 같은 폰트 표시
    ft_fams = set()
    ftd = os.path.join(SCR, "ft_faces")
    for fn in os.listdir(ftd):
        if fn.endswith(".json"):
            from fontTools.ttLib import TTFont
            p = json.load(open(os.path.join(ftd, fn)))["file"]
            nmt = TTFont(p, lazy=True)["name"]
            for rid in (1, 4):
                for rec in nmt.names:
                    if rec.nameID == rid:
                        try:
                            ft_fams.add(fam_key(rec.toUnicode()))
                        except Exception:
                            pass
    for f in faces:
        f["nanum_handwriting"] = int("나눔손글씨" in f["family"] or "nanum-handwrit" in (f.get("pkg") or "")
                                     or f["family_key"].startswith(("nanumpen", "nanumbrush")))
        f["in_finetune_set"] = int(f["family_key"] in ft_fams or fam_key(f.get("full", "")) in ft_fams)
    dev_faces = [f for f in faces if f["in_dev"]]
    drng = random.Random(SEED + 2)
    # v1 dev 글자 재현: v1 splits.json에서 그대로 읽음
    v1s = json.load(open(os.path.join(SCR, "build/font_ko_pretrain/meta/splits.json")))
    assert v1s["heldout_char_ids"] == held_ids
    dev_chars = v1s["dev_char_ids"]
    fam_split = v1s["family_split"]; fam_cat = v1s["family_category"]
else:
    from fontTools.ttLib import TTFont
    rows = sorted(fn for fn in os.listdir(SRC) if fn.endswith(".json"))
    for fn in rows:
        st = json.load(open(os.path.join(SRC, fn)))
        if not st.get("usable"):
            print("제외:", st.get("file"), st.get("error", ""), st.get("blank"), st.get("dup"))
            continue
        p = st["file"]
        nm = TTFont(p, lazy=True)["name"]
        fam = str(nm.getName(1, 3, 1, 0x412) or nm.getName(1, 3, 1, 0x409) or os.path.basename(p)[:-4])
        en = str(nm.getName(1, 3, 1, 0x409) or "")
        faces.append(dict(idx=int(fn[:4]), family=fam, family_en=en, style="Regular", path=p, size_px=st["size_px"], ink=st["ink"],
                          category="handwriting", family_key=fam_key(fam),
                          sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                          extra_avail=st["extra_avail"], extra_shifted=st["extra_shifted"]))
    faces.sort(key=lambda f: f["family"])
    for fid, f in enumerate(faces):
        f["font_id"] = fid

    def load_face(f):
        d = np.load(os.path.join(SRC, f"{f['idx']:04d}.npz"))
        return np.concatenate([d["x"], d["xe"]]), np.concatenate([np.ones(NH, bool), d["avail"]])
    # 사전학습 세트에 같은 폰트가 있으면 그 split을 그대로 따름 (사전학습→파인튜닝 평가에서 test 필체 누출 방지)
    pre_split = {}
    with open(os.path.join(SCR, "build_v2/font_ko_pretrain/meta/fonts.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["in_finetune_set"] == "1":
                pre_split[r["family_key"]] = r["split"]
    keys = sorted({f["family_key"] for f in faces})
    unmatched = [k for k in keys if k not in pre_split]
    rng = random.Random(SEED); rng.shuffle(unmatched)
    n = len(unmatched); nt = round(n * 0.1); nv = round(n * 0.1)
    fam_split = {k: ("test" if i < nt else "val" if i < nt + nv else "train") for i, k in enumerate(unmatched)}
    fam_split.update({k: pre_split[k] for k in keys if k in pre_split})
    print("사전학습 split을 따른 폰트:", len(keys) - len(unmatched), "/ 새로 나눈 폰트:", len(unmatched))
    fam_cat = {k: "handwriting" for k in keys}
    for f in faces:
        f["split"] = fam_split[f["family_key"]]
    drng = random.Random(SEED + 2)
    tr = [f for f in faces if f["split"] == "train"]
    dev_faces = sorted(drng.sample(tr, 10), key=lambda f: f["font_id"])
    v1s = json.load(open(os.path.join(SCR, "build/font_ko_pretrain/meta/splits.json")))
    assert v1s["heldout_char_ids"] == held_ids
    dev_chars = v1s["dev_char_ids"]

PLAN = {
    "train": ([f for f in faces if f["split"] == "train"], seen_ids + EXTRA_IDS),
    "train_unseen_chars": ([f for f in faces if f["split"] == "train"], held_ids),
    "val": ([f for f in faces if f["split"] == "val"], list(range(NCH))),
    "test": ([f for f in faces if f["split"] == "test"], list(range(NCH))),
    "dev": (dev_faces, dev_chars),
}

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
cache = {}


def face(f):
    if f["font_id"] not in cache:
        if len(cache) > 6:
            cache.clear()
        cache[f["font_id"]] = load_face(f)
    return cache[f["font_id"]]


counts = {}
for name, (fs, cids) in PLAN.items():
    d = os.path.join(OUT, name); os.makedirs(d)
    ca = np.array(cids)
    per = [ca[face(f)[1][ca]] for f in fs]
    n = sum(len(p) for p in per)
    X = np.lib.format.open_memmap(os.path.join(d, "images.npy"), mode="w+", dtype=np.uint8, shape=(n, S, S))
    L = np.zeros((n, 2), np.int32); o = 0
    for f, ids in zip(fs, per):
        X[o:o + len(ids)] = face(f)[0][ids]
        L[o:o + len(ids), 0] = f["font_id"]; L[o:o + len(ids), 1] = ids
        o += len(ids)
    X.flush(); del X
    np.save(os.path.join(d, "labels.npy"), L)
    counts[name] = dict(fonts=len(fs), families=len({f["family_key"] for f in fs}), images=n,
                        hangul_images=int((L[:, 1] < NH).sum()), extra_images=int((L[:, 1] >= NH).sum()))

md = os.path.join(OUT, "meta"); os.makedirs(md)
with open(os.path.join(md, "fonts.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    if MODE == "pre":
        w.writerow(["font_id", "family", "style", "family_key", "category", "split", "in_dev", "nanum_handwriting",
                    "in_finetune_set", "license", "source", "source_url", "package_version", "file", "sha256", "size_px",
                    "ink_mean", "extra_chars_available"])
        for f in faces:
            av = int(face(f)[1][NH:].sum())
            w.writerow([f["font_id"], f["family"], f["style"], f["family_key"], f["category"], f["split"], int(f["in_dev"]),
                        f["nanum_handwriting"], f["in_finetune_set"], f["lic_final"], f["src"], f["url"], f.get("version") or "",
                        os.path.basename(f["file"]) + (f"#{f['instance']}" if f.get("instance") else ""), f["sha256"],
                        f["size_px"], f["ink"], av])
    else:
        w.writerow(["font_id", "family", "family_en", "family_key", "category", "split", "in_dev", "license", "license_basis",
                    "file", "sha256", "size_px", "ink_mean", "extra_chars_available"])
        dset = {f["font_id"] for f in dev_faces}
        for f in faces:
            w.writerow([f["font_id"], f["family"], f["family_en"], f["family_key"], "handwriting", f["split"],
                        int(f["font_id"] in dset), "OFL-1.1",
                        "네이버 나눔손글씨 글꼴 (NAVER, SIL OFL 1.1로 배포; @kfonts 패키지 문서와 동일)",
                        os.path.relpath(f["path"], SCR + "/ft_src"), f["sha256"], f["size_px"], f["ink"], f["extra_avail"]])

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", *"ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"]
with open(os.path.join(md, "chars.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["char_id", "char", "codepoint", "kind", "cho", "jung", "jong", "cho_char", "jung_char", "jong_char",
                "extra_id", "heldout", "in_dev"])
    dset = set(dev_chars)
    for j, ch in enumerate(F.ALL_CHARS):
        if j < NH:
            s = ord(ch) - 0xAC00; c, v, t = s // 588, (s % 588) // 28, s % 28
            w.writerow([j, ch, f"U+{ord(ch):04X}", "hangul", c, v, t, CHO[c], JUNG[v], JONG[t], -1, int(j in held), int(j in dset)])
        else:
            w.writerow([j, ch, f"U+{ord(ch):04X}", F.EXTRA_KIND[j - NH], -1, -1, -1, "", "", "", j - NH, 0, 0])
open(os.path.join(md, "chars.txt"), "w", encoding="utf-8").write("".join(F.ALL_CHARS))
json.dump(dict(seed=SEED, family_split=dict(sorted(fam_split.items())), family_category=dict(sorted(fam_cat.items())),
               heldout_char_ids=held_ids, dev_font_ids=[f["font_id"] for f in dev_faces], dev_char_ids=dev_chars,
               extra_char_ids=EXTRA_IDS, counts=counts),
          open(os.path.join(md, "splits.json"), "w"), ensure_ascii=False, indent=1)

pv = os.path.join(OUT, "provenance"); os.makedirs(pv)
if MODE == "pre":
    src_pv = os.path.join(SCR, "build/font_ko_pretrain/provenance")
    shutil.copytree(os.path.join(src_pv, "licenses"), os.path.join(pv, "licenses"))
    shutil.copy(os.path.join(src_pv, "sources.csv"), pv); shutil.copy(os.path.join(src_pv, "excluded.csv"), pv)
else:
    os.makedirs(os.path.join(pv, "licenses"))
    # 네이버 나눔손글씨(펜)의 OFL 원문 (google/fonts ofl/nanumpenscript/OFL.txt, 저작권자 NAVER)
    ofl = os.path.join(SCR, "nanumpenscript_OFL.txt")
    if not os.path.exists(ofl):
        import subprocess
        subprocess.run(["curl", "-fsSL", "-o", ofl,
                        "https://raw.githubusercontent.com/google/fonts/main/ofl/nanumpenscript/OFL.txt"], check=True)
    shutil.copy(ofl, os.path.join(pv, "licenses", "OFL-1.1_NAVER.txt"))
    with open(os.path.join(pv, "sources.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["font_id", "family", "file", "sha256", "source", "license"])
        for f in faces:
            w.writerow([f["font_id"], f["family"], os.path.relpath(f["path"], SCR + "/ft_src"), f["sha256"],
                        "팀이 네이버에서 받은 나눔손글씨 109종 (fine_tune_handwritw.zip)", "OFL-1.1"])

pr = os.path.join(OUT, "previews"); os.makedirs(pr)
TEXT = "안녕하세요, 반가워요! (Hi 2026) “좋아” ♡ a.b-c_g:j?"
for name in ("train", "val", "test"):
    fs = [f for f in faces if f["split"] == name]
    pick = random.Random(SEED + 3).sample(fs, min(24, len(fs)))
    lines = []
    for f in pick:
        X, av = face(f); tiles = []
        for ch in TEXT:
            if ch == " ":
                tiles.append(np.zeros((S, 18), np.uint8)); continue
            j = F.ALL_CHARS.index(ch)
            x = X[j] if av[j] else np.zeros((S, S), np.uint8)
            cols = np.where(x.max(0) > 30)[0]
            x = x[:, cols.min():cols.max() + 1] if len(cols) else x[:, :12]
            tiles.append(np.pad(x, ((0, 0), (3, 3))))
        lines.append(np.concatenate(tiles, 1))
    Wd = max(l.shape[1] for l in lines)
    img = np.concatenate([np.pad(l, ((3, 3), (0, Wd - l.shape[1]))) for l in lines], 0)
    Image.fromarray(255 - img).save(os.path.join(pr, f"{name}_sentence.png"))
    grid = []
    for f in pick[:8]:
        X, av = face(f)
        grid.append(np.concatenate([X[NH + k] if av[NH + k] else np.zeros((S, S), np.uint8) for k in range(NE)], 1))
    g = np.concatenate(grid, 0)
    Image.fromarray(255 - g).save(os.path.join(pr, f"{name}_extra_chars.png"))

stats = dict(mode=MODE, n_fonts=len(faces), n_families=len(fam_split), counts=counts,
             split_families=dict(Counter(fam_split.values())),
             bytes=sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(OUT) for fn in fns))
if MODE == "pre":
    stats["nanum_handwriting_fonts"] = sum(f["nanum_handwriting"] for f in faces)
    stats["in_finetune_set_fonts"] = sum(f["in_finetune_set"] for f in faces)
    stats["extra_avail_hist"] = dict(Counter(int(face(f)[1][NH:].sum()) // 10 * 10 for f in faces))
json.dump(stats, open(os.path.join(OUT, "..", f"stats_{MODE}.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
