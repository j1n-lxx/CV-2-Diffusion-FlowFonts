"""한글 폰트 수집기: npm 패키지 / Google Fonts raw 파일을 받아 폰트별로 검사한다.

사용: python3 -I harvest.py <work_dir> <jobs.json> [--budget 540]
jobs.json: [{"id": "...", "source": "npm"|"gf", "pkg": "...", "tar": url} 또는 {"files": [url,...]}]
결과: <work_dir>/results/<safe_id>.json (패키지별), 통과 폰트 파일은 <work_dir>/keep/<safe_id>/ 에 남긴다.
"""
import hashlib, json, os, re, shutil, subprocess, sys, tarfile, time
from fontTools.ttLib import TTFont, TTCollection

WORK, JOBS = sys.argv[1], sys.argv[2]
BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 540
RES = os.path.join(WORK, "results"); KEEP = os.path.join(WORK, "keep"); DL = os.path.join(WORK, "dl")
for d in (RES, KEEP, DL):
    os.makedirs(d, exist_ok=True)


def ks():
    out = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            try:
                out.append(ord(bytes([hi, lo]).decode("euc-kr")))
            except UnicodeDecodeError:
                pass
    return out


KS = ks(); ALL = range(0xAC00, 0xD7A4)
ALNUM = list(range(48, 58)) + list(range(65, 91)) + list(range(97, 123))
FONT_EXT = {".ttf": 0, ".otf": 1, ".woff2": 2, ".woff": 3, ".ttc": 4, ".otc": 4}
LIC_NAMES = re.compile(r"(licen[cs]e|ofl|copying|readme)", re.I)


def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def nm(f, ids):
    t = f["name"]
    for i in ids:
        for args in ((i, 3, 1, 0x409), (i, 1, 0, 0)):
            r = t.getName(*args)
            if r and str(r).strip():
                return str(r).strip()
        for rec in t.names:
            if rec.nameID == i:
                try:
                    s = rec.toUnicode().strip()
                    if s:
                        return s
                except Exception:
                    pass
    return ""


def classify(text):
    t = text.lower()
    if re.search(r"open font license|openfontlicense|scripts\.sil\.org|\bofl\b|sil open", t):
        return "OFL"
    if "apache license" in t or "apache-2" in t or "apache 2" in t:
        return "Apache-2.0"
    if "creative commons zero" in t or "cc0" in t:
        return "CC0"
    return "other"


def inspect(font, path, idx):
    cps = set((font.getBestCmap() or {}).keys())
    lic_desc, lic_url = nm(font, (13,)), nm(font, (14,))
    return dict(
        family=nm(font, (16, 1)), style=nm(font, (17, 2)), full=nm(font, (4,)), ps=nm(font, (6,)),
        copyright=nm(font, (0,))[:300], lic_desc=lic_desc[:600], lic_url=lic_url,
        ks2350=sum(c in cps for c in KS), all11172=sum(c in cps for c in ALL),
        alnum62=sum(c in cps for c in ALNUM), variable="fvar" in font, glyphs=font["maxp"].numGlyphs,
        face=idx,
    )


def fetch(url, dest):
    r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-m", "300", "-o", dest, url], capture_output=True)
    return r.returncode == 0


def process(job):
    jid = safe(job["id"]); jdir = os.path.join(DL, jid)
    shutil.rmtree(jdir, ignore_errors=True); os.makedirs(jdir)
    files, lic_texts = [], []
    if job["source"] == "npm":
        tgz = os.path.join(jdir, "pkg.tgz")
        if not fetch(job["tar"], tgz):
            return dict(id=job["id"], error="download failed")
        ext = os.path.join(jdir, "x"); os.makedirs(ext)
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                n = os.path.normpath(m.name)
                if n.startswith("..") or os.path.isabs(n):
                    continue
                base = os.path.basename(n); e = os.path.splitext(base)[1].lower()
                if e in FONT_EXT or (LIC_NAMES.search(base) and m.size < 400_000):
                    m.name = n
                    tf.extract(m, ext, filter="data") if hasattr(tarfile, "data_filter") else tf.extract(m, ext)
        for dp, _, fns in os.walk(ext):
            for fn in fns:
                p = os.path.join(dp, fn); e = os.path.splitext(fn)[1].lower()
                if e in FONT_EXT:
                    files.append(p)
                elif LIC_NAMES.search(fn):
                    try:
                        lic_texts.append((os.path.relpath(p, ext), open(p, encoding="utf-8", errors="replace").read()))
                    except Exception:
                        pass
        os.remove(tgz)
    else:  # gf: 개별 파일 URL + 라이선스 URL
        for u in job["files"]:
            p = os.path.join(jdir, os.path.basename(u).replace("%5B", "[").replace("%5D", "]"))
            if fetch(u, p):
                files.append(p)
        if job.get("license_url"):
            lp = os.path.join(jdir, "OFL.txt")
            if fetch(job["license_url"], lp):
                lic_texts.append(("OFL.txt", open(lp, encoding="utf-8", errors="replace").read()))
    pkg_lic = " ".join(t[:4000] for _, t in lic_texts)
    faces = []
    for p in files:
        try:
            e = os.path.splitext(p)[1].lower()
            fl = TTCollection(p, lazy=True).fonts if e in (".ttc", ".otc") else [TTFont(p, lazy=True)]
            for i, f in enumerate(fl):
                info = inspect(f, p, i)
                info["file"] = os.path.relpath(p, jdir); info["ext"] = e
                info["bytes"] = os.path.getsize(p)
                info["sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
                info["license"] = classify(info["lic_desc"] + " " + info["lic_url"])
                if info["license"] == "other":
                    info["license_pkgfile"] = classify(pkg_lic)
                faces.append(info)
        except Exception as ex:
            faces.append(dict(file=os.path.relpath(p, jdir), error=repr(ex)[:200]))
    # 한글 2,350자 지원 face만 남기고, 같은 (family, style)은 확장자 우선순위로 하나만
    good = [f for f in faces if f.get("ks2350") == 2350]
    good.sort(key=lambda f: (f["variable"], FONT_EXT.get(f["ext"], 9), len(f["file"])))
    seen, kept = set(), []
    for f in good:
        k = (f["family"].lower(), f["style"].lower(), f["variable"])
        if k in seen:
            continue
        seen.add(k); kept.append(f)
    kdir = os.path.join(KEEP, jid)
    if kept:
        os.makedirs(kdir, exist_ok=True)
        for f in kept:
            dst = os.path.join(kdir, os.path.basename(f["file"]))
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(jdir, f["file"]), dst)
            f["kept_as"] = os.path.relpath(dst, KEEP)
        for name, txt in lic_texts:
            open(os.path.join(kdir, "LICENSE__" + safe(name)), "w", encoding="utf-8").write(txt)
    shutil.rmtree(jdir, ignore_errors=True)
    return dict(id=job["id"], source=job["source"], pkg=job.get("pkg"), version=job.get("version"),
                url=job.get("tar") or job.get("page"), n_font_files=len(files),
                lic_files=[n for n, _ in lic_texts], faces=faces, kept=[f["kept_as"] for f in kept])


def main():
    jobs = json.load(open(JOBS))
    t0 = time.time(); done = skipped = 0
    for job in jobs:
        out = os.path.join(RES, safe(job["id"]) + ".json")
        if os.path.exists(out):
            continue
        if time.time() - t0 > BUDGET:
            skipped += 1
            continue
        try:
            r = process(job)
        except Exception as ex:
            r = dict(id=job["id"], error=repr(ex)[:300])
        json.dump(r, open(out, "w"), ensure_ascii=False)
        done += 1
    left = sum(not os.path.exists(os.path.join(RES, safe(j["id"]) + ".json")) for j in jobs)
    print(f"processed {done}, remaining {left}, elapsed {time.time()-t0:.0f}s")


main()
