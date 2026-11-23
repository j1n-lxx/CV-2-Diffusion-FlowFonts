"""수집 결과(results/*.json)를 모아 최종 폰트 목록(selected.json)과 제외 목록(excluded.csv)을 만든다.

규칙
- 한글 KS X 1001 2,350자 전부 지원하는 face만.
- 라이선스: 폰트 내장 정보(name ID 13/14) 또는 함께 배포된 LICENSE 파일이 OFL/Apache-2.0/CC0 인 것만 포함.
  (npm package.json의 license 필드는 패키징 라이선스라 판정에 쓰지 않음)
- 중복: (패밀리, 스타일) 정규화 키가 같으면 하나만. 출처 우선순위 Google Fonts > 기타 npm > @kfonts > @noonnu.
- 가변 폰트는 fvar의 named instance를 각각 하나의 스타일로 펼친다(렌더 시 set_variation_by_name).
"""
import csv, glob, json, os, re, sys
from fontTools.ttLib import TTFont

WORK = sys.argv[1]
KEEP = os.path.join(WORK, "keep")
OPEN = {"OFL", "Apache-2.0", "CC0"}


def rank(src_id):
    if src_id.startswith("gf/"):
        return 0
    if src_id.startswith("npm/@kfonts/"):
        return 2
    if src_id.startswith("npm/@noonnu/"):
        return 3
    return 1


def norm(s):
    s = s.lower()
    s = re.sub(r"\b(otf|ttf|std|pro)\b", "", s)
    return re.sub(r"[^0-9a-z가-힣]", "", s)


STYLE_ALIASES = {"normal": "regular", "book": "regular", "roman": "regular", "": "regular",
                 "ultralight": "extralight", "heavy": "black", "demibold": "semibold"}


def nstyle(s):
    k = norm(s)
    return STYLE_ALIASES.get(k, k)


rows = []
for p in sorted(glob.glob(os.path.join(WORK, "results", "*.json"))):
    r = json.load(open(p))
    for f in r.get("faces", []):
        if f.get("ks2350") != 2350 or "kept_as" not in f:
            continue
        if f["license"] != "other":
            lic, basis = f["license"], "font name table"
        else:
            lic = f.get("license_pkgfile", "other")
            lf = " ".join(r.get("lic_files", []))
            basis = ("OFL.txt in google/fonts" if r["id"].startswith("gf/") else
                     "package README (per-font license line)" if "README" in lf else "package LICENSE file")
        if not f["family"]:
            f["family"] = f["ps"] or f["full"] or os.path.splitext(os.path.basename(f["file"]))[0]
            f["family"] = re.sub(r"[-_ ]?(regular|normal)$", "", f["family"], flags=re.I) or f["family"]
        rows.append(dict(src=r["id"], pkg=r.get("pkg"), version=r.get("version"), url=r.get("url"), **f,
                         lic_final=lic, lic_basis=basis))

# 가변 폰트 → named instance 펼치기
expanded = []
for f in rows:
    if not f["variable"]:
        f["instance"] = None
        expanded.append(f)
        continue
    tt = TTFont(os.path.join(KEEP, f["kept_as"]), lazy=True)
    name = tt["name"]
    for inst in tt["fvar"].instances:
        sub = name.getDebugName(inst.subfamilyNameID) or ""
        g = dict(f); g["style"] = sub; g["instance"] = sub
        expanded.append(g)

excluded, chosen = [], {}
for f in sorted(expanded, key=lambda f: (rank(f["src"]), f["variable"], f["ext"] != ".ttf")):
    if f["lic_final"] not in OPEN:
        excluded.append((f, "license:" + f["lic_final"]))
        continue
    key = (norm(f["family"]), nstyle(f["style"]))
    if key in chosen:
        excluded.append((f, "duplicate of " + chosen[key]["src"]))
        continue
    chosen[key] = f

sel = sorted(chosen.values(), key=lambda f: (norm(f["family"]), nstyle(f["style"])))
json.dump(sel, open(os.path.join(WORK, "selected.json"), "w"), ensure_ascii=False, indent=0)
with open(os.path.join(WORK, "excluded.csv"), "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["source", "family", "style", "reason", "license_desc"])
    for f, why in excluded:
        w.writerow([f["src"], f["family"], f["style"], why, f["lic_desc"][:200]])
fams = {norm(f["family"]) for f in sel}
print(f"selected faces: {len(sel)}, families: {len(fams)}")
from collections import Counter
print("by source:", Counter(f["src"].split("/")[0] + ("/" + f["src"].split("/")[1] if f["src"].startswith("npm/@") else "") for f in sel))
print("by license:", Counter(f["lic_final"] for f in sel))
print("excluded reasons:", Counter(w.split(":")[0].split(" of ")[0] for _, w in excluded))
lic_other = Counter(f["family"] for f, w in excluded if w.startswith("license"))
print("license-excluded families (top 40):", lic_other.most_common(40))
