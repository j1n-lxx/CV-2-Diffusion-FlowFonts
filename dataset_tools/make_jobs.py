"""폰트 수집 목록(jobs.json) 만들기. 2026-11-03 에 쓴 목록은 이 폴더의 jobs.json 에 그대로 있습니다.
npm 검색 결과는 날마다 바뀌므로, 같은 데이터셋을 다시 만들려면 이 스크립트 대신 jobs.json 을 쓰세요.

사용: python3 -I make_jobs.py <work>        → <work>/jobs.json, <work>/gfm, <work>/*_manifests.json

출처
  1. Google Fonts: npm 'google-font-metadata' 의 목록에서 subsets 에 korean 이 있는 패밀리 → google/fonts 저장소 파일
  2. npm @kfonts / @noonnu 묶음 전체 + OFL 로 알려진 한글 폰트 패키지 몇 개
라이선스 판정은 여기서 하지 않습니다 (harvest.py → select_fonts.py 가 폰트 파일·문서로 판정).
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tarfile

WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)
RAW = "https://raw.githubusercontent.com/google/fonts/main/"


def get(url):
    r = subprocess.run(["curl", "-fsSL", "-m", "60", url], capture_output=True)
    return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else None


def manifest(name):
    s = get("https://registry.npmjs.org/" + name.replace("/", "%2F") + "/latest")
    try:
        d = json.loads(s)
    except Exception:
        return name, None
    lic = d.get("license"); lic = lic.get("type") if isinstance(lic, dict) else lic
    return name, dict(v=d.get("version"), lic=lic, size=d["dist"].get("unpackedSize"), tar=d["dist"]["tarball"],
                      desc=d.get("description", ""))


# ---- 1. Google Fonts
gfm = os.path.join(WORK, "gfm"); os.makedirs(gfm, exist_ok=True)
_, m = manifest("google-font-metadata")
tgz = os.path.join(gfm, "gfm.tgz")
subprocess.run(["curl", "-fsSL", "-o", tgz, m["tar"]], check=True)
with tarfile.open(tgz) as tf:
    tf.extractall(gfm, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(gfm)
api = json.load(open(os.path.join(gfm, "package/data/api-response.json")))
jobs = []
for fam in [f["family"] for f in api if "korean" in f.get("subsets", [])]:
    d = re.sub(r"[^a-z0-9]", "", fam.lower())
    for lic in ("ofl", "apache", "ufl"):
        pb = get(f"{RAW}{lic}/{d}/METADATA.pb")
        if pb:
            files = sorted(set(re.findall(r'filename:\s*"([^"]+)"', pb)))
            jobs.append(dict(id="gf/" + d, source="gf", pkg=fam, page=f"https://github.com/google/fonts/tree/main/{lic}/{d}",
                             files=[RAW + f"{lic}/{d}/" + f.replace("[", "%5B").replace("]", "%5D") for f in files],
                             license_url=RAW + f"{lic}/{d}/" + ("OFL.txt" if lic == "ofl" else "LICENSE.txt")))
            break
print("Google Fonts 한글 패밀리:", len(jobs))

# ---- 2. npm
names = set()
for q in ["kfonts", "nanum%20handwritting"] + [f"noonnu&from={i}" for i in (0, 250, 500, 750)]:
    s = get(f"https://registry.npmjs.org/-/v1/search?text={q}&size=250")
    for o in (json.loads(s).get("objects", []) if s else []):
        n = o["package"]["name"]
        if n.startswith(("@kfonts/", "@noonnu/")):
            names.add(n)
other = ["pretendard", "@sun-typeface/suit", "wanted-sans", "typeface-spoqa-han-sans2", "spoqa-han-sans", "galmuri",
         "jennysans", "plemocjk-kr"]
with cf.ThreadPoolExecutor(16) as ex:
    man = {k: v for k, v in ex.map(manifest, sorted(names) + other) if v}
json.dump(man, open(os.path.join(WORK, "scope_manifests.json"), "w"), ensure_ascii=False, indent=0)
json.dump({}, open(os.path.join(WORK, "npm_manifests.json"), "w"))   # assemble.py 가 읽는 자리 (설명 문구용)
order = other + sorted(n for n in man if n.startswith("@kfonts/")) + sorted(n for n in man if n.startswith("@noonnu/"))
for n in order:
    if n not in man:
        continue
    if n.startswith("@kfonts/") and n.endswith("-otf") and n[:-4] in man:
        continue                                          # 같은 폰트의 OTF 판은 건너뜀
    jobs.append(dict(id="npm/" + n, source="npm", pkg=n, version=man[n]["v"], tar=man[n]["tar"]))
json.dump(jobs, open(os.path.join(WORK, "jobs.json"), "w"), indent=0)
print("jobs:", len(jobs), "->", os.path.join(WORK, "jobs.json"))
