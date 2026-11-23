"""README.md·로더를 넣고 zip으로 묶는다. 사용: python3 -I finalize.py <work_dir> <out_dir> <zip_path> <loader.py>"""
import json, os, shutil, sys, zipfile

WORK, OUT, ZIP, LOADER = sys.argv[1:5]
st = json.load(open(os.path.join(WORK, "assemble_stats.json")))
c = st["counts"]
shutil.copy(LOADER, os.path.join(OUT, "load_font_ko.py"))


def gb(n):
    return f"{n * 4096 / 1e9:.2f} GB"


def cat(d):
    return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))


rows = "\n".join(
    f"| `{k}/` | {c[k]['families']} | {c[k]['fonts']} | {c[k]['chars']:,} | {c[k]['images']:,} | {gb(c[k]['images'])} | {desc} |"
    for k, desc in [
        ("train", "학습용. 보지 못한 글자(holdout)는 빠져 있음"),
        ("val", "검증용. 처음 보는 패밀리 × 전체 글자"),
        ("test", "최종 평가용. 처음 보는 패밀리 × 전체 글자"),
        ("train_unseen_chars", "train 폰트 × holdout 글자 (본 스타일, 처음 보는 글자)"),
        ("dev", "디버깅·하이퍼파라미터 탐색용 소형 세트 (train의 부분집합)"),
    ])

readme = f"""# font_ko_pretrain — 한글 폰트 사전학습 데이터셋

손글씨 생성 모델의 1단계(폰트 사전학습)용 데이터입니다. 2단계는 손글씨 데이터 파인튜닝입니다.
생성일: 2026-11-03

## 한눈에 보기
| 항목 | 값 |
|---|---|
| 글자 | KS X 1001 완성형 한글 **2,350자** (`meta/chars.txt` 순서 = char_id) |
| 폰트 | **{st['n_fonts']}개 스타일 / {st['n_families']}개 패밀리**, 전부 OFL-1.1 |
| 분류 (패밀리 수) | {cat(st['cat_families'])} |
| 이미지 | 64×64 `uint8`, **흰 글자(255) / 검은 배경(0)** (MNIST와 같은 극성) |
| 라벨 | `labels.npy` (N, 2) `int32` = `[font_id, char_id]` |
| 출처 (스타일 수) | {cat(st['src'])} |

## 폴더별 데이터
| 폴더 | 패밀리 | 스타일 | 글자 | 이미지 | 크기 | 용도 |
|---|---|---|---|---|---|---|
{rows}

- **패밀리 단위 분할(80/10/10, seed 42).** 굵기만 다른 변형(예: Gaegu Light/Bold)은 항상 같은 split에 있어서 스타일이 새어 나가지 않습니다. 분류별로 층화해 split마다 비율이 비슷합니다(train/val/test 패밀리 수: {'; '.join(f"{s} {cat(v)}" for s, v in st['split_cat'].items())}).
- **글자 holdout 235자(10%, seed 43).** 이 글자들은 `train`에 없고, `train_unseen_chars`·`val`·`test`에만 있습니다. 목록은 `meta/chars.csv`의 `heldout` 열에 있습니다.
- 그래서 세 가지 일반화를 따로 잴 수 있습니다:
  - 새 스타일 + 본 글자: `val`/`test`에서 `heldout==0`
  - 본 스타일 + 새 글자: `train_unseen_chars`
  - 새 스타일 + 새 글자: `val`/`test`에서 `heldout==1`

## 불러오기
```python
from load_font_ko import load_split, FontKoDataset
X, y, fonts, chars = load_split("font_ko_pretrain", "train")   # X는 memmap: 전체를 RAM에 올리지 않음
ds = FontKoDataset("font_ko_pretrain", "train", categories={{"handwriting", "print"}})  # 값 [-1,1], (1,64,64)
```
- `X`는 폰트 → 글자 순서로 저장돼 있으니 **반드시 셔플**해서 쓰세요 (`DataLoader(shuffle=True)`).
- 전부 RAM에 올리지 말고 memmap 그대로 쓰는 걸 권장합니다(train만 {gb(c['train']['images'])}).

## 메타데이터
- `meta/fonts.csv`: font_id → family, style, category(`handwriting`/`print`/`pixel`), split, license, 출처, 렌더링 크기, 평균 잉크 비율
- `meta/chars.csv`: char_id → 글자, 유니코드, **초성·중성·종성 인덱스(cho 0–18, jung 0–20, jong 0–27, 0=받침 없음)**, heldout, in_dev
  - 자모 분해는 그대로 조건(condition) 입력으로 쓸 수 있습니다. 새 글자 일반화에는 글자 ID 임베딩보다 자모 조건이 유리합니다.
- `meta/splits.json`: 패밀리별 split·분류, holdout 글자, dev 구성, 개수
- `previews/`: split별 샘플 이미지(보기 편하게 검은 글자/흰 배경으로 반전)

## 렌더링 방식
- 폰트마다 크기 배율 하나: 샘플 글자 잉크 크기의 98퍼센타일 = 54/64px. 폰트 안의 글자 간 상대 크기는 유지되고, 폰트 간 전체 크기만 맞춥니다.
- 글자마다 잉크 영역 중심을 캔버스 중심에 맞춥니다(넘치는 글자만 개별 축소).
- 3배 해상도로 그린 뒤 BOX 필터로 64px로 축소. 가변 폰트는 named instance(굵기)별로 따로 렌더링했습니다.
- 품질 필터: 빈 글자가 있거나 똑같은 이미지(두부·대체 글리프 의심)가 12개를 넘으면 제외했습니다.
- **증강은 넣지 않았습니다.** 손글씨와의 차이를 줄이려면 학습 중에 약한 회전·탄성 왜곡·굵기 변화·노이즈를 적용하세요.

## 학습 팁
- 처음엔 `dev/`로 파이프라인과 하이퍼파라미터를 몇 분 안에 확인한 뒤 `train/`으로 넘어가세요.
- 손글씨 파인튜닝이 목표라면 `category`로 가중치를 줄 수 있습니다(예: handwriting 비율을 높게 샘플링). `pixel`(픽셀·비트맵 폰트)은 64px에서 계단 모양이라 빼는 것도 고려하세요.
- 같은 split·seed를 유지하면 (a) 손글씨만 / (b) 폰트 사전학습→파인튜닝 / (c) 혼합 학습 비교를 같은 조건에서 재현할 수 있습니다.

## 라이선스와 출처
- 포함 기준: 한글 2,350자 전부 지원 + 라이선스 **OFL/Apache/CC0 확인**. 판정 근거(폰트 내장 정보, google/fonts의 OFL.txt, `@kfonts` 패키지 README의 폰트별 라이선스 표기)는 `provenance/sources.csv`의 `license_basis` 열에 있습니다.
- npm 패키지의 `license` 필드는 패키징 라이선스라서 판정에 쓰지 않았습니다. 자체 라이선스 폰트(넥슨·KoPub·제주 글꼴 등)와 폰트 파일·문서로 라이선스를 확인할 수 없는 배포본(일부 눈누 패키지)은 제외했고, 이유와 함께 `provenance/excluded.csv`에 남겼습니다.
- `provenance/licenses/`: 라이선스 원문. 원본 폰트 파일은 넣지 않았습니다(`sources.csv`의 URL과 SHA-256으로 다시 받을 수 있음).
- OFL 폰트로 만든 이미지이므로 학습·연구용으로 쓸 수 있습니다. 데이터셋 자체를 공개 배포할 때는 이 폴더의 `provenance/`를 함께 두세요.

## 한계
- 같은 회사·디자이너의 비슷한 스타일(예: 나눔손글씨 109종, 학교안심 시리즈)이 많아 실제 스타일 다양성은 패밀리 수보다 작을 수 있습니다.
- `category`는 출처 정보와 이름으로 자동 분류한 것이라 일부는 실제 필기보다 활자에 가깝거나 그 반대일 수 있습니다.
- 64×64 고정, KS X 1001 2,350자만 포함합니다.
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)

top = os.path.basename(OUT.rstrip("/"))
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
    for dp, _, fns in os.walk(OUT):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            z.write(p, os.path.join(top, os.path.relpath(p, OUT)))
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    n = len(z.namelist())
print("zip", ZIP, f"{os.path.getsize(ZIP)/1e6:.1f} MB", "files", n, "bad", bad)
