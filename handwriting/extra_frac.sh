#!/usr/bin/env bash
# 추가 실험: 손글씨 데이터 양 × 사전학습 (RF 교사)
#
#   SEED=0 bash extra_frac.sh scratch      # 사전학습 없음 (아무나)
#   SEED=0 bash extra_frac.sh pre          # 사전학습 있음 (runs/pre_rf_seed0.pt 가 있는 사람)
#   이번 비교는 seed 0 하나. 같은 SEED 면 두 사람이 정확히 같은 10% 글자로 학습합니다 (로그의 '지문'이 같아야 함).
#
# 나눔손글씨를 필체마다 글자 FRAC 비율만 남기고, 100% 실험(B)과 같은 학습 스텝(STEPS)만큼 학습한 뒤
# 1·64스텝(w=1.5)으로 평가해서 결과를 tar 로 묶습니다. 설정은 B 와 같습니다 (bs 128, lr 1e-4, base 96, seed 0).
#
# 환경변수 (기본값)
#   FRAC=0.1   SEED=0   STEPS=45660 (= 100% 데이터 30에폭, bs 128)
#   DATA=../font_ko_v2   CLF=runs/clf (없으면 classifier.py 로 학습)   PRE=runs/pre_rf_seed0.pt
set -eo pipefail
MODE=${1:?"사용법: bash extra_frac.sh scratch | pre"}
FRAC=${FRAC:-0.1}; SEED=${SEED:-0}; STEPS=${STEPS:-45660}
DATA=${DATA:-../font_ko_v2}; CLF=${CLF:-runs/clf}; PRE=${PRE:-runs/pre_rf_seed0.pt}
export HF_HUB_DISABLE_XET=1
cd "$(dirname "$0")"; mkdir -p logs runs/eval
say() { echo "== $* ($(date '+%m-%d %H:%M'))"; }

# ---- 확인: 데이터, 분류기, (pre) 사전학습 체크포인트
[ -d "$DATA/font_ko_finetune_nanum" ] || { echo "데이터가 없습니다: $DATA (font_ko_v2.zip 을 저장소 루트에 풀기)"; exit 1; }
if [ ! -s "$CLF/jamo_clf.pt" ]; then say "자모 분류기 학습"; python classifier.py --task jamo --data "$DATA" --epochs 3 --out "$CLF"; fi
if [ ! -s "$CLF/style_clf.pt" ]; then say "필체 분류기 학습"; python classifier.py --task style --data "$DATA" --epochs 5 --out "$CLF"; fi
if [ "$MODE" = pre ]; then
  [ -s "$PRE" ] || { echo "사전학습 체크포인트가 없습니다: $PRE"; exit 1; }
  python - "$PRE" <<'EOF'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False); a = c.get("args", {})
info = dict(stage=c.get("stage"), pred=c["cfg"].get("pred"), base=c["cfg"].get("base"), seed=c.get("seed"),
            epochs=a.get("epochs"), bs=a.get("bs"), exclude_nanum=a.get("exclude_nanum"))
print("사전학습 체크포인트:", info)
bad = [k for k, v in [("stage", "pre"), ("pred", "rf"), ("base", 96)] if info[k] != v]
if bad: sys.exit(f"이 실험에 맞지 않는 체크포인트입니다 ({', '.join(bad)} 확인). RF 폰트 사전학습(stage=pre, pred=rf, base 96)이어야 합니다.")
EOF
elif [ "$MODE" != scratch ]; then
  echo "모드는 scratch 또는 pre"; exit 1
fi

# ---- 데이터 양이 줄어도 학습 스텝은 100% 실험과 같게: epochs = STEPS / (데이터 장수 // 128)
# 뽑힌 글자는 SEED 로 정해짐 → 같은 SEED 면 담당 A·B 가 정확히 같은 부분집합 (지문으로 확인)
read N FP <<<"$(python -c "
import hashlib, torch, data as D
ds = D.load('$DATA', 'ft', 'train', frac=$FRAC, in_memory=False, seed=$SEED)
print(len(ds), hashlib.md5(torch.stack([ds.style, ds.char_id]).numpy().tobytes()).hexdigest()[:12])")"
EPOCHS=$(python -c "print(round($STEPS / max(1, $N // 128), 3))")
say "모드 $MODE  FRAC $FRAC  SEED $SEED  데이터 ${N}장  지문 $FP  에폭 $EPOCHS (≈ $STEPS 스텝)"

COMMON="--stage ft --data $DATA --frac $FRAC --epochs $EPOCHS --bs 128 --lr 1e-4 --amp --seed $SEED"
if [ "$MODE" = scratch ]; then
  TAG=ft_rf_seed${SEED}_frac${FRAC}_scratch
  python train.py $COMMON --objective rf --base 96 2>&1 | tee logs/train_${TAG}.log
else
  TAG=ft_rf_seed${SEED}_frac${FRAC}
  python train.py $COMMON --init "$PRE" 2>&1 | tee logs/train_${TAG}.log
fi
[ -s "runs/$TAG.pt" ] || { echo "학습 결과 파일이 없습니다: runs/$TAG.pt"; exit 1; }

say "평가 $TAG"
python eval.py --ckpt runs/$TAG.pt --data $DATA --steps 1 64 --w 1.5 \
    --jamo_clf $CLF/jamo_clf.pt --style_clf $CLF/style_clf.pt --n 1000 --out runs/eval 2>&1 | tee logs/eval_${TAG}.log
python eval.py --ckpt runs/$TAG.pt --speed_only --device cuda --steps 1 64 --w 1.5 --out runs/eval

OUTTAR=extra_${TAG}.tar.gz
tar czf $OUTTAR runs/eval/*${TAG}_gpu* runs/eval/grid_${TAG}_* runs/${TAG}_train.json runs/grids/${TAG} \
    logs/train_${TAG}.log logs/eval_${TAG}.log
say "끝 → $OUTTAR (체크포인트: runs/$TAG.pt)"; ls -lh $OUTTAR
