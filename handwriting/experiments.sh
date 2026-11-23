#!/usr/bin/env bash
# 비교 실험 전체를 같은 설정으로, 시드별로 돌립니다.  handwriting/ 폴더에서:
#
#   bash experiments.sh                                   # 전부 (시드 0 1 2) — GPU 로 며칠 걸릴 수 있음
#   SEEDS=0 bash experiments.sh                           # 시드 하나만
#   STAGES="teachers" SEEDS="0 1 2" bash experiments.sh   # 단계 골라서 (clf teachers onestep eval speed)
#   RUN="torchrun --standalone --nproc_per_node=2" bash experiments.sh   # GPU 여러 장 (학습·증류·reflow)
#   SPLIT=dev PRE_EP=1 FT_EP=1 bash experiments.sh        # 빠른 점검 (결과는 보고서에 쓰지 않음)
#   SPEED_DEVS=cuda bash experiments.sh                   # 속도는 GPU 만 (CPU 속도는 노트북에서 따로)
#
# 이미 있는 체크포인트·결과는 건너뜁니다 (끊겨도 다시 실행하면 이어서 함).
#
# 비교군 (모두 같은 데이터·네트워크·교사 학습 설정, 아래 '맞춘 설정' 참고)
#   A  디퓨전 + DDIM          ft_v_seed{s}            스텝 1 4 10 50 64 × w 0 1.5 3   (기준선)
#   B  RF, reflow 없음        ft_rf_seed{s}           스텝 1 4 10 50 64 × w 0 1.5 3   (교사 없이 되는가)
#   C  RF → reflow (주 방법)  ft_rf_seed{s}_reflow[_distill]          교사 가중치로 시작
#   D  RF → reflow            ft_rf_seed{s}_reflow_scratch[_distill]  무작위 시작  ─┐ 교사 종류만 다름
#   E  디퓨전 → reflow         ft_v_seed{s}_reflow_scratch[_distill]   무작위 시작  ─┘
#   F  디퓨전 → PD            distill_seed{s}/student_{1,2,4}step
#
# 맞춘 설정
#   - 교사 샘플링 64스텝, guidance w=1.5 를 학생에 구움 (PD 첫 라운드, reflow 쌍 생성 모두)
#   - 학생 학습 30,000 반복: PD 5,000 × 6라운드 = reflow 20,000 + 1스텝 증류 10,000.  배치 128, EMA 0.999
#   - 교사 계산량(교사 실행 횟수)·시간은 체크포인트의 cost 에 기록 → eval 결과·대시보드 표에 나옴
#   - 평가: 같은 분류기, 같은 1,000개, eval 시드 0. 교사는 위 격자, 학생은 1·2·4스텝 (PD 는 1·2·4스텝 학생)
#   - 속도: 같은 문장, 같은 장치에서 GPU 와 CPU 따로
set -euo pipefail

DATA=${DATA:-../font_ko_v2}
SEEDS=${SEEDS:-"0 1 2"}
STAGES=${STAGES:-"clf teachers onestep eval speed"}
SPLIT=${SPLIT:-train}
PRE_EP=${PRE_EP:-10}
FT_EP=${FT_EP:-30}
BS=${BS:-256}
BASE=${BASE:-96}
W=${W:-1.5}
GEN=${GEN:-64}
PD_IT=${PD_IT:-5000}
RF_IT=${RF_IT:-20000}
RF_DIST=${RF_DIST:-10000}
NPAIRS=${NPAIRS:-0}
N_EVAL=${N_EVAL:-1000}
AMP=${AMP:---amp}
RUN=${RUN:-python}
OUT=${OUT:-runs}
JAMO_CLF=${JAMO_CLF:-$OUT/clf/jamo_clf.pt}
STYLE_CLF=${STYLE_CLF:-$OUT/clf/style_clf.pt}
TEACHER_STEPS=${TEACHER_STEPS:-"1 4 10 50 64"}
TEACHER_W=${TEACHER_W:-"0 1.5 3"}
STUDENT_STEPS=${STUDENT_STEPS:-"1 2 4"}

SUF=""; if [ "$SPLIT" != "train" ]; then SUF="_$SPLIT"; fi
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { echo; echo "==== $* ($(date '+%m-%d %H:%M'))"; }
mkdir -p logs "$OUT"

# ---------------------------------------------------------------- 평가용 분류기 (한 번)
if has clf; then
  [ -f "$JAMO_CLF" ]  || { say "자모 분류기"; python classifier.py --task jamo  --data "$DATA" --epochs 3 --out "$(dirname "$JAMO_CLF")" 2>&1 | tee logs/clf_jamo.log; }
  [ -f "$STYLE_CLF" ] || { say "필체 분류기"; python classifier.py --task style --data "$DATA" --epochs 5 --out "$(dirname "$STYLE_CLF")" 2>&1 | tee logs/clf_style.log; }
fi

for s in $SEEDS; do
  V="$OUT/ft_v_seed${s}${SUF}.pt"; RF="$OUT/ft_rf_seed${s}${SUF}.pt"

  # ---------------------------------------------------------------- 교사: 디퓨전 / RF (같은 명령, --objective 만 다름)
  if has teachers; then
    for obj in diffusion rf; do
      p=$([ $obj = rf ] && echo rf || echo v)
      PRE="$OUT/pre_${p}_seed${s}${SUF}.pt"; FT="$OUT/ft_${p}_seed${s}${SUF}.pt"
      [ -f "$PRE" ] || { say "사전학습 $obj seed $s"; $RUN train.py --stage pre --objective $obj --data "$DATA" --split $SPLIT \
          --epochs $PRE_EP --bs $BS --base $BASE --seed $s --out "$OUT" $AMP 2>&1 | tee logs/pre_${p}_seed${s}.log; }
      [ -f "$FT" ]  || { say "파인튜닝 $obj seed $s"; $RUN train.py --stage ft --init "$PRE" --data "$DATA" --split $SPLIT \
          --epochs $FT_EP --bs $BS --seed $s --out "$OUT" $AMP 2>&1 | tee logs/ft_${p}_seed${s}.log; }
    done
  fi

  # ---------------------------------------------------------------- 1스텝 만들기 (학생 학습 30,000 반복으로 통일)
  if has onestep; then
    COMMON=(--data "$DATA" --split $SPLIT --w $W --seed $s $AMP)
    RFCOMMON=(--gen_steps $GEN --iters $RF_IT --distill_iters $RF_DIST --n_pairs $NPAIRS --out "$OUT/reflow")
    [ -f "$OUT/distill_seed${s}/student_1step.pt" ] || { say "F 디퓨전 → PD seed $s"; $RUN distill.py --teacher "$V" --start $GEN \
        --iters $PD_IT --out "$OUT/distill_seed${s}" "${COMMON[@]}" 2>&1 | tee logs/pd_seed${s}.log; }
    B=$(basename "$RF" .pt); BV=$(basename "$V" .pt)
    [ -f "$OUT/reflow/${B}_reflow_distill.pt" ] || { say "C RF → reflow (교사 가중치 시작) seed $s"; $RUN reflow.py --teacher "$RF" \
        --init teacher "${RFCOMMON[@]}" "${COMMON[@]}" 2>&1 | tee logs/reflow_rf_seed${s}.log; }
    # D 는 C 와 같은 쌍을 재사용 (같은 교사·설정 → 같은 쌍 폴더)
    [ -f "$OUT/reflow/${B}_reflow_scratch_distill.pt" ] || { say "D RF → reflow (무작위 시작) seed $s"; $RUN reflow.py --teacher "$RF" \
        --init scratch "${RFCOMMON[@]}" "${COMMON[@]}" 2>&1 | tee logs/reflow_rf_scratch_seed${s}.log; }
    [ -f "$OUT/reflow/${BV}_reflow_scratch_distill.pt" ] || { say "E 디퓨전 → reflow (무작위 시작) seed $s"; $RUN reflow.py --teacher "$V" \
        --init scratch "${RFCOMMON[@]}" "${COMMON[@]}" 2>&1 | tee logs/reflow_v_scratch_seed${s}.log; }
  fi

  # ---------------------------------------------------------------- 평가 (같은 분류기·샘플·시드)
  EV=(--data "$DATA" --jamo_clf "$JAMO_CLF" --style_clf "$STYLE_CLF" --n $N_EVAL --out "$OUT/eval")
  TEACHERS=("$V" "$RF")
  STUDENTS=()
  for f in "$OUT/reflow/$(basename "$RF" .pt)_reflow"*.pt "$OUT/reflow/$(basename "$V" .pt)_reflow"*.pt \
           "$OUT/distill_seed${s}"/student_{1,2,4}step.pt; do if [ -f "$f" ]; then STUDENTS+=("$f"); fi; done
  if has eval; then
    for t in "${TEACHERS[@]}"; do
      if [ -f "$t" ]; then say "평가 교사 $(basename "$t")"
        python eval.py --ckpt "$t" --steps $TEACHER_STEPS --w $TEACHER_W "${EV[@]}" 2>&1 | tee -a logs/eval_seed${s}.log; fi
    done
    for t in "${STUDENTS[@]}"; do say "평가 학생 $(basename "$t")"; \
        python eval.py --ckpt "$t" --steps $STUDENT_STEPS "${EV[@]}" 2>&1 | tee -a logs/eval_seed${s}.log; done
  fi
  # ---------------------------------------------------------------- 속도 (GPU, CPU 따로)
  if has speed; then
    for dev in ${SPEED_DEVS:-cuda cpu}; do
      if [ $dev = cuda ] && ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then continue; fi
      for t in "${TEACHERS[@]}"; do
        if [ -f "$t" ]; then python eval.py --ckpt "$t" --speed_only --device $dev --steps $TEACHER_STEPS --w $W --out "$OUT/eval"; fi
      done
      for t in "${STUDENTS[@]}"; do python eval.py --ckpt "$t" --speed_only --device $dev --steps $STUDENT_STEPS --out "$OUT/eval"; done
    done 2>&1 | tee -a logs/speed_seed${s}.log
  fi
done

say "끝. 결과 보기: python dashboard.py --runs $OUT"
