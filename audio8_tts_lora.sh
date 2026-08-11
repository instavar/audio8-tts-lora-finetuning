#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-${PROJECT_ROOT}/model/audio8_tts_0_6B_preview}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
EVAL_JSONL="${EVAL_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/audio8_tts_lora}"
EXPORT_DIR="${EXPORT_DIR:-${OUTPUT_DIR}/merged}"

if [[ -z "${TRAIN_JSONL}" ]]; then
  echo "TRAIN_JSONL is required." >&2
  exit 2
fi

if [[ "${AUDIT_CORPUS:-0}" == "1" ]]; then
  : "${RAW_TRAIN_JSONL:?set RAW_TRAIN_JSONL when AUDIT_CORPUS=1}"
  : "${RAW_VALIDATION_JSONL:?set RAW_VALIDATION_JSONL when AUDIT_CORPUS=1}"
  : "${RAW_TEST_JSONL:?set RAW_TEST_JSONL when AUDIT_CORPUS=1}"
  : "${INSTAVAR_VOICE_EVAL_DIR:?set INSTAVAR_VOICE_EVAL_DIR to the pinned instavar-voice-evaluation checkout}"
  audit_args=(
    --split "train=${RAW_TRAIN_JSONL}"
    --split "validation=${RAW_VALIDATION_JSONL}"
    --split "test=${RAW_TEST_JSONL}"
  )
  if [[ -n "${CORPUS_GROUP_FIELD:-}" ]]; then
    audit_args+=(--group-field "${CORPUS_GROUP_FIELD}")
  fi
  "${PYTHON}" "${INSTAVAR_VOICE_EVAL_DIR}/main.py" audit-corpus "${audit_args[@]}"
fi

data_args=(--train_jsonl "${TRAIN_JSONL}")
training_args=(--do_train true)
if [[ -n "${EVAL_JSONL}" ]]; then
  data_args+=(--eval_jsonl "${EVAL_JSONL}")
  training_args+=(--do_eval true --eval_strategy steps --eval_steps "${EVAL_STEPS:-20}")
fi

"${PYTHON}" "${PROJECT_ROOT}/audio8_tts_sft.py" \
  --model_name_or_path "${MODEL}" \
  "${data_args[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --export_dir "${EXPORT_DIR}" \
  --overwrite_output_dir "${OVERWRITE_OUTPUT_DIR:-false}" \
  --use_lora true \
  --lora_r "${LORA_R:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --lora_dropout "${LORA_DROPOUT:-0}" \
  --lora_target_modules "${LORA_TARGET_MODULES:-wqkv,wo,w1,w2,w3}" \
  --max_length "${MAX_LENGTH:-512}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --max_steps "${MAX_STEPS:-100}" \
  --warmup_steps "${WARMUP_STEPS:-10}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --logging_first_step true \
  --bf16 "${BF16:-false}" \
  --fp16 "${FP16:-false}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-false}" \
  --report_to "${REPORT_TO:-none}" \
  --seed "${SEED:-42}" \
  --data_seed "${DATA_SEED:-42}" \
  "${training_args[@]}" \
  "$@"
