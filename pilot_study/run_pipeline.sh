#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-api}"
SIZE="${SIZE:-smoke}"
ROLLOUTS_PATH="${ROLLOUTS_PATH:-rollouts_data.pkl}"
CLUSTERED_PATH="${CLUSTERED_PATH:-clustered_data.pkl}"

LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
API_MODEL="${API_MODEL:-gemma-3-27b-it}"

LOCAL_EXTRA_ARGS="${LOCAL_EXTRA_ARGS:-}"
API_EXTRA_ARGS="${API_EXTRA_ARGS:-}"
CLUSTER_EXTRA_ARGS="${CLUSTER_EXTRA_ARGS:-}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

run_local() {
  if [[ "$SIZE" == "smoke" ]]; then
    uv run python 01_generate_rollouts.py \
      --smoke-test \
      --model-name "$LOCAL_MODEL" \
      ${LOCAL_EXTRA_ARGS}
  else
    uv run python 01_generate_rollouts.py \
      --model-name "$LOCAL_MODEL" \
      ${LOCAL_EXTRA_ARGS}
  fi
}

run_api() {
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "GEMINI_API_KEY is not set." >&2
    exit 1
  fi

  if [[ "$SIZE" == "smoke" ]]; then
    uv run python 01_generate_rollouts_api.py \
      --smoke-test \
      --model-name "$API_MODEL" \
      ${API_EXTRA_ARGS}
  else
    uv run python 01_generate_rollouts_api.py \
      --model-name "$API_MODEL" \
      ${API_EXTRA_ARGS}
  fi
}

main() {
  case "$MODE" in
    local)
      run_local
      ;;
    api)
      run_api
      ;;
    *)
      echo "Unknown MODE: $MODE (expected: local or api)" >&2
      exit 1
      ;;
  esac

  uv run python 02_clustering.py \
    --input-path "$ROLLOUTS_PATH" \
    --output-path "$CLUSTERED_PATH" \
    ${CLUSTER_EXTRA_ARGS}

  uv run python 03_metrics_eval.py \
    --input-path "$CLUSTERED_PATH" \
    ${EVAL_EXTRA_ARGS}
}

main "$@"
