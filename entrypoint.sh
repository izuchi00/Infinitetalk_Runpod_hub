#!/bin/bash
set -Eeuo pipefail

COMFY_DIR="${COMFY_DIR:-/ComfyUI}"
RUNPOD_HF_CACHE_ROOT="${RUNPOD_HF_CACHE_ROOT:-/runpod-volume/huggingface-cache/hub}"
LOCAL_HF_HOME="${LOCAL_HF_HOME:-/tmp/huggingface}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
COMFY_READY_TIMEOUT="${COMFY_READY_TIMEOUT:-180}"

log() {
  printf '[entrypoint] %s\n' "$*"
}

fatal() {
  printf '[entrypoint] ERROR: %s\n' "$*" >&2
  exit 1
}

require_dir() {
  [[ -d "$1" ]] || fatal "Required directory not found: $1"
}

require_file() {
  [[ -s "$1" ]] || fatal "Required file not found or empty: $1"
}

resolve_bundle_snapshot() {
  local model_name="${MODEL_NAME:-}"
  local model_root refs_main snapshot_hash candidate marker

  if [[ -n "$model_name" && "$model_name" == */* ]]; then
    model_root="$RUNPOD_HF_CACHE_ROOT/models--${model_name//\//--}"
    refs_main="$model_root/refs/main"

    if [[ -f "$refs_main" ]]; then
      snapshot_hash="$(tr -d '[:space:]' < "$refs_main")"
      candidate="$model_root/snapshots/$snapshot_hash"
      if [[ -d "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi

    if [[ -d "$model_root/snapshots" ]]; then
      candidate="$(find -L "$model_root/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null || true)"
      if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  fi

  marker="$(find -L "$RUNPOD_HF_CACHE_ROOT" -type f \
    -name 'Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors' \
    -print -quit 2>/dev/null || true)"

  [[ -n "$marker" ]] || return 1

  python - "$marker" <<'PY'
import pathlib
import sys

p = pathlib.Path(sys.argv[1]).resolve()
parts = p.parts
try:
    idx = parts.index("snapshots")
except ValueError:
    raise SystemExit(1)

if idx + 1 >= len(parts):
    raise SystemExit(1)

print(pathlib.Path(*parts[: idx + 2]))
PY
}

find_bundle_file() {
  local filename="$1"
  find -L "$BUNDLE_SNAPSHOT" -type f -name "$filename" -print -quit 2>/dev/null || true
}

link_model() {
  local filename="$1"
  local destination_dir="$2"
  local source resolved destination

  source="$(find_bundle_file "$filename")"
  [[ -n "$source" ]] || fatal "Model is missing from cached bundle: $filename"

  resolved="$(readlink -f "$source")"
  require_file "$resolved"

  mkdir -p "$destination_dir"
  destination="$destination_dir/$filename"
  ln -sfn "$resolved" "$destination"
  require_file "$destination"

  log "Linked $filename -> $destination_dir"
}

find_wav2vec_directory() {
  local candidate

  candidate="$(find -L "$BUNDLE_SNAPSHOT" -type d -name 'chinese-wav2vec2-base' -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" && -f "$candidate/config.json" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  candidate="$(find -L "$BUNDLE_SNAPSHOT" -type f -path '*chinese-wav2vec2-base*/config.json' -print -quit 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    dirname "$candidate"
    return 0
  fi

  return 1
}

prepare_wav2vec_direct_model_path() {
  local source_dir="$1"
  local target_dir="$COMFY_DIR/models/transformers/TencentGameMate/chinese-wav2vec2-base"

  [[ -f "$source_dir/config.json" ]] || fatal "Bundled Wav2Vec directory has no config.json: $source_dir"

  mkdir -p "$(dirname "$target_dir")"
  rm -rf "$target_dir"
  ln -s "$(readlink -f "$source_dir")" "$target_dir"

  [[ -f "$target_dir/config.json" ]] || fatal "Failed to link bundled Wav2Vec model into ComfyUI transformers path."
  log "Linked TencentGameMate/chinese-wav2vec2-base -> $target_dir"
}

prepare_wav2vec_hf_cache() {
  local source_dir="$1"
  local repo_root="$LOCAL_HF_HOME/hub/models--TencentGameMate--chinese-wav2vec2-base"
  local snapshot_name="bundled"

  [[ -f "$source_dir/config.json" ]] || fatal "Bundled Wav2Vec directory has no config.json: $source_dir"

  mkdir -p "$repo_root/refs" "$repo_root/snapshots"
  rm -rf "$repo_root/snapshots/$snapshot_name"
  ln -s "$(readlink -f "$source_dir")" "$repo_root/snapshots/$snapshot_name"
  printf '%s' "$snapshot_name" > "$repo_root/refs/main"

  export HF_HOME="$LOCAL_HF_HOME"
  export HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
  export HUGGINGFACE_HUB_CACHE="$LOCAL_HF_HOME/hub"
  export TRANSFORMERS_CACHE="$LOCAL_HF_HOME/hub"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1

  log "Prepared offline Hugging Face cache for TencentGameMate/chinese-wav2vec2-base"
}

require_dir "$COMFY_DIR"
require_dir "$RUNPOD_HF_CACHE_ROOT"

BUNDLE_SNAPSHOT="$(resolve_bundle_snapshot || true)"
[[ -n "$BUNDLE_SNAPSHOT" && -d "$BUNDLE_SNAPSHOT" ]] || \
  fatal "Could not locate the RunPod cached InfiniteTalk bundle under $RUNPOD_HF_CACHE_ROOT. Check the endpoint Model field and MODEL_NAME."

log "Using cached model snapshot: $BUNDLE_SNAPSHOT"

link_model 'Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors' "$COMFY_DIR/models/diffusion_models"
link_model 'Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors' "$COMFY_DIR/models/diffusion_models"
link_model 'Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors' "$COMFY_DIR/models/diffusion_models"
link_model 'MelBandRoformer_fp16.safetensors' "$COMFY_DIR/models/diffusion_models"
link_model 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors' "$COMFY_DIR/models/loras"
link_model 'Wan2_1_VAE_bf16.safetensors' "$COMFY_DIR/models/vae"
link_model 'umt5-xxl-enc-fp8_e4m3fn.safetensors' "$COMFY_DIR/models/text_encoders"
link_model 'clip_vision_h.safetensors' "$COMFY_DIR/models/clip_vision"

WAV2VEC_DIR="$(find_wav2vec_directory || true)"
[[ -n "$WAV2VEC_DIR" && -d "$WAV2VEC_DIR" ]] || \
  fatal "TencentGameMate/chinese-wav2vec2-base was not found inside the cached bundle."
prepare_wav2vec_direct_model_path "$WAV2VEC_DIR"
prepare_wav2vec_hf_cache "$WAV2VEC_DIR"

log "All required InfiniteTalk assets are present."

COMFY_PID=""
cleanup() {
  if [[ -n "${COMFY_PID:-}" ]]; then
    kill "$COMFY_PID" 2>/dev/null || true
    wait "$COMFY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log "Starting ComfyUI..."
python "$COMFY_DIR/main.py" --listen 0.0.0.0 --port 8188 --use-sage-attention &
COMFY_PID=$!

log "Waiting for ComfyUI readiness (timeout: ${COMFY_READY_TIMEOUT}s)..."
for ((i=1; i<=COMFY_READY_TIMEOUT; i++)); do
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    wait "$COMFY_PID" || true
    fatal "ComfyUI exited before becoming ready."
  fi

  if curl -fsS "$COMFY_URL/" >/dev/null 2>&1; then
    log "ComfyUI is ready. Starting RunPod handler."
    python -u /handler.py
    exit $?
  fi

  sleep 1
done

fatal "ComfyUI failed to become ready within ${COMFY_READY_TIMEOUT} seconds."
