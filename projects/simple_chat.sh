#!/usr/bin/env bash
# =========================
# Usage Examples:
# =========================
# ./llama.sh
#   - Runs default model (mistral7binstrq4) on GPU 0 with default prompt.
#
# ./llama.sh prompt="Explain transformers simply"
#   - Uses default model and GPU, overrides the prompt.
#
# ./llama.sh model=llama3instr sys_promp="Context: " prompt="Summarize AI news"
#   - Uses a different model, adds a prefix to the prompt, keeps default GPU.
#
# ./llama.sh 1 model=mistral7binstrq5 prompt="Write a poem about space"
#   - Uses GPU 1, selects specific model, and sets the prompt.
#
# ./llama.sh > output.txt
#   - Saves the chat output only (no script logs) to a file.
#
# Notes:
#   - Available model keys: llama2, llama3instr, mistral7binstrq4, mistral7binstrq5, mistral7q4
#   - Default GPU is 0, default context size is 4096 tokens
#   - The script only prints chat output; debug messages are suppressed.


# =========================
# Environment
# =========================
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1

# =========================
# Defaults
# =========================
MODELS_DIR="$HOME/models"
MODEL_KEY="mistral7binstrq4"
CONTEXT=4096
NGL=99
PROMPT="No prompt provided."
GPU="0"

# =========================
# Model mapping
# Format: key:filename
# =========================
MODEL_MAP=$(cat <<EOF
llama2:llama-2-7b.Q4_0.gguf
llama3instr:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
mistral7binstrq4:mistral-7b-instruct-v0.1.Q4_K_M.gguf
mistral7binstrq5:mistral-7b-instruct-v0.1.Q5_K_M.gguf
mistral7q4:mistral-7b-v0.1.Q4_K_M.gguf
EOF
)

# =========================
# Parse arguments
# =========================
for arg in "$@"; do
    case "$arg" in
        model=*) MODEL_KEY="${arg#*=}" ;;
        sys_prompt=*) SYS_PROMPT="${arg#*=}" ;;
        prompt=*) PROMPT="${arg#*=}" ;;
        *)
            if [[ "$arg" =~ ^[0-9]+$ ]]; then
                GPU="$arg"
            fi
            ;;
    esac
done

# =========================
# Resolve model from mapping
# =========================
MODEL_FILE=$(echo "$MODEL_MAP" | awk -F: -v k="$MODEL_KEY" '$1 == k {print $2}')

if [ -z "$MODEL_FILE" ]; then
    echo "❌ Unknown model key: $MODEL_KEY" >&2
    echo "Available models:" >&2
    echo "$MODEL_MAP" | awk -F: '{print "  " $1}' >&2
    exit 1
fi

MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model file not found: $MODEL_PATH" >&2
    exit 1
fi


# =========================
# Run llama-completion
# Only chat output is printed to stdout
# =========================
CMD=(
    ./build/bin/llama-completion
    --model "$MODEL_PATH"      # path to the .gguf model file
    --system-prompt "$SYS_PROMPT"
    --prompt "$PROMPT"         # initial prompt / input text
    --ctx-size "$CONTEXT"      # context window size (number of tokens model can remember)
    --gpu-layers "$NGL"        # number of layers offloaded to GPU
    --predict 1000             # max number of tokens to generate in this completion
    --escape                   # whether to process escapes sequences (\n, \r, \t, ', ", \) (default: true)
    --conversation             # whether to run in conversation mode
    --repeat-penalty 1.5       # penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
    --chat-template vicuna     # chat template
    --temp 0.5                 # sampling temperature
    --reasoning-format deepseek    # reasoning format (none, auto, deepseek)
)

if [ -n "$GPU" ]; then
    echo Using GPU.
    #clinfo -l
    # --main-gpu select GPU device index
    # --split-mode GPU split mode (none = single GPU)
    CMD+=(--main-gpu "$GPU" --split-mode none)
fi
echo Using model: $MODEL_FILE
# Run and redirect script messages to stderr if needed
"${CMD[@]}" 2>/dev/null

