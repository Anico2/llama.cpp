#!/usr/bin/env bash
# =========================
# Usage Examples:
# =========================
# CLI usage (default mode 'completion'):
# ./llama.sh
#   - Runs the default model (mistral7binstrq4) on GPU 0 with the default prompt.
#
# ./llama.sh prompt="Explain transformers simply"
#   - Uses the default model and GPU, overriding the prompt.
#
# ./llama.sh model=llama3instr sys_prompt="Context: " prompt="Summarize AI news"
#   - Uses a different model, adds a system prompt prefix, keeps default GPU.
#
# ./llama.sh 1 model=mistral7binstrq5 prompt="Write a poem about space"
#   - Uses GPU 1, selects a specific model, and sets the prompt.
#
# ./llama.sh mode=cli model=llama2
#   - Runs in 'cli' mode with the specified model.
#
# ./llama.sh mode=server model=mistral7q4
#   - Runs in 'server' mode (no prompt needed).
#
# Redirect output to a file:
# ./llama.sh prompt="Summarize AI news" > output.txt
#   - Saves only the chat output (no script logs) to a file.
#
# =========================
# Library usage (sourced):
# =========================
# source ./llama.sh
# llama_run model=llama3instr prompt="Hello from library"
#   - Runs the model using the modular, callable API.
#
# Or call main() directly:
# main mode=completion model=mistral7binstrq5 prompt="Explain transformers"
#
# =========================
# Notes:
#   - Available model keys: llama2, llama3instr, mistral7binstrq4, mistral7binstrq5, mistral7q4
#   - Default GPU is 0, default context size is 4096 tokens
#   - Valid modes: completion, cli, server
#   - When sourced, the script exposes a callable function 'llama_run' or 'main'
#   - The script only prints chat output; debug messages are suppressed


# =========================
# Environment
# =========================
setup_env() {
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    export ZES_ENABLE_SYSMAN=1
}

# =========================
# Defaults
# =========================
init_defaults() {
    MODELS_DIR="$HOME/models"
    MODEL_KEY="mistral7binstrq4"
    CONTEXT=4096
    NGL=99
    PROMPT="No prompt provided."
    GPU="0"
    MODE="completion"
}

# =========================
# Allowed parameters
# =========================
init_allowed_keys() {
    ALLOWED_KEYS=(
        model
        sys_prompt
        prompt
        mode
    )
}

# =========================
# Load helpers
# =========================
load_helpers() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/arg_utils.sh"
}

# =========================
# Model mapping
# =========================
init_model_map() {
    MODEL_MAP=$(cat <<EOF
llama2:llama-2-7b.Q4_0.gguf
llama3instr:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
mistral7binstrq4:mistral-7b-instruct-v0.1.Q4_K_M.gguf
mistral7binstrq5:mistral-7b-instruct-v0.1.Q5_K_M.gguf
mistral7q4:mistral-7b-v0.1.Q4_K_M.gguf
EOF
)
}

# =========================
# Parse arguments (validated)
# =========================
parse_args() {
    for arg in "$@"; do
        if [[ "$arg" == *=* ]]; then
            key="${arg%%=*}"

            if [[ ! " ${ALLOWED_KEYS[*]} " =~ " $key " ]]; then
                echo "Unknown parameter: $key" >&2

                suggestion=$(suggest_key "$key")
                [[ -n "$suggestion" ]] && echo "Did you mean: $suggestion ?" >&2

                echo "Allowed parameters: ${ALLOWED_KEYS[*]}" >&2
                exit 1
            fi
        fi

        case "$arg" in
            model=*)      MODEL_KEY="${arg#*=}" ;;
            sys_prompt=*) SYS_PROMPT="${arg#*=}" ;;
            prompt=*)     PROMPT="${arg#*=}" ;;
            mode=*)       MODE="${arg#*=}" ;;
            *)
                [[ "$arg" =~ ^[0-9]+$ ]] && GPU="$arg"
                ;;
        esac
    done
}

# =========================
# Resolve model
# =========================
resolve_model() {
    MODEL_FILE=$(echo "$MODEL_MAP" | awk -F: -v k="$MODEL_KEY" '$1 == k {print $2}')

    if [[ -z "$MODEL_FILE" ]]; then
        echo "Unknown model key: $MODEL_KEY" >&2
        echo "Available models:" >&2
        echo "$MODEL_MAP" | awk -F: '{print "  " $1}' >&2
        exit 1
    fi

    MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Model file not found: $MODEL_PATH" >&2
        exit 1
    fi
}

# =========================
# Select llama binary
# =========================
select_binary() {
    case "$MODE" in
        completion) LLAMA_BIN="./build/bin/llama-completion" ;;
        cli)        LLAMA_BIN="./build/bin/llama-cli" ;;
        server)     LLAMA_BIN="./build/bin/llama-server" ;;
        *)
            echo "Invalid mode: $MODE" >&2
            echo "Valid modes: completion | cli | server" >&2
            exit 1
            ;;
    esac
}

# =========================
# Build command
# =========================
build_cmd() {
    CMD=(
        "$LLAMA_BIN"
        --model "$MODEL_PATH"
        --ctx-size "$CONTEXT"
        --gpu-layers "$NGL"
    )

    [[ -n "$GPU" ]] && CMD+=(--main-gpu "$GPU" --split-mode none)

    if [[ "$MODE" == "completion" || "$MODE" == "cli" ]]; then
        CMD+=(
            --prompt "$PROMPT"
            --system-prompt "$SYS_PROMPT"
            --conversation
            --predict 1000
            --repeat-penalty 1.5
            --chat-template vicuna
            --temp 0.5
            --reasoning-format deepseek
            --escape
        )
    fi
}

# =========================
# Run
# =========================
run() {
    echo "Mode:  $MODE"
    echo "Model: $MODEL_FILE"
    "${CMD[@]}" 2>/dev/null
}

# =========================
# Main
# =========================
main() {
    setup_env
    init_defaults
    init_allowed_keys
    load_helpers
    init_model_map
    parse_args "$@"
    resolve_model
    select_binary
    build_cmd
    run
}

# =========================
# Entrypoint
# =========================
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi