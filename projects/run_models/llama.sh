#!/usr/bin/env bash
# =========================
# Usage Examples:
# =========================
# CLI usage (default mode 'completion'):
# ./llama.sh
#
# ./llama.sh prompt="Explain transformers simply"
#
# ./llama.sh model=llama3instr sys_prompt="Context: " prompt="Summarize AI news"
#
# ./llama.sh 1 model=mistral7binstrq5 prompt="Write a poem about space"
#
# ./llama.sh mode=cli model=llama2
#
# ./llama.sh mode=server model=mistral7q4
#
# Redirect output to a file:
# ./llama.sh prompt="Summarize AI news" > output.txt
#
# =========================
# Library usage (sourced):
# =========================
# source ./llama.sh
# llama_run model=llama3instr prompt="Hello from library"
#
# Or call main() directly:
# main mode=completion model=mistral7binstrq5 prompt="Explain transformers"
#
# =========================
# Notes:
#   - Available model keys: llama2, llama3instr, mistral7binstrq4, mistral7binstrq5, mistral7q4
#   - Default GPU is 0, default context size is 4096 tokens
#   - Valid modes: completion, cli, server
#   - The script only prints chat output; debug messages are suppressed

init_env_vars() {
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    export ZES_ENABLE_SYSMAN=1
}

init_defaults_params() {
    MODELS_DIR="$HOME/models"
    MODEL_KEY="mistral7binstrq4"
    CONTEXT=4096
    NGL=99
    PROMPT="No prompt provided."
    SYS_PROMPT=""
    GPU="0"
    MODE="completion"
}

init_allowed_keys() {
    ALLOWED_KEYS=(
        model
        sys_prompt
        prompt
        mode
    )
}

init_model_map() {
    MODEL_MAP=$(cat <<EOF
llama2_q4:llama-2-7b.Q4_0.gguf
llama3instr_q5:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
mistral7binstr_q4:mistral-7b-instruct-v0.1.Q4_K_M.gguf
mistral7binstr_q5:mistral-7b-instruct-v0.1.Q5_K_M.gguf
qwen3b_q6:Qwen2.5-3B-Q6_K-Instruct.gguf
smollm2_q6:SmolLM2-1.7B-Q6_K-Instruct.gguf
EOF
)
}

load_helpers() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$SCRIPT_DIR/arg_utils.sh"
}


parse_and_check_args() {
    for arg in "$@"; do
        
        if [[ "$arg" == *=* ]]; then
            key="${arg%%=*}"

            if [[ ! " ${ALLOWED_KEYS[*]} " =~ " $key " ]]; then
                echo "Unknown parameter: $key" >&2
                suggestion=$(suggest_key "$key" || true)
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
                if [[ "$arg" =~ ^[0-9]+$ ]]; then
                    GPU="$arg"
                else
                    echo "Invalid argument: $arg" >&2
                    exit 1
                fi
                ;;
        esac
    done
}

resolve_model() {
    if [[ "$MODEL_KEY" == "all" ]]; then

        MODEL_PATH=""
        USE_MODELS_DIR=1
    else
        USE_MODELS_DIR=0
        MODEL_FILE=$(awk -F: -v k="$MODEL_KEY" '$1 == k {print $2}' <<< "$MODEL_MAP")

        if [[ -z "$MODEL_FILE" ]]; then
            echo "Unknown model key: $MODEL_KEY" >&2
            echo "Available models:" >&2
            awk -F: '{print "  " $1}' <<< "$MODEL_MAP" >&2
            exit 1
        fi

        MODEL_PATH="$MODELS_DIR/$MODEL_FILE"

        [[ -f "$MODEL_PATH" ]] || {
            echo "Model file not found: $MODEL_PATH" >&2
            exit 1
        }
    fi
}


select_llama_binary() {
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

build_cmd() {
    CMD=("$LLAMA_BIN")

    # Decide which argument to pass
    if [[ "$MODE" == "server" && "$USE_MODELS_DIR" == 1 ]]; then
        CMD+=(--models_dir "$MODELS_DIR")
    else
        CMD+=(--model "$MODEL_PATH")
    fi

    CMD+=(
        --ctx-size "$CONTEXT"
        --gpu-layers "$NGL"
        --main-gpu "$GPU"
        --split-mode none
    )

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


run() {
    echo "Mode:  $MODE"
    echo "Model: $MODEL_FILE"
    exec "${CMD[@]}"
}


main() {
    init_env_vars
    init_defaults_params
    init_allowed_keys
    init_model_map

    load_helpers
    parse_and_check_args "$@"
    resolve_model

    select_llama_binary

    build_cmd

    run
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
