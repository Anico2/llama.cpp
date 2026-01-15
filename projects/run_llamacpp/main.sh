#!/usr/bin/env bash
# =========================
# Usage Examples:
# =========================
# CLI usage (default mode 'completion'):
# ./run_llamacpp/main.sh
#
# ./run_llamacpp/main.sh prompt="Explain transformers simply"
#
# ./run_llamacpp/main.sh model=llama3instr sys_prompt="Context: " prompt="Summarize AI news"
#
# ./run_llamacpp/main.sh 1 model=mistral7binstrq5 prompt="Write a poem about space"
#
# ./run_llamacpp/main.sh mode=cli model=llama2
#
# ./run_llamacpp/main.sh mode=server model=mistral7q4
# =========================


init_env_vars() {
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    export ZES_ENABLE_SYSMAN=1
}

init_defaults_params() {
    MODELS_DIR="$HOME/models"
    MODEL_KEY="mistral7binstrq4"
    EMBED_MODEL_KEY="nomic-embed-text-v1.5.Q8_0"
    CONTEXT=4096
    NGL=99
    PROMPT="No prompt provided."
    SYS_PROMPT=""
    GPU="0"
    MODE="completion"

    SERVER_PORT=8080
    EMBED_PORT=8081
}


init_allowed_keys() {
    ALLOWED_KEYS=(
        model
        embed_model
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

# Embeddings
nomic_embed:nomic-embed-text-v1.5.Q8_0.gguf
bge_small:bge-small-en-v1.5.Q8_0.gguf
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
            embed_model=*) EMBED_MODEL_KEY="${arg#*=}" ;;
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

resolve_embed_model() {
    EMBED_MODEL_FILE=$(awk -F: -v k="$EMBED_MODEL_KEY" '$1 == k {print $2}' <<< "$MODEL_MAP")

    if [[ -z "$EMBED_MODEL_FILE" ]]; then
        echo "Unknown embedding model key: $EMBED_MODEL_KEY" >&2
        exit 1
    fi

    EMBED_MODEL_PATH="$MODELS_DIR/$EMBED_MODEL_FILE"

    [[ -f "$EMBED_MODEL_PATH" ]] || {
        echo "Embedding model not found: $EMBED_MODEL_PATH" >&2
        exit 1
    }
}

select_llama_binary() {
    case "$MODE" in
        completion) LLAMA_BIN="./build/bin/llama-completion" ;;
        cli)        LLAMA_BIN="./build/bin/llama-cli" ;;
        server|server_plus_embed)
                    LLAMA_BIN="./build/bin/llama-server" ;;
        *)
            echo "Invalid mode: $MODE" >&2
            exit 1
            ;;
    esac
}

add_model_arg() {
    local _cmd_ref=$1

    if [[ "$USE_MODELS_DIR" == 1 ]]; then
        eval "$_cmd_ref+=(--models_dir \"$MODELS_DIR\")"
    else
        eval "$_cmd_ref+=(--model \"$MODEL_PATH\")"
    fi
}

build_cmd() {
    CMD=("$LLAMA_BIN")

    add_model_arg CMD

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


build_server_plus_embed_cmds() {
    MAIN_CMD=("$LLAMA_BIN")
    add_model_arg MAIN_CMD

    MAIN_CMD+=(
        --ctx-size "$CONTEXT"
        --gpu-layers "$NGL"
        --main-gpu "$GPU"
        --split-mode none
        --port "$SERVER_PORT"
        --batch-size "$CONTEXT"
        --ubatch-size "$CONTEXT"
    )

    EMBED_CMD=(
        "$LLAMA_BIN"
        --model "$EMBED_MODEL_PATH"
        --embedding
        --ctx-size "$CONTEXT"
        --gpu-layers "$NGL"
        --main-gpu "$GPU"
        --split-mode none
        --port "$EMBED_PORT"
        --batch-size "$CONTEXT"
        --ubatch-size "$CONTEXT"
        --pooling mean
        --flash-attn on
        --no-webui
    )
}



run() {
    if [[ "$MODE" == "server_plus_embed" ]]; then
        echo "Starting embedding server on port $EMBED_PORT"
        "${EMBED_CMD[@]}" &

        EMBED_PID=$!

        echo "Starting main server on port $SERVER_PORT"
        exec "${MAIN_CMD[@]}"
    else
        echo "Mode:  $MODE"
        echo "Model: $MODEL_FILE"
        exec "${CMD[@]}"
    fi
}



main() {
    init_env_vars
    init_defaults_params
    init_allowed_keys
    init_model_map

    load_helpers
    parse_and_check_args "$@"

    resolve_model

    if [[ "$MODE" == "server_plus_embed" ]]; then
        resolve_embed_model
    fi

    select_llama_binary

    if [[ "$MODE" == "server_plus_embed" ]]; then
        build_server_plus_embed_cmds
    else
        build_cmd
    fi

    run
}


if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
