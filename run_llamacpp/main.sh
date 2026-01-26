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
# ./run_llamacpp/main.sh model=mistral7binstr_q4 prompt="Write a poem about space"
#
# ./run_llamacpp/main.sh mode=cli model=llama2_q4
#
# ./run_llamacpp/main.sh mode=server model=qwen3b_q6
# =========================

init_model_map() {
    MODEL_MAP=$(cat <<EOF
### DECODERS ###
gemma1b_f16:gemma-3-1b-it-f16.gguf
gemma4b_q5:gemma-3-4b-it.Q5_K_M.gguf
liquid25_f16:LFM2.5-1.2B-Instruct-F16.gguf
llama2_q4:llama-2-7b.Q4_0.gguf
llama31instr_q5:Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
llama32instr_q8:Llama-3.2-3B-Instruct-Q8_0.gguf
mistral7binstr_q4:mistral-7b-instruct-v0.1.Q4_K_M.gguf
mistral7binstr_q5:mistral-7b-instruct-v0.1.Q5_K_M.gguf
qwen3b_q6:Qwen2.5-3B-Q6_K-Instruct.gguf
qwen3_06b_f16:Qwen3-0.6B-f16.gguf
qwen3vl_instr_4b_q4:Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf
qwen3vl_think_4b_q4:Qwen3-VL-4B-Thinking-UD-Q4_K_XL.gguf 
smollm2_q6:SmolLM2-1.7B-Instruct-Q6_K_L.gguf

### EMBEDDINGS ###
allminil16_f16:all-MiniLM-L6-v2-ggml-model-f16.gguf
nomic_embed:nomic-embed-text-v1.5.Q8_0.gguf
qwen_reranker_q8:Qwen3-reranker-0.6b-q8_0.gguf
snowflake_embed:snowflake-arctic-embed-m-v1.5.gguf

EOF
)
}

init_env_vars() {
    
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    
    # MEMORY AND PERFORMANCE RELATED VARS 
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    export ZES_ENABLE_SYSMAN=1
    # Reduces CPU-GPU latency
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
    
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"  # we have only one GPU
}

init_defaults_params() {
    ## TODO: put this into external file ##

    ## Default models and related configurations ##
    MODEL_KEY="llama31instr_q5"
    EMBED_MODEL_KEY="nomic_embed"
    CONTEXT=16384 # 0 means that llama.cpp uses model default context size
    PARALLEL=1 # real context becomes approximately -> CONTEXT / PARALLEL
               # so put this > 1 only if necessary

    ## Default prompts and modes ##
    PROMPT="No prompt provided."
    SYS_PROMPT="You are a helpful assistant."
    MODE="completion"
    ####

    ## GPU and hardware configs ##
    NGL=99 # Number of GPU layers, set to 99 to offload as much as possible to GPU
    GPU="0" # Default GPU ID
    SPLIT_MODE="none" # set to 'none' since we currently have 1 GPU only
    THREADS=4
    LLM_BATCH_SIZE=2048
    LLM_UBATCH_SIZE=2048
    EMBEDDING_BATCH_SIZE=2048
    EMBEDDING_UBATCH_SIZE=2048
    EMBEDDING_CONTEXT=2048
    ####

    ## Server ports ##
    LLM_PORT=8080
    EMBEDDING_LLM_PORT=8081
    ####
}

load_helpers() {
    
    source "$SCRIPT_DIR/arg_utils.sh"
}


parse_and_check_args() {

    ALLOWED_KEYS=(
        model
        embed_model
        sys_prompt
        prompt
        mode
        context
    )

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
            context=*)    CONTEXT="${arg#*=}" ;;
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

build_common_llama_args() {
    local _cmd_ref=$1

    eval "$_cmd_ref+=(
        --ctx-size \"$CONTEXT\"
        --gpu-layers \"$NGL\"
        --main-gpu \"$GPU\"
        --split-mode \"$SPLIT_MODE\"
        --threads \"$THREADS\"
        --batch-size "$LLM_BATCH_SIZE"
        --ubatch-size "$LLM_UBATCH_SIZE"
        --flash-attn on # currently flash-attn has issues on Intel GPUs (need to put to on for: smoll/llamainst)
        --parallel "$PARALLEL"
        --cache-type-k q8_0
        --cache-type-v q8_0
        --verbose
        
    )"
}

build_cmd() {
    CMD=("$LLAMA_BIN")
    add_model_arg CMD
    build_common_llama_args CMD

    if [[ "$MODE" == "completion" || "$MODE" == "cli" ]]; then
        CMD+=(
            --prompt "$PROMPT"
            --system-prompt "$SYS_PROMPT"
            --conversation
            --predict 1000
            --repeat-penalty 1.2
            --batch-size "$BATCH_SIZE"
            --ubatch-size "$UBATCH_SIZE"
            --temp 0.1
            --escape
        )
    fi
}


build_server_plus_embed_cmds() {
    MAIN_CMD=("$LLAMA_BIN")
    add_model_arg MAIN_CMD
    build_common_llama_args MAIN_CMD

    MAIN_CMD+=(
        --port "$LLM_PORT"
    )

    EMBED_CMD=(
        "$LLAMA_BIN"
        --model "$EMBED_MODEL_PATH"
        --embedding
        --ctx-size "$EMBEDDING_CONTEXT"
        --gpu-layers "$NGL"
        --main-gpu "$GPU"
        --split-mode "$SPLIT_MODE"
        --threads "$THREADS"
        --port "$EMBEDDING_LLM_PORT"
        --batch-size "$EMBEDDING_BATCH_SIZE"
        --ubatch-size "$EMBEDDING_UBATCH_SIZE"
        --pooling mean
        --cache-type-k f32 
        --cache-type-v f32 
        --no-webui
        --flash-attn on
    )
}

run() {
    if [[ "$MODE" == "server_plus_embed" ]]; then
        echo "Starting embedding server on port $EMBEDDING_LLM_PORT"
        "${EMBED_CMD[@]}" &

        EMBED_PID=$!

        echo "Starting main server on port $LLM_PORT"
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
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODELS_DIR="$SCRIPT_DIR/models_gguf"
    main "$@"
fi
