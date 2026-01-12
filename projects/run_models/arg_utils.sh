#!/usr/bin/env bash
# =========================
# Argument utilities
# =========================
# Helper functions for validating and suggesting CLI parameters used by llama.sh

suggest_key() {
    local bad="$1"
    local best=""
    local best_len=0

    for k in "${ALLOWED_KEYS[@]}"; do
        local i=0
        while [[ "${bad:i:1}" == "${k:i:1}" && -n "${bad:i:1}" ]]; do
            ((i++))
        done

        if (( i > best_len )); then
            best_len=$i
            best="$k"
        fi
    done

    [[ $best_len -ge 2 ]] && echo "$best"
}
