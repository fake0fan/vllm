#!/bin/bash
set -euo pipefail

declare -a PIDS=()

###############################################################################
# Configuration -- override via env before running
###############################################################################
MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOG_PATH="${LOG_PATH:-./logs}"
mkdir -p $LOG_PATH

ENCODE_PORT="${ENCODE_PORT:-19534}"
PREFILL_PORT="${PREFILL_PORT:-19535}"
DECODE_PORT="${DECODE_PORT:-19536}"
PROXY_PORT="${PROXY_PORT:-10001}"

GPU_E="${GPU_E:-5}"
GPU_P="${GPU_P:-6}"
GPU_D="${GPU_D:-4}"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-12000}"   # wait_for_server timeout

NUM_PROMPTS="${NUM_PROMPTS:-100}"    # number of prompts to send in benchmark

export UCX_TLS=all
export UCX_NET_DEVICES=all

###############################################################################
# Helpers
###############################################################################
START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG=$LOG_PATH/encoder_${START_TIME}.log
P_LOG=$LOG_PATH/p_${START_TIME}.log
D_LOG=$LOG_PATH/d_${START_TIME}.log
PROXY_LOG=$LOG_PATH/proxy_${START_TIME}.log

wait_for_server() {
    local port=$1
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -s localhost:$port/v1/chat/completions > /dev/null; do
            sleep 1
        done" && return 0 || return 1
}

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1   # prevent re-entrancy
    
    # Kill all tracked PIDs
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill any remaining processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done
    
    # Kill the entire process group as backup
    kill -- -$$ 2>/dev/null
    
    echo "All processes stopped."
    exit 0
}

trap cleanup INT
trap cleanup USR1
trap cleanup TERM

###############################################################################
# Encoder worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
    --gpu-memory-utilization 0.4 \
    --port "$ENCODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 128 \
    --ec-transfer-config "{
        \"ec_connector\": \"MooncakeECConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {
            \"protocol\": \"rdma\",
            \"device_name\": \"\"
        }
    }" \
    >"${ENC_LOG}" 2>&1 &

PIDS+=($!)

###############################################################################
# Prefill worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_P" \
vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$PREFILL_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --ec-transfer-config "{
        \"ec_connector\": \"MooncakeECConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {
            \"protocol\": \"rdma\",
            \"device_name\": \"\"
        }
    }" \
    --kv-transfer-config '{
        "kv_connector":"MooncakeConnector",
        "kv_role":"kv_producer"
    }' \
    >"${P_LOG}" 2>&1 &

PIDS+=($!)

###############################################################################
# Decode worker
###############################################################################
CUDA_VISIBLE_DEVICES="$GPU_D" \
vllm serve "$MODEL" \
    --gpu-memory-utilization 0.7 \
    --port "$DECODE_PORT" \
    --enforce-eager \
    --enable-request-id-headers \
    --max-num-seqs 128 \
    --kv-transfer-config '{
        "kv_connector":"MooncakeConnector",
        "kv_role":"kv_consumer"
    }' \
    >"${D_LOG}" 2>&1 &

PIDS+=($!)

# Wait for workers
wait_for_server $ENCODE_PORT
wait_for_server $PREFILL_PORT
wait_for_server $DECODE_PORT

###############################################################################
# Proxy
###############################################################################
python ../disagg_epd_proxy.py \
    --host "0.0.0.0" \
    --port "$PROXY_PORT" \
    --encode-servers-urls "http://localhost:$ENCODE_PORT" \
    --prefill-servers-urls "http://localhost:$PREFILL_PORT" \
    --decode-servers-urls "http://localhost:$DECODE_PORT" \
    >"${PROXY_LOG}" 2>&1 &

PIDS+=($!)

wait_for_server $PROXY_PORT
echo "All services are up!"

###############################################################################
# Benchmark
###############################################################################
vllm bench serve \
    --model $MODEL \
    --dataset-name random-mm \
    --num-prompts 100 \
    --random-input-len 150 \
    --random-output-len 100 \
    --random-range-ratio 0.0 \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt '{"image":10,"video":0}' \
    --random-mm-bucket-config '{(560, 560, 1): 1.0}' \
    --ignore-eos \
    --backend openai-chat \
    --endpoint /v1/chat/completions \
    --port $PROXY_PORT

# cleanup
echo "cleanup..."
cleanup