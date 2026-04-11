#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph
# NUM_ITERATIONS=${NUM_ITERATIONS:-1}
NUM_ITERATIONS=8
# DP_LOCAL_SGD_INTERVAL=${DP_LOCAL_SGD_INTERVAL:-1}
DP_LOCAL_SGD_INTERVAL=8
DEFAULT_OUTPUT_DIR=${SCRIPT_DIR}/llama/
# if [ "${NUM_ITERATIONS}" -gt 1 ] || [ "${DP_LOCAL_SGD_INTERVAL}" -gt 1 ]; then
#     DEFAULT_OUTPUT_DIR=${SCRIPT_DIR}/llama_local_sgd/
# fi
# OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}
OUTPUT_DIR=${DEFAULT_OUTPUT_DIR}

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B
# Configuration from qwen_32b_config.json and user requirements:
# seq=4096, batch=1024, tp=8, pp=4, dp=4
            #    --dvocal 128256 --dmodel 4096 --dff 14336 seq=8192 num_stacks=32\
(
cd ${STG}
# batch = 128, dp = 4, pp = 4,  batch / dp / local_micro_batch(2) = 16
            #    --seq 8192 --batch 128 \
python3 main.py --output_dir "${OUTPUT_DIR}" \
                --output_name llama.%d.et \
               --dp 4 --tp 1 --pp 4 \
               --seq 8192 --batch 128 \
                --dvocal 128256 --dmodel 4096 --dff 14336 \
               --head 32 --kvhead 8 --num_stacks 32 \
               --micro_batch 8 \
               --num_iterations "${NUM_ITERATIONS}" \
               --dp_local_sgd_interval "${DP_LOCAL_SGD_INTERVAL}" \
               --model_type llama \
               --mixed_precision true \
               --flash_attention true \
               --weight_sharded 0
)
