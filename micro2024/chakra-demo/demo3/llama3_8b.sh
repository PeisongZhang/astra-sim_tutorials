#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B
# Configuration from qwen_32b_config.json and user requirements:
# seq=4096, batch=1024, tp=8, pp=4, dp=4
            #    --dvocal 128256 --dmodel 4096 --dff 14336 seq=8192 num_stacks=32\
(
cd ${STG}
# batch = 128, dp = 4, pp = 4,  batch / dp / local_micro_batch(2) = 16
python3 main.py --output_dir ${SCRIPT_DIR}/llama/ \
               --output_name llama.%d.et \
               --dp 4 --tp 1 --pp 4 \
               --seq 8192 --batch 128 \
               --dvocal 128256 --dmodel 4096 --dff 14336 \
               --head 32 --kvhead 8 --num_stacks 32 \
               --micro_batch 8 \
               --model_type llama \
               --mixed_precision true \
               --weight_sharded 0
)
