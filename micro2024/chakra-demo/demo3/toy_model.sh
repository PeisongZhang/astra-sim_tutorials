#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
STG=${SCRIPT_DIR}/../../symbolic_tensor_graph

# Run Symbolic Tensor Graph (STG) Generator for Qwen-32B
# Configuration from qwen_32b_config.json and user requirements:
# seq=4096, batch=1024, tp=8, pp=4, dp=4
(
cd ${STG}
python3 main.py --output_dir ${SCRIPT_DIR}/toy_model/ \
               --output_name toymodel.%d.et \
               --dp 8 --tp 1 --pp 8 \
               --seq 4096 --batch 1024 \
               --dvocal 152064 --dmodel 5120 --dff 27648 \
               --head 40 --kvhead 8 --num_stacks 8 \
               --micro_batch 4 \
               --model_type llama \
               --weight_sharded 0
)
