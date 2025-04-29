set -x
set -e

# 设置时区为上海
#cat /etc/issue
#apk add tzdata
rm -f /etc/localtime
ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
date

rm -rf ~/.cache/pip
conda clean --all -y

pip config set global.trusted-host nexus.kcs.ke.com
pip config set global.index-url http://nexus.kcs.ke.com/repository/kcs-pip-proxy/simple

pip install --no-cache-dir ms_swift==3.2.0
pip install --no-cache-dir qwen-vl-utils==0.0.10
pip install --no-cache-dir lmdeploy==0.7.1
pip install --no-cache-dir vllm==0.7.3
pip install --no-cache-dir transformers==4.50.0
pip install --no-cache-dir "decord" -U

export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_11
export NCCL_ALGO=nvls
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_QPS_PER_CONNECTION=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

is_multinode=true
is_multinode=false
if [ $is_multinode = true ]; then
    nproc_per_node=8
    nnodes=${WORLD_SIZE}
    node_rank=${RANK}
    master_addr=$(cat /etc/aistudio/master-host)
    master_port=6000
else
    nproc_per_node=8
    nnodes=1
    node_rank=0
    master_addr=localhost
    master_port=23458
fi

NNODES=${nnodes}
NPROC_PER_NODE=${nproc_per_node}
NODE_RANK=${node_rank}
MASTER_ADDR=${master_addr}
MASTER_PORT=${master_port}
MACRO_BATCH=64
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$(expr ${MACRO_BATCH} / ${NPROC_PER_NODE} / ${NNODES} / ${PER_DEVICE_BATCH_SIZE})
max_length=14400
num_train_epochs=3


MODEL_PATH=open_models/Qwen/Qwen2.5-VL-3B-Instruct
OUTPUT_DIR=Model/Qwen2.5-VL-3B-Instruct-R2V-h100-${num_train_epochs}

MODEL_PATH=open_models/Qwen/Qwen2.5-VL-3B-Instruct-add-token-init-with-r2v
OUTPUT_DIR=Model/Qwen2.5-VL-3B-Instruct-add-token-init-with-r2v-h100-${num_train_epochs}

MODEL_PATH=open_models/Qwen/Qwen2.5-VL-3B-Instruct-add-token-new
OUTPUT_DIR=Model/Qwen2.5-VL-3B-Instruct-add-token-R2V-h100-${num_train_epochs}

if [ ! -d "$OUTPUT_DIR" ]; then
  # 如果目录不存在，则创建目录
  mkdir -p "$OUTPUT_DIR"
  echo "目录 $OUTPUT_DIR 已创建。"
else
  echo "目录 $OUTPUT_DIR 已存在。"
fi

echo ${DATASET}
echo 'NODE_RANK='${NODE_RANK}
WORK_SPACE=`pwd`/ms-swift-main
WORK_SPACE=`pwd`
echo 'WORK_SPACE='${WORK_SPACE}
cd ${WORK_SPACE}
LOG_FILE=${OUTPUT_DIR}/training_log_node_${NODE_RANK}.txt
echo 'LOG_FILE='${LOG_FILE}
touch ${LOG_FILE}
# 2400 * 1820
# 448 * 448 * 6
# 448 * 448 * 9
torchrun \
    --master_port ${MASTER_PORT} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    swift/cli/sft.py \
    --model ${MODEL_PATH} \
    --model_type qwen2_5_vl \
    --model_kwargs '{"min_pixels": 3136, "max_pixels": 1806336}' \
    --dataset 'train_sample.jsonl' \
    --attn_impl flash_attn \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_steps 1000000 \
    --save_steps 10000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length ${max_length} \
    --output_dir ${OUTPUT_DIR} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 20 \
    --split_dataset_ratio 0 \
    --deepspeed zero1 \
    2>&1 | tee ${LOG_FILE}
    #--deepspeed zero2
    #--custom_dataset_info dataset_info.json \ 
    #--system 'You are a helpful assistant.' \
sleep 100d
set +x
set +e
