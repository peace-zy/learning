#!/bin/bash

# infer for single image
function infer_for_single_image()
{
    image_file=$1
    CUDA_VISIBLE_DEVICES=7 python tools/infer/infer_for_clip_offline.py \
        -m online_clip/cspd_image_similarity/__model__ \
        -p online_clip/cspd_image_similarity/__params__ \
        --image_file $image_file \
        --use_gpu=1 \
        --batch_size=1
}

# infer for benchmark
function infer_for_benchmark()
{
    batch_size=$1
    echo ${batch_size}
    CUDA_VISIBLE_DEVICES=7 python tools/infer/infer_for_clip_offline.py \
        -m online_clip/cspd_image_similarity/__model__ \
        -p online_clip/cspd_image_similarity/__params__ \
        --use_gpu=1 \
        --batch_size=${batch_size} \
        --model_name='clip' \
        --enable_benchmark=True
}

# infer for single image with tensorrt
function infer_for_single_image_in_trt()
{
    image_file=$1
    CUDA_VISIBLE_DEVICES=7 python tools/infer/infer_for_clip_offline.py \
        -m online_clip/cspd_image_similarity/__model__ \
        -p online_clip/cspd_image_similarity/__params__ \
        --image_file $image_file \
        --use_gpu=1 \
        --batch_size=1 \
        --use_tensorrt True
}

# infer for single image with tensorrt
# 在scale前添加unsqueeze op
function infer_for_single_image_with_add_unsqueeze_in_trt()
{
    image_file=$1
    CUDA_VISIBLE_DEVICES=7 python tools/infer/infer_for_clip_offline.py \
        -m add_unsqueeze_for_scale/__model__ \
        -p add_unsqueeze_for_scale/__params__ \
        --image_file $image_file \
        --use_gpu=1 \
        --batch_size=1 \
        --use_tensorrt True
}

function usage()
{
    echo -e """usage:\e[32m
        <使用线上clip离线测试-batchsize=1>
        sh infer_for_clip_offline.sh 0 {input}

        <使用线上clip离线测试-benchmark>
        sh infer_for_clip_offline.sh 1 {batch_size}

        <使用线上clip离线开启trt测试>
        sh infer_for_clip_offline.sh 2 {input}

        <使用线上clip添加unsqueeze离线开启trt测试>
        sh infer_for_clip_offline.sh 3 {input}
        \e[0m
        """
}

#===========================
#   MAIN SCRIPT
#===========================

if [ $# -eq 0 ];
then
    usage
    exit
fi


mode=$1
if [ ${mode} = 0 ];then
    echo -e "\e[32m<使用线上clip离线测试-batchsize=1>\e[0m"
    input=$2
    infer_for_single_image ${input}
elif [ ${mode} = 1 ];then
    echo -e "\e[32m<使用线上clip离线测试-benchmark>\e[0m"
    batch_size=$2
    infer_for_benchmark ${batch_size}
elif [ ${mode} = 2 ];then
    echo -e "\e[32m<使用线上clip离线开启trt测试>\e[0m"
    input=$2
    infer_for_single_image_in_trt ${input}
elif [ ${mode} = 3 ];then
    echo -e "\e[32m<使用线上clip添加unsqueeze离线开启trt测试>\e[0m"
    input=$2
    infer_for_single_image_with_add_unsqueeze_in_trt ${input}
else
    echo "invalid input param"
fi
