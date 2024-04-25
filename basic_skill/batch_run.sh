#!/bin/bash
set -eu
# set -xv

SINGLE_GPU_TH=$1
START_LINE=$2
END_LINE=$3
TOTAL_GPU_NUM=$(nvidia-smi --query-gpu=index --format=csv,nounits,noheader|wc -l)
echo 'TOTAL_GPU_NUM:'$TOTAL_GPU_NUM
GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,nounits,noheader|tr " " "\n")
echo 'GPU_LIST:'$GPU_LIST
THREAD_NUM=`expr $SINGLE_GPU_TH \* $TOTAL_GPU_NUM`
echo 'THREAD_NUM:'$THREAD_NUM


mkfifo tmp.$$
exec 5<>tmp.$$
rm tmp.$$

python_script='/aistudio/workspace/aigc_ssd/haoying/sd/dpo/infer_for_train.py'
python_env='/aistudio/workspace/system-default/envs/diffusers/bin/python'

# 预先写入指定数量的换行符，一个换行符代表一个进程;
for ((i=0;i<${THREAD_NUM};i++))
do
    echo -ne "\n" >&5
done


for ((i=0;i<${THREAD_NUM};i++))
do
{
   read -u5
   {
       echo 'LAUNCH--'${i}
       cuda_vis_mod=`expr ${i} % $TOTAL_GPU_NUM + 1`
       echo 'cuda_vis_mod:'$cuda_vis_mod
       cuda_vis_idx=$(echo $GPU_LIST|cut -d " " -f $cuda_vis_mod)
       echo 'cuda_vis_idx:'$cuda_vis_idx
       CUDA_VISIBLE_DEVICES=$cuda_vis_idx ${python_env} ${python_script} --split ${THREAD_NUM} --split_part ${i} --start_line ${START_LINE} --end_line ${END_LINE}
       echo -ne "\n" >&5
   }&
}
done

echo "Check the remaining threads..."

for ((i=0;i<${THREAD_NUM};i++))
do
    read -u5
    {
    echo "remain (${THREAD_NUM}-${i}) thread to run..."
    }
done


echo "RUN ALL DONE"
