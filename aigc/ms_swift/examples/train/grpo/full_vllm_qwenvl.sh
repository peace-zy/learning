# Two GPUs are left for vLLM inference acceleration.
# pip install math_verify # reward function
# pip install git+https://github.com/huggingface/trl.git
# GPU memory: 8 * 60GiB
    #--vllm_device auto \
MASTER_ADDR=localhost \
MASTER_PORT=1001 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model 'open_models/Qwen/Qwen2.5-VL-3B-Instruct' \
    --model_kwargs '{"min_pixels": 3136, "max_pixels": 1806336}' \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs external_r1v_acc format \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_num_seqs 8 \
    --vllm_max_model_len 14400 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'open_dataset/multimodal-open-r1-8k-verified' \
    --max_completion_length 14400 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-7 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 6 \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --async_generate true \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero2 \
    --log_completions true \
    --num_iterations 1 \
    --num_infer_workers 2
