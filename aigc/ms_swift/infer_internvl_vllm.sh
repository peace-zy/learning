MODEL_PATH=Model/R2V_InternVL2_2B_v3_without_space_wo_real_new_develop
CUDA_VISIBLE_DEVICES=6 \
swift infer \
    --model ${MODEL_PATH} \
    --model_type internvl2 \
    --model_kwargs '{"max_num": 6, "input_size": 448}' \
    --val_dataset 'test_for_internvl.jsonl' \
    --torch_dtype bfloat16 \
    --stream false \
    --temperature 0 \
    --top_p=1.0 \
    --repetition_penalty=1.0 \
    --max_new_tokens 20000 \
    --template internvl2 \
    --task_type causal_lm \
    --attn_impl flash_attn \
    --padding_side left \
    --infer_backend vllm \
    --split_dataset_ratio 0 \
    --use_chat_template true \
    --gpu_memory_utilization 0.3 \
    --result_path internvl_result_test_vllm.jsonl
