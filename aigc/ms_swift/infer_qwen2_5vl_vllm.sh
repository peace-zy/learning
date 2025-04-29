
MODEL_PATH=Model/Qwen2.5-VL-3B-Instruct-add-token-R2V-h100-3/v0-20250411-145737/checkpoint-61395
#MASTER_ADDR=10.232.64.46 \
NNODES=1 \
NPROC_PER_NODE=8 \
MASTER_ADDR=localhost \
MASTER_PORT=1001 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model ${MODEL_PATH} \
    --model_type qwen2_5_vl \
    --model_kwargs '{"min_pixels": 3136, "max_pixels": 1806336}' \
    --gpu_memory_utilization 0.7 \
    --val_dataset 'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_easy_standard_frame_variant_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_cad_to_paper_and_bend_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_friend_business_anjuke_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_friend_business_kujiale_with_furniture_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_friend_business_kujiale_without_furniture_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_friend_business_lianjiahistory_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_cg_friend_business_woaiwojia_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_hard_real_new_house_develop_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_medium_app_screen_shot_beike_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_medium_app_screen_shot_xhs_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_medium_cg_cad_to_paper_wo_answer.jsonl' \
                'ms_swift_project/dataset/qwenvl_sft_data/test_data/test_r2v_medium_standard_frame_to_cad_wo_answer.jsonl' \
    --torch_dtype bfloat16 \
    --stream false \
    --temperature 0 \
    --top_p 1.0 \
    --repetition_penalty 1.0 \
    --max_new_tokens 20000 \
    --template qwen2_5_vl \
    --task_type causal_lm \
    --attn_impl flash_attn \
    --padding_side right \
    --infer_backend vllm \
    --split_dataset_ratio 0 \
    --use_chat_template true \
    --result_path r2v_result_a100_6_1w_add_token_vllm.jsonl
    #--result_path r2v_result_a100_local_6_1w_vllm_h100.jsonl
    #--result_path r2v_result_a100_local_6w_vllm.jsonl
    #--infer_backend pt \
    #--result_path r2v_result_a100.jsonl
    #--result_path result_h100.jsonl
    #--result_path result.jsonl
