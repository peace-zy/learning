CUDA_VISIBLE_DEVICES=4 python train_network.py \
         --enable_bucket \
         --pretrained_model_name_or_path="stable-diffusion-webui/models/Stable-diffusion/bra_v5_zh_1.0.safetensors" \
         --train_data_dir="kohya_ss/dataset/zhongzhiya" \
         --resolution="512,512" \
         --output_dir="kohya_ss/trained_model/zhongzhiya_vglm_db_kohya" \
         --network_alpha="1" \
         --save_model_as=safetensors \
         --network_module=networks.lora \
         --text_encoder_lr=5e-05 \
         --unet_lr=0.0001 \
         --network_dim=8 \
         --output_name="zhongzhiya" \
         --lr_scheduler_num_cycles="20" \
         --learning_rate="0.0001" \
         --lr_scheduler="cosine" \
         --train_batch_size="1" \
         --save_every_n_epochs="1" \
         --mixed_precision="no" \
         --save_precision="float" \
         --cache_latents \
         --optimizer_type="Lion" \
         --max_data_loader_n_workers="0" \
         --bucket_reso_steps=64 \
         --max_train_steps=6400 \
         --bucket_no_upscale


数据集合命名，epoch_触发词/ .caption表示prompt
