CUDA_VISIBLE_DEVICES=2 python tune_fgvc.py \
    --train-type "finetune" \
    --config-file "configs/finetune/cars.yaml" \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "96" \
    DATA.FEATURE "sup_vitb16" \
    DATA.DATAPATH "/ssd2/zhangyan75/vpt/data/stanfordcars/" \
    MODEL.MODEL_ROOT "./" \
    OUTPUT_DIR "./cars_finetune_test" \
    SOLVER.TOTAL_EPOCH "20" \
    MODEL.SAVE_CKPT "True"
