# ernie  
code https://github.com/PaddlePaddle/ERNIE
# train
CUDA_VISIBLE_DEVICES=7 python demo/finetune_classifier.py --from_pretrained ernie-1.0 --data_dir data/xnli --save_dir xnli_cls_model
# infer
CUDA_VISIBLE_DEVICES=7 python demo/infer_classifier.py --from_pretrained ernie-1.0 --data_dir data/xnli --init_checkpoint xnli_cls_model/ckpt.bin
