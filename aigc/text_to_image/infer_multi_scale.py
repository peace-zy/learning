import os
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline,UNet2DConditionModel, AutoencoderKL, DiffusionPipeline,DPMSolverMultistepScheduler
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModel
import argparse
import random

def main(unet_path , save_dir):
    negative_prompt = 'blurry, raw phoQto, high saturation, multiple watermark, watermark, (over-bright:1.5)'
    generator = torch.Generator(device="cuda")

    model_path = '/aistudio/workspace/aigc_chubao/cll/aigc_models/realisticVisionV51_v51VAE/'

    try:
        unet = UNet2DConditionModel.from_pretrained(unet_path, use_safetensors=True, torch_dtype=torch.float16).to('cuda')
    except:
        unet = UNet2DConditionModel.from_pretrained(unet_path, use_safetensors=False, torch_dtype=torch.float16).to(
            'cuda')
    pipe = StableDiffusionPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # pipe_dpo.set_progress_bar_config(disable=True)


    # 测试集
    df = pd.read_json('/aistudio/workspace/aigc_ssd/haoying/sd/train_data/V05/v05_all_data_for_test.jsonl',lines=True)

    max = 768
    min = 576
    # 设置随机种子
    random.seed(0)
    # 找到在范围内可以被64整除的所有数
    numbers = [i for i in range(min, max + 1) if i % 64 == 0]
    # 从这些数中随机选择一个，重复1141次
    random_list = [(random.choice(numbers), random.choice(numbers)) for _ in range(len(df))]

    # save_dir = args.save_dir
    # save_dir = '/aistudio/workspace/aigc_ssd/haoying/sd/infer_result/multi_scale/ckpt_50000_576_768/'
    os.makedirs(save_dir,exist_ok=True)
    for i, row in df.iloc[:].iterrows():
        width,height = random_list[i]

        prompt = row['text']
        if row['file_name'] not in os.listdir(save_dir):
            # 随机从5个分辨率中选择一个
            # width, height = reso_list[i%5]
            print(i, row['file_name'], width, height)
            # width = args.width
            # height = args.height
            # image = pipe(prompt,width=width,height=height,num_inference_steps=25,negative_prompt=negative_prompt).images[0]
            image = pipe(prompt,width=width,height=height,num_inference_steps=25,negative_prompt=negative_prompt,size_condition=(height,width)).images[0]
            image.save(os.path.join(save_dir,row['file_name']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unet_path', type=str, default='/aistudio/workspace/aigc_ssd/haoying/sd/ft_model/V_multi_scale/V07_9_scale_random_dataloader_5bucket/checkpoint-50000/unet')
    parser.add_argument('--save_dir', type=str, default='/aistudio/workspace/aigc_ssd/haoying/sd/infer_result/multi_scale/random_5csv_ckpt_50000_random_resolution_add_cond_random')
    args = parser.parse_args()
    main(args.unet_path, args.save_dir)


# if __name__ == '__main__':
#
#     unet_path1 = '/aistudio/workspace/aigc_ssd/haoying/sd/ft_model/V_multi_scale/V07_9_scale_random_dataloader_5bucket/checkpoint-50000/unet'
#     # resolution_list1 = [(448, 896), (576, 768), (768, 768), (768, 576), (896, 448)]
#     save_dir1 = '/aistudio/workspace/aigc_ssd/haoying/sd/infer_result/multi_scale/random_5csv_ckpt_50000_random_resolution_add_cond_random'
#     main(unet_path1,save_dir1)
#
#     unet_path2 = '/aistudio/workspace/aigc_ssd/haoying/sd/ft_model/V_multi_scale/V07_9_scale_random_dataloader_7bucket/checkpoint-50000/unet'
#     # resolution_list2 = [(448, 896),(576, 832),(576, 704),(768, 768), (704, 576),(832, 576),(896, 448)]
#     save_dir2 = '/aistudio/workspace/aigc_ssd/haoying/sd/infer_result/multi_scale/random_7csv_ckpt_50000_random_resolution_add_cond_random'
#     main(unet_path2, save_dir2)
#
#     unet_path3 = '/aistudio/workspace/aigc_ssd/haoying/sd/ft_model/V_multi_scale/V07_9_scale_random_dataloader_9bucket/checkpoint-50000/unet'
#     # resolution_list3 = [(448, 896),(576, 896),(576, 768),(704, 832),(768, 768), (832, 704),(768, 576),(896, 576),(896, 448)]
#     save_dir3 = '/aistudio/workspace/aigc_ssd/haoying/sd/infer_result/multi_scale/random_9csv_ckpt_50000_random_resolution_add_random'
#     main(unet_path3, save_dir3)

