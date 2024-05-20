import json
import os
from multiprocessing import Pool, Manager, Lock
import random
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None

# 常量设置
input_json = '/mnt/bella/multimodal_data/ALLaVA-4V/allava_vflan/ALLaVA-Instruct-VFLAN-4V.json'
output_jsonl = '/mnt/bella/multimodal_data/ALLaVA-4V/allava_vflan/ALLaVA-Instruct-VFLAN-4V-llava.json'
image_dir = '/mnt/bella/multimodal_data/ALLaVA-4V/'
num_processes = 64  # 进程数

def is_valid_image(image_path):
   try:
       Image.open(image_path)
       return True
   except IOError:
       return False

def process_line(data_dict):
    id = data_dict['id']
    image = data_dict['image']
    img_path = os.path.join(image_dir, image)
    if is_valid_image(img_path):
        # messages = data_dict['messages']
        # prompt = messages[0]['content'].replace("<|image|>","<image>\n")
        # # prompt = messages[0]['content'].replace("","\n")
        # if not img_path:
        #     return None
        # human_value_ori = prompt
        # conv = [{"from": "human", "value": human_value_ori},{"from": "gpt", "value": messages[1]['content']}]
        item = {"id":id, "image":image, "conversations":data_dict['conversations']}
        return json.dumps(item, ensure_ascii=False) + '\n'
    return None

def main():
    # 读取文件内容
    with open(input_json, 'r') as fr:
        data_dicts = json.load(fr)

    # 创建一个进程池
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_line, data_dicts)

    # 写入结果到文件
    with open(output_jsonl, 'w') as fw:
        for result in results:
            if result is not None:
                fw.write(result)
                fw.flush()

if __name__ == '__main__':
    main()
