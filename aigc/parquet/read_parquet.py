from datasets import load_dataset
import os
import glob
from tqdm import tqdm
import argparse
import requests
import json
import base64
import pyarrow.parquet as pq

#dataset = load_dataset("parquet", data_files={'test': 'ocrvqa/howard-hou/OCR-VQA/data/test-00000-of-00002-45d65f5057dd1a9e.parquet'})
#dataset = load_dataset("parquet", data_files={'train': 'ocrvqa/howard-hou/OCR-VQA/data/train-*.parquet','test': 'ocrvqa/howard-hou/OCR-VQA/data/test-*.parquet'})
def load_flickr30k():
    dataset = load_dataset("parquet", data_files={'test': '/mnt/aigc_chubao/zhangyan461/dataset/vlm/open_dataset/flickr30k/umaru97/flickr30k_train_val_test/data/*.parquet'})
    out_path = 'flickr30k/images'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data in tqdm(dataset['test']):
        pil_image = data['image']
        image_file_name = data['filename']
        save_file = os.path.join(out_path, image_file_name)
        if os.path.exists(save_file):
            continue
        pil_image.save(save_file)

def load_nocaps():
    """
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=L size=732x1024 at 0x7F36CE432020>, 'image_coco_url': 'https://s3.amazonaws.com/nocaps/val/0013ea2087020901.jpg', 'image_date_captured': '2018-11-06 11:04:33', 'image_file_name': '0013ea2087020901.jpg', 'image_height': 1024, 'image_width': 732, 'image_id': 0, 'image_license': 0, 'image_open_images_id': '0013ea2087020901', 'annotations_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'annotations_captions': ['A baby is standing in front of a house.', 'A little girl in a white jacket and sandals.', 'A young child stands in front of a house.', 'A child is wearing a white shirt and standing on a side walk. ', 'A little boy is standing in his diaper with a white shirt on.', 'A child wearing a diaper and shoes stands on the sidewalk.', 'A child is wearing a light-colored shirt during the daytime.', 'A little kid standing on the pavement in a shirt.', 'Black and white photo of a little girl smiling.', 'a cute baby is standing alone with white shirt']}
    """
    dataset = load_dataset("parquet", data_files={'test': 'nocaps/HuggingFaceM4/NoCaps/default/validation/*.parquet'})
    out_path = 'nocaps/val'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data in tqdm(dataset['test']):
        pil_image = data['image']
        image_file_name = data['image_file_name']
        save_file = os.path.join(out_path, image_file_name)
        pil_image.save(save_file)

def load_vizwiz():
    data_files={'test': 'vizwiz/Multimodal-Fatima/VizWiz_test/data/*.parquet',
                'train': 'vizwiz/Multimodal-Fatima/VizWiz_train/data/*.parquet',
                'val': 'vizwiz/Multimodal-Fatima/VizWiz_validation/data/*.parquet'}
    data_files={'val': 'vizwiz/Multimodal-Fatima/VizWiz_validation/data/*.parquet'}
    data_files={'test': 'vizwiz/Multimodal-Fatima/VizWiz_test/data/*.parquet'}
    data_files={'train': 'vizwiz/Multimodal-Fatima/VizWiz_train/data/*.parquet'}

    dataset = load_dataset("parquet", data_files=data_files)
    for k, v in data_files.items():
        out_path = os.path.join('vizwiz', k)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for data in tqdm(dataset[k]):
            pil_image = data['image']
            image_file_name = data['filename']
            save_file = os.path.join(out_path, image_file_name)
            if os.path.exists(save_file):
                continue
            pil_image.save(save_file)

def load_ocrvqa():
    #dataset = load_dataset("parquet", data_files={'test': 'ocrvqa/howard-hou/OCR-VQA/data/*.parquet'})
    dataset = load_dataset("parquet", data_files={'test': 'ocrvqa/howard-hou/OCR-VQA/data/test*.parquet'})
    out_path = 'ocrvqa/images_test'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data in tqdm(dataset['test']):
        image_url = data['image_url']
        GIF = False
        if image_url.endswith('.gif'):
            GIF = True
        if not GIF:
            pil_image = data['image']
            if pil_image.mode == 'RGB':
                image_file_name = data['image_id'] + '.jpg'
            elif pil_image.mode == 'RGBA':
                image_file_name = data['image_id'] + '.png'
            save_file = os.path.join(out_path, image_file_name)
            if os.path.exists(save_file):
                continue
            pil_image.save(save_file)
        else:
            bin_data = requests.get(image_url).content
            save_file = os.path.join(out_path, data['image_id'] + '.gif')
            print(image_url, save_file)
            with open(save_file, 'wb') as f:
                f.write(bin_data)


def load_mme():
    image_json_file = 'mme/data/MME_images.json'
    with open(image_json_file, 'r') as f:
        image = json.load(f)

    '''
    anno_path = 'mme/LaVIN'
    anno = {}
    for root, dirs, files in os.walk(anno_path):
        for fname in files:
            if fname.endswith('txt'):
                subdir = os.path.splitext(fname)[0].upper()
                with open(os.path.join(root, fname), 'r') as f:
                    for line in f:
                        fields = line.strip().split('\t')
                        image_file = fields[0]
                        if subdir not in anno:
                            anno[subdir] = []
                        part = os.path.splitext(image_file)
                        key = subdir + '_' + os.path.splitext(image_file)[0]
                        anno[key] = key +  os.path.splitext(image_file)[0]
    '''


    out_path = 'mme/images'
    for image_id, b64_data in image.items():
        #ARTWORK_IMG_39026
        fields = image_id.split('_')
        key = 'ARTWORK'
        key = 'LANDMARK'
        if key in image_id:
            save_path = os.path.join(out_path, key.lower())
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            image_name = fields[-1] + '.jpg'
            img_bin = base64.b64decode(b64_data)
            with open(os.path.join(save_path, image_name), 'wb') as f:
                f.write(img_bin)

def load_laion2b_multi():
    dataset = load_dataset("parquet", data_files={'test': '/chubao/tz-data-two/multimodal/LAION-5B/raw/laion2b-multi/OpenDataLab___LAION-5B/raw/laion2b-multi/part-00000-fc82da14-99c9-4ff6-ab6a-ac853ac82819-c000.snappy.parquet'})
    out_path = 'laion2b_multi/images'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for data in tqdm(dataset['test']):
        pil_image = data['image']
        image_file_name = data['filename']
        save_file = os.path.join(out_path, image_file_name)
        if os.path.exists(save_file):
            continue
        pil_image.save(save_file)

def load_laion2b_en():
    dataset = load_dataset("parquet", data_files={'test': '/chubao/tz-data-two/multimodal/LAION-5B/raw/laion2b-en/OpenDataLab___LAION-5B/raw/laion2b-en/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet'})
    """
    out_path = 'laion2b_multi/images'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    """
    for data in tqdm(dataset['test']):
        pil_image = data['image']
        image_file_name = data['filename']
        save_file = os.path.join(out_path, image_file_name)
        if os.path.exists(save_file):
            continue
        pil_image.save(save_file)

def load_yfcc100m():
    dataset = load_dataset("parquet", data_files={'test': '/chubao/tz-data-two/multimodal/YFCC-100M/justram/yfcc100m_openai_subset/train/all-00000-of-00149.parquet'})
    for data in tqdm(dataset['test']):
        print(data)

def load_laion2b_mul():
    #dataset = load_dataset("parquet", data_files={'test': '/chubao/tz-data-two/multimodal/LAION-5B/raw/laion2b-multi/OpenDataLab___LAION-5B/raw/laion2b-multi/part-00000-fc82da14-99c9-4ff6-ab6a-ac853ac82819-c000.snappy.parquet'})
    cache_path = "/chubao/tz-data-two/multimodal/LAION-5B/cache"
    cache_dir = os.makedirs(cache_path, exist_ok=True)
    data_root = "/chubao/tz-data-two/multimodal/LAION-5B/raw/laion2b-multi/OpenDataLab___LAION-5B/raw/laion2b-multi"
    data_list = glob.glob(os.path.join(data_root, "*.parquet"))
    #data_list = [os.path.join(data_root, fname) for fname in os.listdir(data_root)]
    out_path = "laion2b-zh"
    os.makedirs(out_path, exist_ok=True)
    idx = 0
    idx = 4
    step = 1
    data_list = data_list[4:]
    for i in tqdm(range(0, len(data_list), step)):
        data_files = data_list[i:i + step]
        print(idx, data_files)
        dataset = load_dataset("parquet", data_files={'test': data_files}, cache_dir=cache_path)
        # {'sample_id': 1017833016590.0, 'url': 'https://photo-cms-baonghean.zadn.vn/c204x114/Uploaded/2021/aslxmlyrmyl/2021_06_18/bna_ct2375427_1862021.jpg', 'text': 'Cách ly xã hội thành phố Vinh theo Chỉ thị 16 của Thủ tướng Chính phủ', 'width': 204, 'height': 114, 'similarity': 0.306819349527359, 'hash': -9223242262030947308, 'punsafe': 0.014692336320877075, 'pwatermark': 0.2970988154411316, 'image_suffix': 'jpg', 'license': '?', 'nsfw': 'UNLIKELY', 'aesthetic_score': None, 'language': 'vi'}
        filtered_dataset = dataset["test"].filter(lambda example: example['language'] == 'zh')
        filtered_dataset.to_parquet(f"{out_path}/laion2b-zh-{idx}.parquet")
        idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    args = parser.parse_args()

    func = 'load_{}'.format(args.dataset)

    print('\033[32mloading\033[0m {}'.format(args.dataset))
    eval(func)()
    print('\033[32mload  successfully\033[0m {}'.format(args.dataset))

if __name__ == '__main__':
    main()
