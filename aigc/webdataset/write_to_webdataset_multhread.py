import multiprocessing as mp
import webdataset as wds
import pickle
import os
import io
import json
import shutil
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset

def clear_cache():
    cache_path = "/home/xx/.cache/huggingface/datasets"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

def process_thread(idx, data_f, save_dir, encode_format="jpg"):
    fname = os.path.join(save_dir, "ymcc_100m_{:0>5}.tar".format(idx))
    stream = wds.TarWriter(fname)
    dataset = load_dataset("parquet", data_files={'test': data_f})
    #step = 1 / len(dataset['test'])
    # {'photoid': 689433, 'uid': '48600082269@N01', 'title': 'Gastown, Vancouver', 'description': 'water street', 'downloadurl': 'http://farm1.staticflickr.com/1/689433_2c170de1b2.jpg', 'key': 'ff54a0ff55feeaefe0ace74cabd7d546', 'image': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=500x375 at 0x7FBDF010C250>}
    bar = tqdm(total=len(dataset["test"]), desc=f"thread_{idx}")
    for data_idx, data in enumerate(dataset['test']):
        sample = {}
        sample["__key__"] = "ymcc_100m_%06d" % data_idx
        # 创建一个字节流管道
        img_bytes = io.BytesIO()
        # 将图片数据存入字节流管道， format可以按照具体文件的格式填写
        data["image"].save(img_bytes, format="PNG")
        # 从字节流管道中获取二进制
        binary_str = img_bytes.getvalue()

        sample[encode_format] = binary_str
        data.pop("image")
        data["name"] = f"{sample['__key__']}.{encode_format}"
        sample["json"] = json.dumps(data, ensure_ascii=False)
        stream.write(sample)
        #bar.update(1 * step)
        bar.update(1)
    stream.close()

"""
def dataset2tar(data_files, save_dir, encode_format="jpg"):
    #dataset = load_dataset("parquet", data_files={'test': data_files})
    num = len(data_files)
    WORKS = 4
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKS) as executor:
        # Start the load operations and mark each future with its URL
        future_to_data = {executor.submit(process_thread, idx, data_f, save_dir, encode_format): (idx, data_f) for idx, data_f in enumerate(data_files)}
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=num, desc="main", colour="green"):
            (idx, data_f) = future_to_data[future]
            try:
                data = future.result()
                #with open(save_file, 'a+') as f:
                #    f.write(data)
            except Exception as exc:
                print('{}-{} generated an exception: {}'.format(idx, data_f, exc))
                continue
"""

def dataset2tar(data_files, save_dir, encode_format="jpg"):
    #dataset = load_dataset("parquet", data_files={'test': data_files})
    num = len(data_files)
    WORKS = 6
    data_parts = [data_files[i:i + WORKS] for i in range(0, num, WORKS)]
    bar = tqdm(total=num, desc="main", colour="green")
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKS) as executor:
        # Start the load operations and mark each future with its URL
        for part in tqdm(data_parts, desc="submit"):
            future_to_data = {executor.submit(process_thread, idx, data_f, save_dir, encode_format): (idx, data_f) for idx, data_f in part}
            for future in concurrent.futures.as_completed(future_to_data):
                (idx, data_f) = future_to_data[future]
                try:
                    data = future.result()
                    bar.update(1)
                except Exception as exc:
                    print('{}-{} generated an exception: {}'.format(idx, data_f, exc))
                    continue
            clear_cache()

def main():
    data_path = "/chubao/tz-data-two/multimodal/YFCC-100M/justram/yfcc100m_openai_subset/train"
    data_files = [(idx, os.path.join(data_path, parquet_file)) for idx, parquet_file in enumerate(os.listdir(data_path))]
    #save_dir = "/mnt/cfs/multimodal_data/YFCC-100M/images"
    save_dir = "/mnt/cfs/chubaofs-tz01-mm-data/multimodal_data/YFCC-100M/images"
    os.makedirs(save_dir, exist_ok=True)
    clear_cache()
    #data_files = data_files[7:]
    dataset2tar(data_files, save_dir)

if __name__ == "__main__":
    main()
