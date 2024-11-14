import webdataset as wds
import os
import io
import json
import shutil
import glob
from tqdm import tqdm
import concurrent.futures
import traceback
from datasets import load_dataset

def clear_cache():
    cache_path = "/root/.cache/huggingface/datasets"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)

def process_thread(idx, data_f, save_jsonl_dir, save_tar_dir, encode_format="jpg"):
    #fname = os.path.join(save_dir, "ymcc_100m_{:0>5}.tar".format(idx))
    stream = None
    out_file = os.path.join(save_jsonl_dir, f"{os.path.basename(data_f).split('.')[0]}.jsonl")
    with open(data_f, "r") as f, open(out_file, "w") as out_f:
        for line in tqdm(f, desc=f"processing {data_f}", total=10000):
            try:
                fields = line.strip().split("\t")
                if len(fields) != 3:
                    print(f"{data_f}:{fields[0]} has error fields num {len(fields)}")
                    continue
                line_id, video_name = fields[0], fields[1]
                fname = os.path.join(save_tar_dir, f"{video_name}.tar")
                stream = wds.TarWriter(fname)
                data_dict = json.loads(fields[2])
                out_dict = {
                    "video_clip_num": data_dict["video_clip_num"],
                    "clip_data_list": [],
                }
                clip_data_list = data_dict["clip_data_list"]
                for clip_data in clip_data_list:
                    frame_path_list = clip_data["frame_path_list"]
                    deep_speech_list = clip_data["deep_speech_list"]
                    if len(frame_path_list) != len(deep_speech_list):
                        print(f"frame_path_list and deep_speech_list has different length {len(frame_path_list)} vs {len(deep_speech_list)}")
                        continue
                    out_frame_path_list = []
                    for idx, frame_path in enumerate(frame_path_list):
                        if not os.path.exists(frame_path):
                            print(f"{frame_path} not exists")
                            continue
                        rel_path = frame_path.replace("xxx", "")
                        out_frame_path_list.append(rel_path)
                        subdir, image_name = os.path.split(rel_path)
                        image_name = image_name.split(".")[0]

                        sample = {}
                        sample["__key__"] = f"{subdir}/{image_name}"

                        # 从字节流管道中获取二进制
                        with open(frame_path, "rb") as f:
                            binary_str = f.read()

                        sample[encode_format] = binary_str
                        sample["json"] = json.dumps(deep_speech_list[idx])
                        stream.write(sample)
                    out_dict["clip_data_list"].append({
                        "frame_path_list": out_frame_path_list,
                    })
                out_f.write(f"{line_id}\t{video_name}\t{json.dumps(out_dict)}\n")
                stream.close()
                #break
            except Exception as e:
                traceback.print_exc()
                print(f"error: {e}")
                if stream is not None:
                    stream.close()
                continue

def dataset2tar(data_files, save_dir, encode_format="jpg"):
    save_jsonl_dir = os.path.join(save_dir, "jsonl")
    save_tar_dir = os.path.join(save_dir, "tar")
    os.makedirs(save_jsonl_dir, exist_ok=True)
    os.makedirs(save_tar_dir, exist_ok=True)
    max_workers = 30
    timeout = 50
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, data_f in enumerate(data_files):
            futures.append(executor.submit(process_thread, idx, data_f, save_jsonl_dir, save_tar_dir))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Prcossing"):
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"Thread raised an exception: {e}")
            clear_cache()

def main():
    data_path = "new_jsonl_train_data"
    data_files = glob.glob(f"{data_path}/*.jsonl")
    save_dir = "data"

    clear_cache()
    #data_files = data_files[7:]
    dataset2tar(data_files, save_dir)

if __name__ == "__main__":
    main()
