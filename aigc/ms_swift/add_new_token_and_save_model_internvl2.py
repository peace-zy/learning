import os
import shutil
import json
import torch
import copy
from transformers import AutoModel, AutoTokenizer, AddedToken
from tqdm import tqdm

import sys
InvertedWallType = {
    "⾃动": 0,
    "外墙": 1,
    "内墙": 2,
    "虚拟墙": 3,
    "玻璃落地墙": 4,
    "栅栏": 5,
    "玻璃墙": 6,
    "矮墙": 7,
    "悬空墙": 8
}

def add_new_tokens_to_save(model_name_or_path='models/OpenGVLab/InternVL2-8B', output_dir='./new'):
    #max_seq_length = 14400
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = model_name_or_path
    #tokenizer.model_max_length = max_seq_length

    new_token_list = []
    max_floor_num = 15 + 1
    wall_token = [f'"墙_{idx}_' for idx in range(1, max_floor_num)]
    new_token_list.extend(wall_token)
    wall_type_tokens = [f'"{wall_type}"' for wall_type in InvertedWallType.keys()]
    new_token_list.extend(wall_type_tokens)
    add_special_tokens = ['{"墙体列表":', '"id":', '"坐标":', '"x":', '"y":',
                          '[{', '"墙体宽度":', '"弧形墙凸出的距离":', '"类型列表":', 
                          '"墙体附件列表":', '"起始点":', '"结束点":', '"长度":',
                          '"rotateX":', '"rotateY":', '"分间列表":', '"轮廓":']
    new_token_list.extend(add_special_tokens)
    door_window_token = [f'"{line_item}_{idx}_' for line_item in InvertedDoorWindowType.keys() for idx in range(1, max_floor_num)]
    new_token_list.extend(door_window_token)
    entrance_door = [f'"入户{line_item}_{idx}_' for line_item in InvertedDoorWindowType.keys() for idx in range(1, max_floor_num) if '门' in line_item]
    new_token_list.extend(entrance_door)
    sub_room_token = [f'"{sub_room}_{idx}_' for sub_room in InvertedSubRoomType.keys() for idx in range(1, max_floor_num)]
    new_token_list.extend(sub_room_token)
    unique_token_list = []
    for token in new_token_list:
        if token not in unique_token_list:
            unique_token_list.append(token)
    #token_list = list(set(token_list))
    #num_new_tokens = tokenizer.add_tokens(unique_token_list, special_tokens=False)
    need_add_tokens = []
    for token in unique_token_list:
        new_token = AddedToken(
            token, single_word=False, lstrip=True, 
            rstrip=True, special=False, normalized=True)
        need_add_tokens.append(new_token)

    num_new_tokens = tokenizer.add_tokens(need_add_tokens, special_tokens=False)
    #tokenizer.added_tokens_encoder
    print(f'Added {num_new_tokens} new tokens')

    tokenizer.save_pretrained(output_dir)
    # added_tokens.json tokenizer_config.json
    return tokenizer.total_vocab_size

def init_tokenizer(model_name_or_path='models/OpenGVLab/InternVL2-8B'):
    max_seq_length = 14400
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = model_name_or_path
    tokenizer.model_max_length = max_seq_length
    
    return tokenizer

def main():
    test_jsonl = 'test.jsonl'
    
    original = 'Model/OpenGVLab/InternVL2-2B'
    new = 'Model/OpenGVLab/InternVL2-2B-add-token'

    if os.path.exists(new):
        shutil.rmtree(new)

    print(f'Adding new tokens to the tokenizer and save to the path...{new}')
    total_vocab_size = add_new_tokens_to_save(original, new)
    print('New tokens added and saved successfully!')
    print(f'total_vocab_size={total_vocab_size}')

    print(f'Copying the original model file from [{original}] to [{new}]')
    for file in tqdm(os.listdir(original), desc='Copying'):
        old_file = os.path.join(original, file)
        new_file = os.path.join(new, file)
        if not os.path.exists(new_file):
            print(f'Copying {file}')
            if os.path.isfile(old_file):
                shutil.copy(old_file, new)
            else:
                shutil.copytree(old_file, new_file)

    print('Model files copied successfully!')
    print(f'Updating the config.json file in the path...{new}')
    new_config_file = os.path.join(new, 'config.json')
    new_config_file_bk = os.path.join(new, 'config.json.bk')
    with open(new_config_file, 'r') as f:
        config = json.load(f)
    config['llm_config']['vocab_size'] = total_vocab_size
    with open(new_config_file, 'w') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print('config.json file updated successfully!')
    shutil.copy(new_config_file, new_config_file_bk)

    print(f'Loading the original model from the path...{original}')
    model = AutoModel.from_pretrained(
        original,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True).eval().cuda()
    print('original Model loaded successfully!')

    # Initialize the model with the original model's parameters
    need_init_with_original = {
        'language_model.model.tok_embeddings.weight': None,
        'language_model.output.weight': None,
    }
    for key, value in model.named_parameters():
        if key in need_init_with_original:
            need_init_with_original[key] = copy.deepcopy(value)
    del model
    torch.cuda.empty_cache()
    print('Get key data and model destroyed successfully!')

    print(f'Loading the new model from the path...{new}')
    model = AutoModel.from_pretrained(
        new,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
        trust_remote_code=True).eval().cuda()
    print('new Model loaded successfully!')
    print('Initializing the new model with the original model\'s parameters...')
    for key, value in model.named_parameters():
        if key in need_init_with_original:
            original_vocab_size = need_init_with_original[key].shape[0]
            print(f'original_vocab_size={original_vocab_size}')
            print(f'original_data={value.data}')
            value.data[:original_vocab_size] = need_init_with_original[key].data
            print(f'new_data={value.data}')
            
    print('Model initialized successfully!')

    new_path = os.path.join(new, 'temp')
    os.makedirs(new_path, exist_ok=True)
    print(f'Saving the model to the new path...{new_path}')
    model.save_pretrained(new_path)
    print('Model saved successfully!')
    if '8B' in original:
        need_update_filenames = [
            'model-00001-of-00004.safetensors',
            'model-00002-of-00004.safetensors',
            'model-00003-of-00004.safetensors',
            'model-00004-of-00004.safetensors',
            'model.safetensors.index.json'
        ]
    else:
        need_update_filenames = [
            'model.safetensors'
        ]
    
    for file in os.listdir(new_path):
        if file in need_update_filenames:
            shutil.move(os.path.join(new_path, file), os.path.join(new, file))
    shutil.rmtree(new_path)
    
    #shutil.move(new_config_file_bk, new_config_file)

    original_tokenizer = init_tokenizer(original)
    print(f'original total_vocab_size={original_tokenizer.total_vocab_size}')
    new_tokenizer = init_tokenizer(new)
    print(f'new total_vocab_size={new_tokenizer.total_vocab_size}')


    with open(test_jsonl, 'r') as f:
        for line in tqdm(f.readlines()[:2], desc='Processing'):
            data = json.loads(line)
            conversations = data['conversations']
            text = conversations[1]['value']
            #text = text.replace(' ', '')
            original_input_ids = original_tokenizer(text, return_tensors='pt', padding=False, truncation=False).input_ids
            new_input_ids = new_tokenizer(text, return_tensors='pt', padding=False, truncation=False).input_ids
            ratio = f'{(new_input_ids.size(1) / original_input_ids.size(1)) * 100:.2f}'
            
            tokens = new_tokenizer.convert_ids_to_tokens(new_input_ids[0])
            print('\t'.join(tokens))
            print(f'frame_id={data["id"]}, {original_input_ids.size(1)} vs {new_input_ids.size(1)} {ratio}')
            break
            #all_tokens.update(tokens)
            #for input_id in new_input_ids[0]:
                #print(new_tokenizer.decode(input_id))
            #    print(new_tokenizer.convert_ids_to_tokens(input_id))
    
    '''
    with open('all_tokens_without_whitespace.txt', 'w') as f:
        for token in all_tokens:
            f.write(token + '\n')
    '''
    
    return

if __name__ == '__main__':
    main()


