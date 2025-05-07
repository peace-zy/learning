import json
import os
from tqdm import tqdm

def load_annotation(annotation_file):
    with open(annotation_file, 'r') as f:
        for line in tqdm(f.readlines(), desc='Processing'):
            data = json.loads(line)
            yield data

def train_data_to_qwenvl_format(annotation_file, root, out_qwenvl_file):
    num = 0
    with open(out_qwenvl_file, 'w') as f:
        for data in load_annotation(annotation_file):
            image_name = data['image']
            conversations = data['conversations']
            question = conversations[0]['value'].replace('<image>\n', '<image>')
            answer = conversations[1]['value']
            qwenvl_data = {
                'messages': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ],
                'image': [os.path.join(root, image_name)]
            }
            num += 1
            f.write(json.dumps(qwenvl_data, ensure_ascii=False) + '\n')
    return num

def test_data_to_qwenvl_format(annotation_file, root, out_qwenvl_file):
    num = 0
    with open(out_qwenvl_file, 'w') as f:
        for data in load_annotation(annotation_file):
            image_name = data['image']
            question = f"<image>{data['question']}"
            answer = data['answer']
            qwenvl_data = {
                'messages': [
                    {'role': 'user', 'content': question},
                    {'role': 'assistant', 'content': answer}
                ],
                'image': [os.path.join(root, image_name)]
            }
            num += 1
            f.write(json.dumps(qwenvl_data, ensure_ascii=False) + '\n')
    return num

def test_data_to_qwenvl_format_wo_answer(annotation_file, root, out_qwenvl_file):
    num = 0
    with open(out_qwenvl_file, 'w') as f:
        for data in load_annotation(annotation_file):
            image_name = data['image']
            question = f"<image>{data['question']}"
            answer = data['answer']
            qwenvl_data = {
                'messages': [
                    {'role': 'user', 'content': question},
                    #{'role': 'assistant', 'content': answer}
                ],
                'image': [os.path.join(root, image_name)]
            }
            num += 1
            f.write(json.dumps(qwenvl_data, ensure_ascii=False) + '\n')
    return num

def convert(internvl_dataset_file, out_qwenvl_dataset):
    with open(internvl_dataset_file, 'r') as f:
        dataset = json.load(f)
    train_data = []
    test_data = []
    train_root = f'{out_qwenvl_dataset}/train_data'
    test_root = f'{out_qwenvl_dataset}/test_data'
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)
    #dataset_info = []
    for ds_name, ds_info in tqdm(dataset.items(), desc='Converting'):
        train_image_root = ds_info.get('root', '')
        annotation = ds_info.get('annotation', '')
        difficulty = ds_info.get('difficulty', '')
        description = ds_info.get('description', '')
        if train_image_root:
            save_train_file = f'{train_root}/train_r2v_{difficulty}_{ds_name}.jsonl'
            print(f'Processing {ds_name} train data')
            train_num = train_data_to_qwenvl_format(annotation, train_image_root, save_train_file)

            train_data.append({
                'name': ds_name,
                'annotation': save_train_file,
                'length': train_num,
                'difficulty': difficulty,
                'description': description,
            })
        else:
            print(f'No root for {ds_name}')
        test_info = ds_info.get('test_info', '')
        if not test_info:
            print(f'No test info for {ds_name}')
            continue
        else:
            test_image_root = test_info.get('root', '')
            test_annotation = test_info.get('annotation', '')
            if not test_image_root:
                print(f'No test root for {ds_name}')
                continue
            save_test_file = f'{test_root}/test_r2v_{difficulty}_{ds_name}.jsonl'
            print(f'Processing {ds_name} test data')
            test_num = test_data_to_qwenvl_format(test_annotation, test_image_root, save_test_file)
            test_data.append({
                'name': ds_name,
                'annotation': save_test_file,
                'length': test_num,
                'difficulty': difficulty,
                'description': description,
            })

            save_test_file = f'{test_root}/test_r2v_{difficulty}_{ds_name}_wo_answer.jsonl'
            print(f'Processing {ds_name} test data')
            test_num = test_data_to_qwenvl_format_wo_answer(test_annotation, test_image_root, save_test_file)
            test_data.append({
                'name': ds_name,
                'annotation': save_test_file,
                'length': test_num,
                'difficulty': difficulty,
                'description': description,
            })

    with open(f'{out_qwenvl_dataset}/train_data.json', 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(f'{out_qwenvl_dataset}/test_data.json', 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

def main():
    internvl_dataset_file = 'dataset/r2v/train_json/r2v_sft_version4_without_space.json'
    out_qwenvl_dataset = 'qwenvl_sft_data'
    os.makedirs(out_qwenvl_dataset, exist_ok=True)
    convert(internvl_dataset_file, out_qwenvl_dataset)
if __name__ == '__main__':
    main()
