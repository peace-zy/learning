import json
import os
import sys
from tqdm import tqdm
import logging
import glob
import cv2
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from evaluate.tools import convert_unit, frame_to_png, polt_png_xyz, get_frame_json
sys.path.append(os.path.join(project_root, 'to_frame_2_0'))
from to_frame_2_0.src.frame_json_to_new_standard_converter import FrameJsonToNewStandardConverter

table_attribute = '<table border=1 id="show_table" style="top:30px;left:0px;position:absolute;word-break:break-all">\n'
table_header = '<tr><td>dataset</td><td>frame_id</td><td>原始图片</td><td>合成图</td><td>选择角度</td></tr>\n'
select_html = '''
<select name="angle">
    <option value="0">0</option>
    <option value="90">90</option>
    <option value="180">180</option>
    <option value="270">270</option>
    <option value="360">360</option>
</select>
'''
export_script = '''
<script>
    function exportToJson(prefix) {
        const table = document.getElementById('show_table');
        const rows = table.getElementsByTagName('tr');
        let data = [];

        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].getElementsByTagName('td');
            const angle = cells[4].getElementsByTagName('select')[0].value;
            data.push({
                dataset_name: cells[0].innerText,
                frame_id: cells[1].innerText,
                ori_image_file: cells[2].getElementsByTagName('img')[0].src,
                entrance_null_image_file: cells[3].getElementsByTagName('img')[0].src,
                angle: angle
            });
        }

        const json = JSON.stringify(data);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = prefix + '_选择结果.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function exportUpdatedHtml(prefix) {
        const table = document.getElementById('show_table');
        const rows = table.getElementsByTagName('tr');

        // Update the select elements with the current values
        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].getElementsByTagName('td');
            const select = cells[4].getElementsByTagName('select')[0];
            const selectedValue = select.value;
            const options = select.getElementsByTagName('option');
            for (let j = 0; j < options.length; j++) {
                if (options[j].value === selectedValue) {
                    options[j].setAttribute('selected', 'selected');
                } else {
                    options[j].removeAttribute('selected');
                }
            }
        }

        const htmlContent = document.documentElement.outerHTML;
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = prefix + '_更新后的.html';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
</script>
'''
prefix_url = 'http://10.232.64.19-8801.zt.ke.com/'

def polt_png(new_json, out_image_file, source="xyz"):
    new_json = convert_unit(new_json)
    new_json = get_frame_json(new_json)
    if source != "xyz":
        new_png_data = frame_to_png(json.dumps(new_json, indent=4, ensure_ascii=False))
    else:
        new_png_data = polt_png_xyz(json.dumps(new_json, ensure_ascii=False, ).encode("utf-8"))

    header = new_png_data[:8]
    if header != b'\x89PNG\r\n\x1a\n':
        return False
    with open(out_image_file, "wb") as f:
        f.write(new_png_data)
    return True

def get_frame_json_offline(frame_json_path):
    try:
        with open(frame_json_path, 'r') as f:
            frame_json = f.read()
        data = json.loads(frame_json)
        if isinstance(data, dict):
            return data
        else:
            frame_json = json.loads(data)
            return frame_json
    except Exception as e:
        logging.exception(f"get_frame_json_offline [frame_json_path] failed, error: {e}")
        return None

def load_frame_json_data():


    frame_json_file_dict = {}
    for dataset, vector_dir in tqdm(frame_json_dict.items(), desc='Loading vector files'):
        file_list = glob.glob(os.path.join(vector_dir, '*.json'))
        logging.info(f'{dataset} has {len(file_list)} files')
        for file in file_list:
            frame_json_file_dict[os.path.basename(file).split('.')[0]] = file
    return frame_json_file_dict

def get_ori_render_image(frame_id, frame_json_file_dict, output_dir):
    if frame_id not in frame_json_file_dict:
        error = f'frame_id {frame_id} not in frame_json_file_dict'
        logging.warning(error)
        return 1, None, error

    frame_json_file = frame_json_file_dict[frame_id]
    frame_json = get_frame_json_offline(frame_json_file)
    floorplans = frame_json['floorplans']

    for floorplan in floorplans:
        lineItems = floorplan['lineItems']
        for lineItem in lineItems:
            if lineItem['entrance'] is not None:
                lineItem['entrance'] = None

    output_image_file = os.path.join(output_dir, f'{frame_id}.png')
    if not os.path.exists(output_image_file):
        new_png_data = frame_to_png(json.dumps(frame_json, indent=4, ensure_ascii=False))
        header = new_png_data[:8]
        if header != b'\x89PNG\r\n\x1a\n':
            return 1, None, 'frame_to_png failed'
        with open(output_image_file, "wb") as f:
            f.write(new_png_data)

    return 0, output_image_file, None

def process_eval_data(test_jsonl_file, output_dir='temp_show_test_image', frame_json_file_dict={}):
    output_image_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)
    show_data = {}

    with open(test_jsonl_file, 'r') as f:
        for line in tqdm(f.readlines(), desc="Processing"):
            data = json.loads(line)
            dataset_name = data['image'].split('/')[-2]
            if dataset_name not in show_data:
                show_data[dataset_name] = []

            frame_id = data['id']
            ori_image_file = data['image']
            status, entrance_null_image_file, error = get_ori_render_image(frame_id, frame_json_file_dict, output_image_dir)
            if status:
                return 1, None, error

            show_data[dataset_name].append((dataset_name, frame_id, ori_image_file, entrance_null_image_file))

    output_html_dir = os.path.join(output_dir, 'html')
    os.makedirs(output_html_dir, exist_ok=True)
    for ds_name, ds_info in tqdm(show_data.items(), desc='Writing html'):
        with open(os.path.join(output_html_dir, f'{ds_name}.html'), 'w') as f:
            f.write('<html><head><title>选择角度</title></head><body>')
            f.write(table_attribute)
            f.write(table_header)

            for dataset_name, frame_id, image_file, merged_image_file in ds_info:
                merged_image_file = os.path.abspath(merged_image_file)
                f.write(f'<tr><td>{dataset_name}</td><td>{frame_id}</td><td><img src="{prefix_url}{image_file}" width=300 border=1 controls></td><td><img src="{prefix_url}{merged_image_file}" width=300 border=1 controls></td><td>{select_html}</td></tr>\n')

            f.write('</table>')
            f.write(f'<button onclick="exportToJson(\'{ds_name}\')">导出到Json</button>')
            f.write(f'<button onclick="exportUpdatedHtml(\'{ds_name}\')">导出更新后的HTML</button>')
            f.write(export_script)
            f.write('</body></html>')

def main():
    test_jsonl_file = 'InternVL2_8B_aistudio_version2_new_v2_res.jsonl'
    logging.info('[Start] load_frame_json_data')
    frame_json_file_dict = load_frame_json_data()
    logging.info('[End] load_frame_json_data')
    logging.info(f'frame_json_file_dict has {len(frame_json_file_dict)} files')
    logging.info('[Start] process_eval_data')
    process_eval_data(
        test_jsonl_file,
        output_dir='temp_show_test_image',
        frame_json_file_dict=frame_json_file_dict
    )
    logging.info('[End] process_eval_data')

if __name__ == '__main__':
    main()
