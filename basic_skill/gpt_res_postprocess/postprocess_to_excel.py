import os
import re
import json
from tqdm import tqdm
import csv
import pandas as pd
from collections import OrderedDict
from itertools import zip_longest

CATEGORIES = ["感知", "认知", "推理"]
FIELDS = OrderedDict({
    "题型": "question_type",
    "问题": "question",
    "选项1": "{}['options'].get('A', '')",
    "选项2": "{}['options'].get('B', '')",
    "选项3": "{}['options'].get('C', '')",
    "选项4": "{}['options'].get('D', '')",
    "答案": "answer"
})

def convert_to_standard_json(non_standard_json, match_json_pattern):
    def ensure_brace_before_answer(text):
        # 定义正则表达式，匹配 answer 前面的字符，并检查是否存在 }
        pattern = re.compile(r'},\s*answer')

        # 搜索匹配
        match = pattern.search(text)

        if match:
            #print("在 answer 前面找到了 }")
            return text
        else:
            #print("在 answer 前面没有找到 }，正在添加...")
            # 使用正则表达式找到 answer 前面的部分
            modified_text = re.sub(r'(answer)', r'}, \1', text)
            return modified_text
    #print("before", non_standard_json)
    standard_dict = None
    non_standard_json = non_standard_json.replace('"', '').replace('，', ',').replace('：', ':')
    #print("before", non_standard_json)
    non_standard_json = ensure_brace_before_answer(non_standard_json)
    match = match_json_pattern.search(non_standard_json)
    if match:
        category = match.group(1).strip()
        question_type = match.group(2).strip()
        question = match.group(3).strip()
        options_str = match.group(4).strip()
        answer = match.group(5).strip()

        # 解析 options 字段
        options_pattern = re.compile(r'(\w+):\s*([^,]+)')
        options = dict(options_pattern.findall(options_str))
        #print(options)
        standard_dict = {
            "category": category,
            "question_type": question_type,
            "question": question,
            "options": options,
            "answer": answer
        }
    else:
        print(f"{non_standard_json} 未能匹配到所需的字段")
    #print("after", standard_dict)
    return standard_dict

def convet_one_jsonl_to_csv(jsonl_file):
    csv_file = jsonl_file.replace(".jsonl", ".csv")
    #print(csv_file)
    # 打开JSON Lines文件进行读取
    with open(jsonl_file, 'r', encoding='utf-8') as jsonl_f:
        # 打开CSV文件进行写入
        with open(csv_file, 'w', newline='', encoding='utf-8') as csv_f:
            # 创建CSV写入器
            csv_writer = csv.writer(csv_f)

            # 初始化一个变量来存储CSV的列名
            headers = None

            # 逐行读取JSON Lines文件
            for line in jsonl_f:
                #print(line)
                # 解析每一行的JSON对象
                json_obj = json.loads(line)
                standard_dict = {
                    "类型": json_obj["category"],
                    "题型": json_obj["question_type"],
                    "问题": json_obj["question"],
                    "选项1": json_obj["options"].get("A", ""),
                    "选项2": json_obj["options"].get("B", ""),
                    "选项3": json_obj["options"].get("C", ""),
                    "选项4": json_obj["options"].get("D", ""),
                    "答案": json_obj["answer"]
                }

                # 如果还没有写入列名，则写入列名
                if headers is None:
                    headers = list(standard_dict.keys())
                    csv_writer.writerow(headers)

                # 写入JSON对象的值
                csv_writer.writerow(standard_dict.values())

def save_to_csv(formated_data, out_csv_file):
    # 打开CSV文件进行写入
    with open(out_csv_file, 'w', newline='', encoding='gbk', errors='ignore') as csv_f:
        # 创建CSV写入器
        csv_writer = csv.writer(csv_f)
        # 初始化一个变量来存储CSV的列名
        headers = None
        for frame_id, json_obj in tqdm(formated_data.items(), desc="写入CSV文件"):
            for idx, (category, questions) in enumerate(json_obj.items()):
                for i, question in enumerate(questions):
                    standard_dict = {
                        "户型ID": frame_id if (idx + i) == 0 else "",
                        "类型": category
                    }

                    for field in FIELDS:
                        if "选项" in field:
                            standard_dict[field] = eval(FIELDS[field].format("question"))
                        else:
                            standard_dict[field] = question[FIELDS[field]]

                    # 如果还没有写入列名，则写入列名
                    if headers is None:
                        headers = list(standard_dict.keys())
                        csv_writer.writerow(headers)

                    # 写入JSON对象的值
                    csv_writer.writerow(list(standard_dict.values()))

def save_to_excel(formated_data, out_excel_file):
    # 初始化一个列表来存储所有行数据
    data_rows = []
    for frame_id, json_obj in tqdm(formated_data.items(), desc="写入Excel文件"):
        for idx, (category, questions) in enumerate(json_obj.items()):
            for i, question in enumerate(questions):
                standard_dict = {
                    #"户型ID": frame_id if (idx + i) == 0 else "",
                    "户型ID": frame_id,
                    "类型": category,
                }
                for field in FIELDS:
                    if "选项" in field:
                        standard_dict[field] = eval(FIELDS[field].format("question"))
                    else:
                        standard_dict[field] = question[FIELDS[field]]
                standard_dict["备注"] = ""
                standard_dict["标注员"] = ""
                # 将字典转换为列表并添加到数据行列表中
                data_rows.append(standard_dict)

    # 将数据行列表转换为DataFrame
    df = pd.DataFrame(data_rows)
    MERGE_COLOUM = True
    if MERGE_COLOUM:
        df = df.set_index(["户型ID", "类型", "题型"])
        # 将DataFrame写入Excel文件
        df.to_excel(out_excel_file, index=True, merge_cells=True)
    else:
        df.to_excel(out_excel_file, index=False, merge_cells=True)

def save_to_excel_for_annotation(formated_data, out_excel_file):
    data_rows = []

    for frame_id, json_obj in tqdm(formated_data.items(), desc="写入Excel文件for标注"):
        zipped_data = list(zip_longest(*[json_obj[c] for c in CATEGORIES], fillvalue=""))
        #print(zipped_data)
        for idx, data in enumerate(zipped_data):
            standard_dict = {
                ("户型ID", ""): frame_id if idx == 0 else ""
            }
            for category, ele in zip(CATEGORIES, data):
                for field in FIELDS:
                    if "选项" in field:
                        standard_dict[(category, field)] = eval(FIELDS[field].format("ele"))
                    else:
                        standard_dict[(category, field)] = ele[FIELDS[field]]
                standard_dict[(category, "备注")] = ""
                standard_dict[(category, "标注员")] = ""
            #print(standard_dict)
            data_rows.append(standard_dict)

    # 将数据行列表转换为DataFrame
    df = pd.DataFrame(data_rows)

    df.columns = pd.MultiIndex.from_tuples(df.columns)
    sheet_name = "Sheet1"
    writer = pd.ExcelWriter(out_excel_file)
    df1 = pd.DataFrame(columns=df.columns)
    df2 = df.droplevel(0, axis=1)

    df1.to_excel(writer, sheet_name=sheet_name, index=True)
    df2.to_excel(writer, sheet_name=sheet_name, index=True, merge_cells=True, startrow=1)
    #writer.save()
    writer.close()
    # 使用 ExcelWriter 写入 Excel 文件

def postprocess(input_dir, output_dir):
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))
    pattern = r'{"category":.*'
    """
    \s*：
        \s 表示匹配任何空白字符，包括空格、制表符、换行符等。
        * 表示匹配前面的元素零次或多次。因此，\s* 表示可以有零个或多个空白字符。
    ([^,]+)：
        ( 和 ) 是捕获组，用于捕获匹配的内容。捕获组中的内容可以通过 group 方法提取。
        [^,] 是一个字符类，表示匹配除逗号 , 之外的任何字符。^ 在字符类中表示取反。
        + 表示匹配前面的元素一次或多次。因此，[^,]+ 表示匹配一个或多个非逗号字符。
    """


    #match_json_pattern = re.compile(r'category:\s*([^,]+),\s*question_type:\s*([^,]+),\s*question:\s*([^,]+),\s*options:\s*{([^}]+)},\s*answer:\s*([^}]+)')

    match_json_pattern = re.compile(
        r'category:\s*([^}]+),\s*question_type:\s*([^}]+),\s*question:\s*([^}]+),\s*options:\s*{([^}]*)},\s*answer:\s*([^}]+)')

    formated_data = OrderedDict({})
    for file in tqdm(all_files, desc="Processing files"):
        output_file = os.path.join(output_dir, os.path.basename(file).replace(".txt", ".jsonl"))
        lines = []
        try:
            with open(file, 'r') as f:
                lines = json.load(f)
                if isinstance(lines, list):
                    lines = [json.dumps(line, ensure_ascii=False) for line in lines]
                    with open(output_file, 'w') as wf:
                        wf.write("\n".join(lines))
                elif isinstance(lines, dict):
                    with open(output_file, 'w') as wf:
                        wf.write(json.dumps(lines, ensure_ascii=False))
        except:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    match = re.search(pattern, line)
                    if match:
                        #lines.append(add_quotes_around_non_brace_chars(match.group()))
                        match_str = match.group()
                        standard_json = convert_to_standard_json(match_str, match_json_pattern)
                        #standard_json = convert_to_standard_json(match_str)
                        if standard_json is not None:
                            lines.append(json.dumps(standard_json, ensure_ascii=False))
                        else:
                            raise Exception(f"Error in file: {file}")
                with open(output_file, 'w') as wf:
                    wf.write("\n".join(lines))

        if not lines:
            print(f"Empty file: {file}")
            continue
        else:
            frame_id = os.path.basename(file).replace("_gpt4o", "").replace(".txt", ".jpg")
            prefix_url = "http://10.232.64.19-8801.zt.ke.com/nfs/a100-80G-16/wangjing/eval_data/户型评估数据_0702"
            frame_id = f"{prefix_url}/{frame_id}"
            for line in lines:
                valid_data = json.loads(line)
                if frame_id not in formated_data:
                    formated_data[frame_id] = {c: [] for c in CATEGORIES}
                category = valid_data["category"].replace("逻辑", "推理")
                valid_data.pop("category")
                formated_data[frame_id][category].append(valid_data)
        #if os.path.exists(output_file):
        #    convet_one_jsonl_to_csv(output_file)

    #save_to_csv(formated_data, "all_valid_data.csv")
    save_to_excel(formated_data, "all_valid_data.xlsx")
    save_to_excel_for_annotation(formated_data, "all_valid_data_anno.xlsx")
def main():
    input_dir = "generated_questions"
    output_dir = "post_generated_questions"
    os.makedirs(output_dir, exist_ok=True)
    postprocess(input_dir, output_dir)

if __name__ == "__main__":
    main()
