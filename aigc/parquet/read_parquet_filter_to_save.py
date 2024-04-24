from datasets import load_dataset

# 读取Parquet文件
dataset = load_dataset('parquet', data_files='input.parquet')

# 筛选数据
filtered_dataset = dataset.filter(lambda example: example['column_name'] == 'value')  # 这里的'column_name'和'value'需要替换为你的列名和值

# 存储筛选后的数据到新的Parquet文件
filtered_dataset.save_to_disk('output.parquet')
