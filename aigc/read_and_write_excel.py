import pandas
import sys
import traceback
from tqdm import tqdm

def read_excel(excel_file):
    """read_excel"""
    data = pandas.read_excel(excel_file, header=0)
    #data = pandas.read_excel(excel_file)
    return data

def get_data(data):
    """读取每一行数据"""
    [rows, cols] = data.shape
    for r in tqdm(range(rows)):
        yield (data.iloc[r].values)

#def write_data(out_f, w_data, out_head=['命中元素内容']):
def write_data(w_data, out_head=['命中元素内容']):
    """write_data"""
    # set column of output
    row_data = dict(zip(out_head, w_data.tolist()))
    return row_data
    out_info = [str(row_data[c]) for c in out_head]
    print(out_info)
    out_f.write('\t'.join(out_info) + '\n')

def main():
    inflile = sys.argv[1]
    data = read_excel(inflile)
    out_head = data.keys().values.tolist()
    out_data = []
    for row in get_data(data):
        try:
            out_data.append(write_data(row, out_head))
        except Exception as e:
            traceback.print_exc()
            continue
    df = pandas.DataFrame(out_data)
    writer = pandas.ExcelWriter('out.xlsx', engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}})
    df.to_excel(writer, index=False, encoding='utf-8')
    writer.close()


    return

if __name__ == '__main__':
    main()
