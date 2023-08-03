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

def main():
    infile = sys.argv[1]


    data = read_excel(infile)
    out_head = data.keys().values.tolist() + ['文生图-业务摘要', '文生图-业务点']
    out_data = []
    MAX_NUM = 10
    for row in get_data(data):
        try:
            abstract, business = row[-2], row[-1]
            abstract = abstract.strip().replace('\t', ',')

            business = business.strip().split('\t')
            d = []
            for prompt in business:
                url = random.choice(servers)
                res = run(prompt, url, '2')
                res = '||'.join([prompt] + res)
                d.append(res)
            business_urls = '\t'.join(d)

            url = random.choice(servers)
            abstract_urls = '\t'.join(run(abstract, url, str(MAX_NUM - 2 * len(business))))

            row_data = dict(zip(out_head, row.tolist() + [abstract_urls, business_urls]))
            out_data.append(row_data)
            #break
        except Exception as e:
            traceback.print_exc()
            row_data = dict(zip(out_head, row.tolist() + ['', '']))
            out_data.append(row_data)
            continue
    df = pandas.DataFrame(out_data)
    save_name = '生成图-' + os.path.basename(infile)
    writer = pandas.ExcelWriter(os.path.join(save_path, save_name), engine='xlsxwriter', engine_kwargs={'options': {'strings_to_urls': False}})
    df.to_excel(writer, index=False, encoding='utf-8')
    writer.close()
