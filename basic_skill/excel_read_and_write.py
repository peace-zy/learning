import pandas as pd
import sys
from tqdm import tqdm

def get_data(infile, sheet_name, usecols=None):
    sheet = pd.read_excel(infile, sheet_name=sheet_name, usecols=usecols)
    rows, cols = sheet.shape
    head = sheet.columns.to_list()
    for i in tqdm(range(rows)):
        #url, risk1, risk2, risk3 = sheet.iloc[i, 0:4]
        #yield (url, risk1, risk2, risk3)
        #data = sheet.iloc[i, 0:len(usecols.split(","))]
        data = sheet.iloc[i, :]
        yield (head, data)

def main():
    infile = sys.argv[1]
    #usecols = "A,B,C,D,E,F,G"
    usecols = None
    FIRST_VALID_ROW = True
    FIRST_NONVALID_ROW = True
    start_valid = 0
    start_nonvalid = 0
    with pd.ExcelWriter("output_valid.xlsx") as valid_writer:
        with pd.ExcelWriter("output_nonvalid.xlsx") as nonvalid_writer:
            for head, data in get_data(infile, sheet_name=0, usecols=usecols):
                valid_cell_num = 0
                for i in range(len(data)):
                    if pd.notnull(data.iloc[i]):
                        valid_cell_num += 1

                out_data = pd.DataFrame([data.values.tolist()])
                if valid_cell_num == 7:
                    if FIRST_VALID_ROW:
                        out_data.to_excel(valid_writer, index=False, header=head, startrow=start_valid)
                        #out_data.to_excel("output_valid.xlsx", mode="a", index=False, header=head)
                        #out_data.to_excel(writer)
                        FIRST_VALID_ROW = False
                        start_valid += 1
                    else:
                        out_data.to_excel(valid_writer, index=False, header=False, startrow=start_valid)
                        #out_data.to_excel("output_valid.xlsx", mode="a", index=False, header=False)
                    start_valid += 1
                else:
                    if FIRST_NONVALID_ROW:
                        out_data.to_excel(nonvalid_writer, index=False, header=head, startrow=start_nonvalid)
                        FIRST_NONVALID_ROW = False
                        start_nonvalid += 1
                    else:
                        out_data.to_excel(nonvalid_writer, index=False, header=False, startrow=start_nonvalid)
                    start_nonvalid += 1
                    #out_data.to_excel("output_nonvalid.xlsx", mode="a", index=False, header=False)

    return

if __name__ == "__main__":
    main()
