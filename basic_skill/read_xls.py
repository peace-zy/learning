#-*- coding: utf-8 -*-
import pandas as pd
import sys
import os

def get_data(infile, sheet_name, usecols):
    sheet = pd.read_excel(infile, sheet_name=sheet_name, usecols=usecols)
    print(sheet)
    rows, cols = sheet.shape
    for i in range(rows):
        url, risk1, risk2, risk3 = sheet.iloc[i, 0:4]
        yield (url, risk1, risk2, risk3)


def parse_online_vulgar_badcase(infile):
    record = {}
    sheet_name = 0
    usecols = [3, 19, 20, 21]
    usecols = 'D,T,U,V'
    for url, risk1, risk2, risk3 in get_data(infile, sheet_name, usecols):
        #print (url, risk1, risk2, risk3)
        #print (type(url), type(risk1), type(risk2), type(risk3))
        if isinstance(risk1, float):
            continue
        if isinstance(risk2, float):
            risk2 = ''
        if isinstance(risk3, float):
            risk3 = ''
        if risk1 not in record:
            record[risk1] = []
        record[risk1].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                    (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                    (('|' + risk3.encode('utf-8')) if risk3 else '')})

        if risk2:
            if risk2 not in record:
                record[risk2] = []
            record[risk2].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                        (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                        (('|' + risk3.encode('utf-8')) if risk3 else '')})
        if risk3:
            if risk3 not in record:
                record[risk3] = []
            record[risk3].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                        (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                        (('|' + risk3.encode('utf-8')) if risk3 else '')})

    print(len(record.keys()))
    for k in record.keys():
        print(k.encode('utf-8'))
    return record

def parse_vulgar_database(infile):
    record = {}
    sheet_name = 1
    usecols = 'B,R,S,T'
    for url, risk1, risk2, risk3 in get_data(infile, sheet_name, usecols):
        #print (url, risk1, risk2, risk3)
        #print (type(url), type(risk1), type(risk2), type(risk3))
        if isinstance(risk1, float):
            continue
        if isinstance(risk2, float):
            risk2 = ''
        if isinstance(risk3, float):
            risk3 = ''
        if risk1 not in record:
            record[risk1] = []
        record[risk1].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                    (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                    (('|' + risk3.encode('utf-8')) if risk3 else '')})

        if risk2:
            if risk2 not in record:
                record[risk2] = []
            record[risk2].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                        (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                        (('|' + risk3.encode('utf-8')) if risk3 else '')})
        if risk3:
            if risk3 not in record:
                record[risk3] = []
            record[risk3].append({'url': url, 'risk': risk1.encode('utf-8') + \
                                                        (('|' + risk2.encode('utf-8')) if risk2 else '') + \
                                                        (('|' + risk3.encode('utf-8')) if risk3 else '')})

    print(len(record.keys()))
    for k in record.keys():
        print(k.encode('utf-8'))
    return record


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_out(data, save_path):
    make_path(save_path)
    for risk, detail in data.items():
        risk = risk.encode('utf-8')
        savefile = os.path.join(save_path, risk + '.txt')
        with open(savefile, 'w') as f:
            for d in detail:
                print(d['url'])
                print(d['risk'])
                f.write('{}\t{}\n'.format(d['url'], d['risk']))

def main():
    infile = sys.argv[1]
    save_path = sys.argv[2]
    record = parse_vulgar_database(infile)
    print(type(record))
    save_out(record, save_path)

    return

if __name__ == '__main__':
    main()
