#-*-coding: utf-8 -*-
"""
Desc  :   html
"""

import sys
import os
import traceback
import argparse
import requests
import copy

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file", type=str, help="input txt file")
    parser.add_argument("--attach_file", type=str, default=None, help="input txt addtion file")
    parser.add_argument("--pre_load_html_file", type=str, default="show_image_with_note.html",
                                                help="local file to show")
    parser.add_argument("--url_prefix", type=str, default=None, help="local file to show")
    parser.add_argument("--cols", type=int, default=6, help="column number of table")
    parser.add_argument("--save_path", type=str, default='out_html', help="path to save generated html file")
    parser.add_argument("--save_name", type=str, default=None, help="name of generated html file")
    return parser.parse_args()

def make_path(path):
    """make_path"""
    if not os.path.exists(path):
        os.makedirs(path)

def rm_file(f):
    """remove file"""
    if os.path.exists(f):
        os.remove(f)

class HtmlGenerator(object):
    """HtmlGenerator"""
    def __init__(self, pre_load_html_file='show_image_with_note.html'):
        self.pre_load_html_file = pre_load_html_file
        self.out_html_info = self.get_pre_load_html()
        self.addition = None
        self.attach = None

        # image
        self.image_field = '<a href="{}" src="{}"> <img src="{}" width=216 border=1 controls></a>'

        # raido default 3
        self.radio = \
                    """<label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                    <input type="radio" display:block name="result{}" id="radio_0" value="Right" onclick="update_radio(this)" />对</label>
                    <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                   <input type="radio" display:block name="result{}" id="radio_1" value="Wrong" onclick="update_radio(this)" />错</label>
                    <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                   <input type="radio" display:block name="result{}" id="radio_2" value="Hold" onclick="update_radio(this)" />待定</label>"""

        # tabel head
        self.table_head_field = '<th>{}</th>\n'
        # row field
        self.row_field = '<tr>{}</tr>\n'
        # cell field
        self.cell_field = '<td>{}</td>\n'
        # blank row-des-blank row
        self.des = '<br />{}<br />'
        self.text = '<br/><input type="text" name="text_in" size="15" maxlength="50" value="备注">' \
                    '<button onclick="update_text_with_button(this)">保存</button>'

    def get_pre_load_html(self):
        """get_pre_load_html"""
        if not os.path.exists(self.pre_load_html_file):
            raise 'FileNotFoundError: no such file or directory {}'.format(self.pre_load_html_file)
        return open(self.pre_load_html_file, 'r').readlines()

    def add_image(self, image_url, des='', radio=()):
        """add_image"""
        if isinstance(image_url, str):
            add_info = self.image_field.format(image_url, image_url, image_url)
        else:
            add_info = self.image_field.format(*image_url)
        if des:
            add_info += self.des.format(des)
        if radio:
            add_info += self.radio.format(*radio)
        add_info += self.text
        return add_info

    def read_file(self, txt_file, delimiter='\t'):
        """read_file"""
        if not os.path.exists(txt_file):
            raise 'FileNotFoundError: no such file or directory {}'.format(txt_file)
        with open(txt_file, 'r') as f:
            for line in f:
                fields = line.rstrip().split(delimiter)
                yield fields

    def show_image_with_note(self, args, add_radio=False, table_head=['query_image']):
        """show_image_with_note"""
        out_html_info = copy.deepcopy(self.out_html_info)
        out_html_info.append('<table border=1 id="show_table" '
                             'style="top:30px;left:0px;position:absolute;word-break:break-all">\n')
        # create table head
        out_html_info.append(''.join([self.table_head_field.format(h) for h in table_head]))
        id_r = 1
        cnt = 0
        cells = []
        for fields in self.read_file(args.txt_file):
            image_url, addition = fields[0], fields[1:]

            if cells and len(cells) % args.cols == 0:
                out_html_info.append(self.row_field.format(''.join(cells)))
                cells = []
                id_r += 1

            radio = []
            if add_radio:
                radio = [cnt] * 3
            cnt += 1

            des = 'image'
            if self.attach is not None and image_url in self.attach:
                des = self.attach[image_url]
            elif addition:
                des = 'Des: {}'.format(' '.join(addition))
            cell_item = self.add_image(image_url, des=des, radio=radio)
            cells.append(self.cell_field.format(cell_item))
        if cells:
            out_html_info.append(self.row_field.format(''.join(cells)))
        out_html_info.append('\n</table>')

        return out_html_info


def image_in_txt2html(args):
    """txt2html"""
    txt_file = args.txt_file
    save_path = args.save_path
    make_path(save_path)
    save_name = args.save_name
    if save_name is None:
        save_name = os.path.basename(txt_file).split('.')[0] + '.html'
    html_file = os.path.join(save_path, save_name)
    rm_file(html_file)
    print(html_file)

    attach = {}
    if args.attach_file is not None:
        with open(args.attach_file, 'r') as f:
            for line in f:
                fields = line.rstrip().split('\t')
                info = 'image'
                if len(fields) == 1:
                    url = fields[0]
                elif len(fields) > 1:
                    url, info = fields[0], ' '.join(fields[1:0])
                attach[url] = info

    html_generator = HtmlGenerator(pre_load_html_file=args.pre_load_html_file)
    if attach:
        html_generator.attach = attach
    query_html_info = html_generator.show_image_with_note(args, add_radio=True, table_head=['query_image'])

    with open(html_file, 'w') as file:
        file.writelines(query_html_info)

def main():
    """main"""
    """python gen_html_for_image_url.py [infile]"""
    args = parse_args()
    image_in_txt2html(args)
    return

if __name__ == "__main__":
    main()
