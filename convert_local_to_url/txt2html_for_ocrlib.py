#-*-coding: utf-8 -*-
"""
Author:   zhangyan75@baidu.com
Date  :   21/12/08 10:34:34
Desc  :
"""

import sys
import os
import traceback
import argparse
import cv2
import copy
import numpy as np
import requests
import hashlib
import json
from tqdm import tqdm

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="input file")
    parser.add_argument("--pre_url", type=str, default="http://10.255.120.17:8080/zhangyan", help="pre_url")
    parser.add_argument("--show_image_path", type=str, help="path to shown show", default="show")
    parser.add_argument("--ori_image_path", type=str, help="path to ori image", default="ori")
    parser.add_argument("--region_path", type=str, help="path to region", default="region")
    parser.add_argument("--save_path", type=str, help="path to saved")
    return parser.parse_args()

def make_path(path):
    """创建路径"""
    if not os.path.exists(path):
        os.makedirs(path)

def rm_file(f):
    """删除文件"""
    if os.path.exists(f):
        os.remove(f)

def parse_ocrlib(args):
    txt_file = args.infile
    save_path = args.save_path
    save_show_image_path = os.path.join(save_path, args.show_image_path)
    make_path(save_show_image_path)
    save_ori_image_path = os.path.join(save_path, args.ori_image_path)
    make_path(save_ori_image_path)
    save_region_path = os.path.join(save_path, args.region_path)
    make_path(save_region_path)

    url_local = []
    show_info = []
    with open(txt_file, 'r') as f:
        for line in tqdm(f.readlines()):
            fields = line.rstrip().split('\t')
            if len(fields) < 1:
                continue
            try:
                lp_url, bos_url, pic_url, ocr_str = fields[0], fields[1], fields[2], fields[3]

                response = requests.get(bos_url)
                cont = np.array(bytearray(response.content), dtype=np.uint8)
                fname = hashlib.md5(cont).hexdigest()
                image_name = fname + '.jpg'
                image = cv2.imdecode(cont, cv2.IMREAD_COLOR)
                if image is None:
                    print('{} cannot read'.format(line))
                    continue
                ori_image = copy.deepcopy(image)
                ori_image_file = os.path.join(save_ori_image_path, image_name)
                cv2.imwrite(ori_image_file, ori_image)

                ocr = json.loads(ocr_str)

                originals = ocr['originals']
                details = ocr['details']

                regions = []
                gen_image_url = os.path.join(args.pre_url, args.show_image_path, image_name)
                for idx, ocr_text_item in enumerate(details):
                    if 'auxiliary' in ocr_text_item:
                        # processing for detail
                        ocr_text = ocr_text_item['text']
                        r_left = ocr_text_item['left']
                        r_top = ocr_text_item['top']
                        r_width = ocr_text_item['width']
                        r_height = ocr_text_item['height']

                        region = ori_image[r_top:r_top + r_height, r_left:r_left + r_width]
                        region_name = '{}_{}.jpg'.format(fname, idx)
                        region_file = os.path.join(save_region_path, region_name)
                        cv2.imwrite(region_file, region)
                        gen_region_url = os.path.join(args.pre_url, args.region_path, region_name)
                        regions.append({'region_url': gen_region_url, 'ocr': ocr_text})

                        # 绿色文本框
                        cv2.rectangle(image, (r_left, r_top), (r_left + r_width, r_top + r_height),  (0, 255, 0), 1, 4)
                        # 框id粉红色
                        cv2.putText(image, str(idx), (r_left + 2, r_top), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


                        # processing for original
                        cut_img_id = ocr_text_item['auxiliary']['cut_img_id']
                        ori = originals[cut_img_id]
                        paragraphs = ori['ocr']['paragraphs']
                        for graph in paragraphs:
                            min_poly_location = graph['min_finegrained_poly_location']
                            points = np.array([[p['x'], p['y'] + cut_img_id * ori['height']] for p in min_poly_location['points']], np.int32)
                            # 红色轮廓框
                            cv2.polylines(image, [points], True, (0, 0, 255))
                        ret = ori['ocr']['ret']
                        for r in ret:
                            poly_location = r['poly_location']
                            #{'points': [{'y': 434, 'x': 99}, {'y': 407, 'x': 99}, {'y': 407, 'x': 12}, {'y': 434, 'x': 12}]}
                            #points = np.array([[p['x'], p['y'] + cut_img_id * ori['height']] for p in poly_location['points']], np.int32)
                            #cv2.polylines(image, [points], True, (255, 128, 0))
                            charset = r['charset']
                            for c in charset:
                                #{'width': '17', 'top': '406', 'height': '29', 'left': '17'}
                                c_rect = c['rect']
                                x = int(c_rect['left'])
                                y = int(c_rect['top']) + cut_img_id * ori['height']
                                w = int(c_rect['width'])
                                h = int(c_rect['height'])
                                # 字符框蓝色
                                #cv2.rectangle(image, (x, y), (x + w, y + h),  (255, 0, 0), 1, 4)
                                #cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 255, 255), 1, 4)

                show_info.append({'image_url': gen_image_url, 'regions': regions})
                show_image_file = os.path.join(save_show_image_path, image_name)
                cv2.imwrite(show_image_file, image)
                url_local.append('\t'.join([lp_url, bos_url, pic_url, ori_image_file, show_image_file]) + '\n')

            except Exception as e:
                print('line parse errors')
                traceback.print_exc()
                #print(line)
                continue
    return (url_local, show_info)

def txt2html(args):
    """txt2html"""
    txt_file = args.infile
    html_file = os.path.basename(txt_file).split('.')[0] + '.html'
    rm_file(html_file)

    js_part = '''
<body>
    <div id="div_top" style="top:0px;left:50px;width:100%;height: 30px;position:absolute;">
        <button onclick="Count()">汇总</button>
        <button onclick="ExportData()">导出数据</button>
        <button onclick="ExportHtml()">导出Html</button>
    </div>
</body>

<!--jspart-->
<script language="javascript" type="text/javascript">
    function update(sender) {
        console.log(sender);
        sender.defaultChecked = !sender.defaultChecked;
        sender.checked = sender.defaultChecked;
    }

    function ExportHtml()
    {
        var str=window.document.body.outerHTML+"</html>";
        var blob=new Blob([str],{
            type: "text/plain"
        })

        var tmpa = document.createElement("a");
        //var p_h1=document.getElementsByClassName("p_h1")[0];
        //console.log(p_h1)
        var parts = window.location.href.split('/')
        tmpa.download = parts[parts.length - 1]
        //tmpa.download = (p_h1?p_h1.innerHTML:"test")+".html";
        tmpa.href = URL.createObjectURL(blob);
        tmpa.click();//导出后事件需要重新绑定，或者直接使用innHTML定义？
        setTimeout(function () {
            URL.revokeObjectURL(blob);
        }, 100);
    }
    function Count() {
        var num_image = 0
        var num_region = 0
        var num_right = 0
        var num_wrong = 0
        var num_hold = 0
        var show_table =document.getElementById("show_table");
        //console.log(show_table.rows)
        //console.log(show_table.rows.length)
        for (var i = 1; i < show_table.rows.length; ++i) {
            num_image += 1
            var cells = show_table.rows[i].cells
            for (var j = 1; j < cells.length; ++j) {
                num_region += 1
                //console.log(cells[j].innerHTML)
                //radios = cells[j].getElementsByClassName('radio-inline')
                //for (var k = 0; k < radios.length; ++k) {
                //    console.log(radios[k].getElementsByTagName('input').length)
                //    console.log(radios[k].getElementsByTagName('input')[0].checked)
                //}
                radios = cells[j].getElementsByTagName('input')
                for (var k = 0; k < radios.length; ++k) {
                    //console.log(radios[k].checked)
                    //console.log(radios[k].value)
                    if (radios[k].checked) {
                        if (radios[k].value == 'Right') {
                            num_right += 1
                        }
                        else if (radios[k].value == 'Wrong') {
                            num_wrong += 1
                        }
                        else if (radios[k].value == 'Hold') {
                            num_hold += 1
                        }
                    }
                }
            }
        }
        alert("图像数量=" + num_image + ", 文本框数量=" + num_region + ", 对的数量=" + num_right + ", 错的数量=" + num_wrong + ", 待定数量=" + num_hold + ", 未标注数量=" + (num_region - num_right - num_wrong - num_hold) + ", 精确率=" + Number(100 * num_right / num_region).toFixed(2) + "%")

    }

    function download(filename, text) {
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
        element.setAttribute('download', filename);
        element.click();
    }

    function ExportData() {
        var right_line = ""
        var wrong_line = ""
        var show_table =document.getElementById("show_table");
        //console.log(show_table.rows)
        //console.log(show_table.rows.length)
        for (var i = 1; i < show_table.rows.length; ++i) {
            var cells = show_table.rows[i].cells
            for (var j = 1; j < cells.length; ++j) {
                //console.log(cells[j].innerHTML)
                //radios = cells[j].getElementsByClassName('radio-inline')
                //for (var k = 0; k < radios.length; ++k) {
                //    console.log(radios[k].getElementsByTagName('input').length)
                //    console.log(radios[k].getElementsByTagName('input')[0].checked)
                //}
                img_src = cells[j].getElementsByTagName("img")[0].src
                radios = cells[j].getElementsByTagName('input')
                for (var k = 0; k < radios.length; ++k) {
                    //console.log(radios[k].checked)
                    //console.log(radios[k].value)
                    if (radios[k].checked) {
                        if (radios[k].value == 'Right') {
                            right_line += img_src + "\n"
                        }
                        else if (radios[k].value == 'Wrong') {
                            wrong_line += img_src + "\n"
                        }
                    }
                }
            }
        }
        console.log("right")
        download('right.txt', right_line)
        console.log("wrong")
        download('wrong.txt', wrong_line)
        alert("导出数据成功")

    }

    //刷新存储
    //window.onload = function () {
    //    ExportHtml()
    //}

</script>
'''

    wlines = []
    wlines.append(js_part)
    wlines.append('<table border=1 id="show_table" style="top:30px;left:0px;position:absolute;word-break:break-all">\n')
    wlines.append('<tr><td>show image</td><td >region</td></tr>\n')

    (url_local, show_info) = parse_ocrlib(args)
    num_region = 0
    id_r = 0
    for info in show_info:
        gen_image_url = info['image_url']
        wl = '<tr><td><a href="{}" src="{}"> '\
             '<img src="{}" width=216 border=1 controls></a></td>'.\
             format(gen_image_url, gen_image_url, gen_image_url)
        regions = info['regions']

        num_region += len(regions)
        c = 0
        for idx, r in enumerate(regions):
            #select = """<label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: large;">
            select = """<label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                        <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Right" onclick="update(this)" />对</label>
                        <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                       <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Wrong" onclick="update(this)" />错</label>
                        <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                       <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Hold" onclick="update(this)" />待定</label>""".\
                       format(id_r, c, id_r, c + 1, id_r, c + 2)
            c += 3
            id_r += 3

            gen_region_url = r['region_url']
            ocr_text = r['ocr']

            #wl += '<td><a href="{}" src="{}"> <img src="{}" width=185 border=1 controls></a>\n{}\n\n{}</td>'\
            #        .format(gen_region_url, gen_region_url, gen_region_url, info, select)
            #wl += '<td style="word-break:break-all">{}\n<a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a>\n{}</td>'\
            #        .format(info, gen_region_url, gen_region_url, gen_region_url, select)
            wl += '<td>框序号: {}<a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a><br />OCR: {}<br />{}</td>'\
                    .format(idx, gen_region_url, gen_region_url, gen_region_url, ocr_text.encode('utf-8'), select)


        wl += '</tr>\n'
        wlines.append(wl)

    print ('num_images={}, num_regions={}'.format(len(show_info), num_region))
    wlines.append('\n</table>')
    with open(html_file, 'w') as file:
        file.writelines(wlines)
    with open('url_local_map.txt', 'w') as file:
        file.writelines(url_local)

def main():
    """main"""
    """python txt2html.py \
              --infile rm_small_size_6.txt \
              --pre_url http://10.255.120.17:8080/zhangyan/rm_small_size_6_new \
              --show_image_path images \
              --region_path ocr_region"""
    args = parse_args()
    txt2html(args)
    return

if __name__ == "__main__":
    main()
