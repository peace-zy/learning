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

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="input file")
    parser.add_argument("--pre_url", type=str, default="http://10.255.120.17:8080/zhangyan", help="pre_url")
    parser.add_argument("--show_image_path", type=str, help="path to shown image")
    parser.add_argument("--region_path", type=str, help="path to region")
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

def txt2html(args):
    """txt2html"""
    txt_file = args.infile
    html_file = os.path.basename(txt_file).split('.')[0] + '.html'
    rm_file(html_file)

    js_part = '''
<body>
    <div id="div_top" style="top:0px;left:50px;width:100%;height: 30px;position:absolute;">
        <button onclick="Count()">汇总</button>
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
        var num_old = 0
        var num_new = 0
        var num_same = 0
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
                        if (radios[k].value == 'Old') {
                            num_old += 1
                        }
                        else if (radios[k].value == 'New') {
                            num_new += 1
                        }
                        else if (radios[k].value == 'Same') {
                            num_same += 1
                        }
                    }
                }
            }
        }
        alert("图像数量=" + num_image + ", 文本框数量=" + num_region + ", 旧版本好数量=" + num_old + ", 新版本好数量=" + num_new + ", 相似数量=" + num_same + ", 未标注数量=" + (num_region - num_old - num_new - num_same) + ", 旧版本好占比=" + Number(100 * num_old / num_region).toFixed(2) + "%" + ", 新版本好占比=" + Number(100 * num_new / num_region).toFixed(2) + "%")

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

    save_path = args.save_path
    save_image_path = os.path.join(save_path, args.show_image_path)
    make_path(save_image_path)
    save_region_path = os.path.join(save_path, args.region_path)
    make_path(save_region_path)

    url_local = []
    show_info = []
    with open(txt_file, 'r') as f:
        for line in f:
            fields = line.rstrip().split('\t')
            try:
                url = fields[0]
                response = requests.get(url)
                cont = np.array(bytearray(response.content), dtype=np.uint8)
                fname = hashlib.md5(cont).hexdigest()
                image_name = fname + '.jpg'
                image = cv2.imdecode(cont, cv2.IMREAD_COLOR)
                if image is None:
                    print('{} cannot read'.format(line))
                    continue
                ori_image = copy.deepcopy(image)

                res = json.loads(fields[1])['res']
                regions = []
                for idx, info in enumerate(res):
                    old_ocr = info['word_2']
                    new_ocr = info['word_1']
                    old_rect = info['rect_2']
                    new_rect = info['rect_1']
                    if old_rect:
                        xmin, ymin, xmax, ymax = old_rect[:]
                    elif new_rect:
                        xmin, ymin, xmax, ymax = new_rect[:]
                    else:
                        raise ValueError

                    region = ori_image[ymin:ymax, xmin:xmax]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, 4)
                    cv2.putText(image, str(idx), (xmin + 2, ymin), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    region_name = '{}_{}.jpg'.format(fname, idx)
                    region_file = os.path.join(save_region_path, region_name)
                    cv2.imwrite(region_file, region)
                    gen_region_url = os.path.join(args.pre_url, args.region_path, region_name)
                    regions.append({'region_url': gen_region_url, 'new_ocr': new_ocr, 'old_ocr': old_ocr})

                image_file = os.path.join(save_image_path, image_name)
                url_local.append('{}\t{}\n'.format(url, image_file))
                cv2.imwrite(image_file, image)
                gen_image_url = os.path.join(args.pre_url, args.show_image_path, image_name)
                show_info.append({'image_url': gen_image_url, 'regions': regions})

            except Exception as e:
                print('line parse errors')
                traceback.print_exc()
                print(line)
                continue
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
                        <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Old" onclick="update(this)" />旧</label>
                        <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                       <input type="radio" display:block name="result{}" id="optionsRadios{}" value="New" onclick="update(this)" />新</label>
                        <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                       <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Same" onclick="update(this)" />一样</label>""".\
                       format(id_r, c, id_r, c + 1, id_r, c + 2)
            c += 3
            id_r += 3

            gen_region_url = r['region_url']
            # regions.append({'region_url': gen_region_url, 'new_ocr': new_ocr, 'old_ocr': old_ocr})
            old_ocr = r['old_ocr']
            new_ocr = r['new_ocr']

            #wl += '<td><a href="{}" src="{}"> <img src="{}" width=185 border=1 controls></a>\n{}\n\n{}</td>'\
            #        .format(gen_region_url, gen_region_url, gen_region_url, info, select)
            #wl += '<td style="word-break:break-all">{}\n<a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a>\n{}</td>'\
            #        .format(info, gen_region_url, gen_region_url, gen_region_url, select)
            wl += '<td>框序号: {}<a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a><br />Old: {} <br />New: {}<br />{}</td>'\
                    .format(idx, gen_region_url, gen_region_url, gen_region_url, old_ocr.encode('utf-8'), new_ocr.encode('utf-8'), select)


        wl += '</tr>\n'
        wlines.append(wl)

    print ('num_images={}, num_regions={}'.format(len(wlines), num_region))
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
