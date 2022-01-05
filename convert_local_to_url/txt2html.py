#-*-coding: utf-8 -*-
"""
Desc  : 通过html显示图像
refs：https://bbs.csdn.net/topics/210078734
      https://www.cnblogs.com/ljzc002/p/12048895.html
"""

import sys
import os
import traceback
import argparse

def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, help="input file")
    parser.add_argument("--pre_url", type=str, default="http://10.255.120.17:8080/zhangyan", help="pre_url")
    parser.add_argument("--show_image_path", type=str, help="path to shown image")
    parser.add_argument("--region_path", type=str, help="path to region")
    return parser.parse_args()

def txt2html(args):
    """txt2html"""
    txt_file = args.infile
    html_file = os.path.basename(txt_file).split('.')[0] + '.html'
    if os.path.exists(html_file):
        os.remove(html_file)

    js_part = '''
<body>
    <div id="div_top" style="top:0px;left:0px;width:100%;height: 30px;position:absolute;">
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
        var num_r = 0
        var num_w = 0
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
                            num_r += 1
                        }
                        else if (radios[k].value == 'Wrong') {
                            num_w += 1
                        }
                    }
                }
            }
        }
        alert("图像数量=" + num_image + ", 文本框数量=" + num_region + ", 正确数量=" + num_r + ", 错误数量=" + num_w + ", 未标注数量=" + (num_region - num_r - num_w) + ", 精确率=" + Number(100 * num_r / num_region).toFixed(2) + "%")

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

    show_dict = {}
    with open(txt_file, 'r') as f:
        for idx, line in enumerate(f):
            fields = line.rstrip().split('\t')
            try:
                lp_url = fields[0]
                pic_url = fields[1]
                bos_url = fields[2]
                region_url = fields[3]
                region_name = os.path.basename(region_url)
                image_name = region_name.split('_')[0] + '.jpg'

                gen_image_url = os.path.join(args.pre_url, args.show_image_path, image_name)
                gen_region_url = os.path.join(args.pre_url, args.region_path, region_name)
                #info = fields[4].split(',{')[0].decode('utf-8').encode('gbk')
                info = fields[4].split(',{')[0]
                pos = info.rfind(',')
                ab_type = fields[5]
                if gen_image_url not in show_dict:
                    show_dict[gen_image_url] = []
                show_dict[gen_image_url].append({'region_url': gen_region_url, 'info': info[:pos] + '||' + ab_type})

            except Exception as e:
                print('line parse errors')
                traceback.print_exc()
                print(line)
                continue
    num_region = 0
    id_r = 0
    for gen_image_url, regions in show_dict.items():
        wl = '<tr><td><a href="{}" src="{}"> '\
             '<img src="{}" width=216 border=1 controls></a></td>'.\
             format(gen_image_url, gen_image_url, gen_image_url)
        num_region += len(regions)
        c = 0
        for r in regions:
            select = """<label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: large;">
                        <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Right" onclick="update(this)" />Right</label>
                        <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: large;">
                       <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Wrong" onclick="update(this)" />Wrong</label>""".\
                       format(id_r, c, id_r, c + 1)
            c += 2
            id_r += 2

            gen_region_url = r['region_url']
            info = r['info']



            #wl += '<td><a href="{}" src="{}"> <img src="{}" width=185 border=1 controls></a>\n{}\n\n{}</td>'\
            #        .format(gen_region_url, gen_region_url, gen_region_url, info, select)
            #wl += '<td style="word-break:break-all">{}\n<a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a>\n{}</td>'\
            #        .format(info, gen_region_url, gen_region_url, gen_region_url, select)
            wl += '<td><a href="{}" src="{}"> <img src="{}" width=150 border=1 controls></a><br />{}<br />{}</td>'\
                    .format(gen_region_url, gen_region_url, gen_region_url, info, select)


        wl += '</tr>\n'
        wlines.append(wl)

    print ('num_images={}, num_regions={}'.format(len(wlines), num_region))
    wlines.append('\n</table>')
    with open(html_file, 'w') as file:
        file.writelines(wlines)

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
