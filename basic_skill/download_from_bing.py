import requests
from bs4 import BeautifulSoup
import json
import eventlet
import os

urlshu = 1      #url中first = urlshu
pictureshu = 1  #图片下载时的名字（加上异常图片的第几张图片）
soupshu = 0     #每35张soup列表中第soupshu个
whileshu = 35   #用于while循环的数（因为每个页面35张图片）



url1 = 'https://cn.bing.com/images/async?q='
url2 = '&first=%d&count=35&cw=1177&ch=737&relp=35&tsc=ImageBasicHover&datsrc=I&layout=RowBased&mmasync=1'
head1 = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.64'
    }
#有35张图片的网页的请求标头

head2 = {
'Cookie': 'Hm_lvt_d60c24a3d320c44bcd724270bc61f703=1624084633,1624111635',
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.64'
    }
    
#具体图片的网页的请求标头

print('请输入查找内容:')
#content = input()
content = "泄露商业机密"
print('请输入查找图片数量:')
#number = int(input())
number = 500   
url = url1 + content + url2 #合并成去往有35张图片的网址
root = os.getcwd()
path = root  + '/download_image/'+content+'/'
if not os.path.exists(path):
    os.makedirs(path)
while whileshu:
    r0 = requests.get(url%urlshu,headers = head1).text  #发送get请求并将内容转化成文本
    soup = BeautifulSoup(r0,features="lxml").find_all('a','iusc')   #解析r0并找到a和class=iusc的标签
    data = str(soup[soupshu].get('m'))  #将列表soup的第soupshu位中的m提取出来
    zidian = json.loads(data)   #将data转化成字典形式
    ifopen = 1      #用于下方判断是否下载图片
    with eventlet.Timeout(1,False):     #设定1秒的超时判断
        try:
            picture = requests.get(zidian['murl'],headers = head2).content  #发送get请求并返回二进制数据
        except:
            print('图片%d超时异常'%pictureshu)    #说明图片异常
            ifopen = 0      #取消下载此图片，否则会一直卡着然后报错
    while ifopen == 1:
        text = open(path +'/'+ '%d'%pictureshu + '.jpg','wb')  #将图片下载至文件夹'图片'中
        text.write(picture)     #上行代码中'wb'为写入二进制数据
        text.close()
        ifopen = 0
        number = number - 1
        
    pictureshu = pictureshu + 1
    soupshu = soupshu + 1
    whileshu = whileshu - 1
    print(number)
    if whileshu == 0:   #第一页图片下载完成，将网址的first进1
        urlshu = urlshu + 1
        whileshu = 35
        soupshu = 0
        
    if number == 0:     #下载图片数量达标，退出循环
        break
