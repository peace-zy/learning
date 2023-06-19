#-*-coding:utf-8-*-
"""
本文件实现了入库前-crop种子图评估功能。

"""

import streamlit as st
import logging
import os
import hashlib
import traceback
import json
import sys
import base64
import datetime

sys.path.append('../')
import util

st.set_page_config(page_title="入库前-crop候选种子图评估", layout='wide')
st.sidebar.header("入库前-crop候选种子图评估")
st.markdown("## 入库前-crop候选种子图评估")

SAVE_SEARCH = 'search_data'
CROP_DEEPBLUE_SERVER_URLS = ["http://10.255.120.17:8598/DeepBlueQueryService/query"]
CROP_DEEPBLUE_SERVER_URLS = ["http://10.255.120.17:8560/DeepBlueQueryService/query"]
TABLE_NAME = 'crop_person_fingerprints'

def save_b64_search(image_binary, merge_result):
    """存储检索结果base64 + search"""
    b64data = base64.b64encode(image_binary)
    util.gen_html_for_deepblue_res.make_path(SAVE_SEARCH)
    today = datetime.date.today()
    save_file = os.path.join(SAVE_SEARCH, '{}_b64_search_res.txt'.format(today))
    b64_res = '{}\t{}\n'.format(b64data.decode('utf-8'), json.dumps(merge_result))
    with open(save_file, 'a+') as f:
        f.write(b64_res)

def save_url_search(url, merge_result):
    """存储检索结果url + search"""
    util.gen_html_for_deepblue_res.make_path(SAVE_SEARCH)
    today = datetime.date.today()
    save_file = os.path.join(SAVE_SEARCH, '{}_url_search_res.txt'.format(today))
    url_res = '{}\t{}\n'.format(url, json.dumps(merge_result))
    with open(save_file, 'a+') as f:
        f.write(url_res)

def crop_fingerprint_proposal_eval():
    """crop候选种子图评估"""
    image_binary, image_input = util.get_image()
    topn = st.sidebar.text_input("topn", placeholder="默认top200，可自行设置最大500")
    if not topn:
        topn = 200
    else:
        topn = int(topn)

    #insert_baidu_shitu = st.sidebar.selectbox(label='是否引入百度识图', options=("否", "是"))
    insert_baidu_shitu = "否"
    is_run = st.sidebar.button("点击运行")

    if image_binary is None:
        return

    md5name = hashlib.md5(image_binary).hexdigest()

    if is_run:
        with st.spinner("Wait for a moment ...."):
            s_fea_info = []
            s_feas = []

            if insert_baidu_shitu == '是':
                search_urls = util.baidu_shitu_crawler.request_baidu_shitu(query=image_binary,
                                                                      CONT_TYPE='BIN', PAGENUM=4)
                #logging.info('search_urls: {}'.format(search_urls))

                s_image_tag = util.get_qianfan_tag(search_urls, "URL", util.TAGPATHS)
                if s_image_tag is not None:
                    _, s_query_fea = util.parse_tag_to_get_fea(s_image_tag)
                    for q_fea in s_query_fea:
                        s_fea_info.append({'url': q_fea['url'], 'thres': 0})
                        s_feas.append(q_fea['fea'])

            image_tag = util.get_image_tag(image_binary, util.TAGPATHS)
            if image_tag is None:
                st.error("Get feature failed!💀")
            else:
                #st.json(image_tag)
                ann_fea, query_fea = util.parse_tag_to_get_fea(image_tag)
                if ann_fea:
                    #if isinstance(image_input, str) and image_input.startswith('http'):
                    #    with open('query_fea.txt', 'a') as f:
                    #        f.write('{}\t{}\n'.format(image_input, ann_fea[0]['fea']))
                    distance_sim_res = []
                    if s_feas:
                        _, distance_sim_res = util.cal_sim_with_online_zhongzi.cal_sim(s_fea_info, s_feas, query_fea[0]['fea'])
                    #logging.info('search_res: {}'.format(distance_sim_res))
                    save_name = md5name + '.html'
                    search_result = util.request_dss(contents=ann_fea, deepblue_server_urls=CROP_DEEPBLUE_SERVER_URLS, table_name=TABLE_NAME)

                    if search_result:
                        merge_result = search_result[TABLE_NAME]
                        for idx in range(len(merge_result)):
                            merge_result[idx]['des'] = '<font color="red">送审数据</font>'
                        args = {'save_path': util.SAVE_PATH,
                                'save_name': save_name,
                                'pre_load_html_file': util.PRE_LOAD_HTML_FILE,
                                'post_load_html_file': util.POST_LOAD_HTML_FILE,
                                'cols': 4,
                                'url_prefix': None}

                        if distance_sim_res or len(CROP_DEEPBLUE_SERVER_URLS) > 1:
                            merge_result.extend(distance_sim_res)
                            merge_result = sorted(merge_result, key=lambda x:x['distance'])
                            args['des_schema'] = ['score', '数据源']
                        merge_result = merge_result[:topn]
                        args['search_result'] = merge_result

                        if isinstance(image_input, str) and image_input.startswith('http'):
                            save_url_search(image_input, merge_result)
                        try:
                            out_html_info = util.gen_html_for_deepblue_res.read_deepblue_res2html(args)
                        except Exception as e:
                            traceback.print_exc()
                            st.error("Run show failed!")
                    else:
                        st.error("Run retrieve failed!")
                    #st.markdown(''.join(out_html_info), unsafe_allow_html=True)
                    #from streamlit.compoents.v1 import html
                    #html(''.join(out_html_info))
                    st.components.v1.html(''.join(out_html_info), height=15000)

crop_fingerprint_proposal_eval()
