# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
import logging
import traceback
import string
import re
import streamlit as st
from hashlib import md5

# Set your own appid/appkey.
appid = '20230517001681356'
appkey = 'KU6l_KLBOHHrGQMLzwpx'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

#query = 'Hello World! This is 1st paragraph.\nThis is 2nd paragraph.'

en_re = re.compile(r'[A-Za-z]',re.S)

# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    """make_md5"""
    return md5(s.encode(encoding)).hexdigest()

def is_all_chinese(text):
    """is_all_chinese"""
    for char in text:
        if not '\u4e00' <= char <= '\u9fa5':
            return False
    return True

def contains_chinese(text):
    """contais_chinese"""
    for char in text:
        if '\u4e00' <= char <= '\u9fa5':
            return True
    return False

def is_all_english(text):
    """is_all_english"""
    for char in text:
        if char not in string.ascii_lowercase + string.ascii_uppercase:
            return False
    return True

def contains_english(text):
    """contains_english"""
    res = re.findall(en_re, text)
    if len(res):
        return False
    else:
        return True

def trans_en_to_zh(query):
    """trans_en_to_zh"""
    from_lang = 'en'
    to_lang =  'zh'
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request

    """
    {
        "from": "en",
        "to": "zh",
        "trans_result": [
            {
                "src": "Hello World! This is 1st paragraph.",
                "dst": "你好，世界！这是第一段。"
            },
            {
                "src": "This is 2nd paragraph.",
                "dst": "这是第2段。"
            }
        ]
    }
    """

    result = ""
    TRY_NUM = 1
    try:
        while TRY_NUM:
            response = requests.post(url,
                                      headers=headers,
                                      params=payload,
                                      timeout=50)
            if response.status_code == requests.codes.ok:
                result = response.json()['trans_result'][0]['dst']
                break
            TRY_NUM -= 1
    except Exception as e:
        traceback.print_exc()
        logging.error("trans post error! {}".format(e))
    if TRY_NUM == 0 and not result:
        logging.info("trans post error! response code: {response.status_code}, response text: {response.text}")

    return result

def trans_zh_to_en(query):
    """trans_zh_to_en"""
    from_lang = 'zh'
    to_lang =  'en'
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request

    result = ""
    TRY_NUM = 1
    try:
        while TRY_NUM:
            response = requests.post(url,
                                      headers=headers,
                                      params=payload,
                                      timeout=50)
            if response.status_code == requests.codes.ok:
                result = response.json()['trans_result'][0]['dst']
                break
            TRY_NUM -= 1
    except Exception as e:
        traceback.print_exc()
        logging.error("trans post error! {}".format(e))
    if TRY_NUM == 0 and not result:
        logging.info("trans post error! response code: {response.status_code}, response text: {response.text}")

    return result


#if __name__ == '__main__':
#    query = 'Hello World!'
#    trans_en_to_zh(query)
