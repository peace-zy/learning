
from flask import Flask, request, jsonify
import os
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import traceback
from log import Logger
#torch.manual_seed(1234)
app = Flask(__name__)

logger = Logger(os.path.basename(__file__)).logger

class VLCHAT(object):
    def __init__(self, model_path='output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1'):
        if not os.path.exists(model_path):
            logger.error("model path error [{}]".format(model_path))
            raise FileNotFoundError
        self.model_path = model_path

    def init(self):
        logger.info("model_path=[{}]".format(self.model_path))
        logger.info("load tokenizer")
        # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        #tokenizer.padding_side = 'left'

        # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # 使用CPU进行推理，需要约32GB内存
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # 默认gpu进行推理，需要约24GB显存
        logger.info("load model")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="cuda", trust_remote_code=True).eval()

        # 可指定不同的生成长度、top_p等相关超参（transformers 4.32.0及以上无需执行此操作）
        # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        self.model.generation_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.generation_config.top_p = 0.01

    def chat(self, in_query):
        # 第一轮对话
        """
        query = tokenizer.from_list_format([
            {'image': '/mnt/aigc_chubao/liyulong/data/v1227/img_single_svg_2/11000014889168.png'}, # Either a local path or an url
            {'text': "用中文讲解下这个户型"},
        """
        in_query = [{k: v} for k, v in in_query.items()] 
        query = self.tokenizer.from_list_format(in_query)

        logger.info("query={}".format(query))
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        #response = model.chat_batch(tokenizer, query=query, history=None)
        logger.info("response={}".format(response))
        return response
 
model_path = "/aistudio/workspace/qwen_train/Qwen-VL/output_qwen_sekuai_des_align_zero2_a800_vit_on_cosine_2048_ep1"
vlchat = VLCHAT(model_path)
vlchat.init()
 
@app.route('/predict', methods=['POST'])
def predict():
    # 在这里编写处理HTTP GET请求的逻辑
    try:
        params = request.get_json()
        request_id = params["request_id"]
        result = {"request_id": request_id, "error_code": 0, "error_info": "sucess", "data": {}}
        logger.info(params)
    except Exception as e:
        #traceback.print_exc(e)
        info = {"error_code": 1, "error_info": "请求数据失败{}".format(e)}
        logger.error(info)
        return jsonify(info)
    if "query" not in params:
        info = {"error_code": 2, "error_info": "query字段缺失"}
        logger.error(info)
        return jsonify(info)
    else:
        if "image" not in params["query"]:
            info = {"error_code": 2, "error_info": "image字段缺失"}
            logger.error(info)
            return jsonify(info)
        if "text" not in params["query"]:
            info = {"error_code": 2, "error_info": "text字段缺失"}
            logger.error(info)
            return jsonify(info)
    if params:
        try:
            response = vlchat.chat(params["query"])
            result["data"] = response
        except Exception as e:
            #traceback.print_exc(e)
            logger.error(e)
            result["error_code"] = 3
            result["error_info"] = e
            return jsonify(result)

    return jsonify(result)

 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8086, debug=False)
