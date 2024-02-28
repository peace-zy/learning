import gradio as gr
import json
import uuid
import time
import requests
from argparse import ArgumentParser
import os
import random
import traceback
from difflib import Differ

GRADIO_TEMP_DIR = "tmp"
if not os.path.exists(GRADIO_TEMP_DIR):
    os.makedirs(GRADIO_TEMP_DIR)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

images = {}
def get_args():
    parser = ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args

def predict(image_file):
    global images
    image_name = os.path.basename(image_file)
    print(image_file, images[image_name])
    #image_file = url_sekuai_map[os.path.basename(image_file)]
    image_file = images[image_name]["sekuai"]
    gt = images[image_name]["gt"]
    print(image_file)
    url = "http://10.229.132.208:8086/predict" 
    post_data = {"request_id": str(uuid.uuid1()),
                 "query": {
                    "image": image_file, # Either a local path or an url
                    "text": "用中文讲解下这个户型"
                }
            }
    headers = {'Content-Type': 'application/json'}
    start = time.time()
    response = requests.post(url=url, headers=headers, json=post_data, timeout=200)
    end = time.time()
    print("elaps time = {}ms".format((end - start) * 1000))
    res = json.loads(response.content.decode("utf-8"))
    if res["error_code"] == 0:
        return (res["data"], gt)

    return (res["error_info"], "")

def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

def launch_demo(args):
    global images
    with open("ori_sekuai_gt.txt", "r") as f:
        for line in f:
            try:
                fields = line.strip().split("\t")
                ori_image_file, sekuai_image_file, gt = fields[0], fields[1], fields[2]
                image_name = os.path.basename(ori_image_file)
                images[image_name] = {"ori": ori_image_file, 
                                      "sekuai": sekuai_image_file,
                                      "gt": gt}
            except Exception as e:
                traceback.print_exc()
                continue
    examples = []
    NUM = 100
    idx = 0
    for k, v in images.items():
        examples.append(v["ori"])
    examples = random.choices(examples, k=NUM)

    #image_prompt = gr.Image(type="filepath", label="上传图片", value="https://img.ljcdn.com/hdic-frame/cbebd44c-6194-485e-b346-a51fde5b666e.png!m_fill,w_750,h_562,l_fbk", sources="upload", width=512)
    #caption = gr.Textbox(type="text", label="户型基础描述", value=None)

    with gr.Blocks() as demo:
        #examples = []
        gr.Markdown("""<center><font size=6>户型基础信息描述</center>""")
        with gr.Tab(""):
            with gr.Group():
                with gr.Row():
                    #image = gr.Image(type="filepath", label="", value="/aistudio/workspace/research/flask_test/new_floor_plan_black.png", sources="upload", width=512)
                    image = gr.Image(type="filepath", label="示例图片", value="/aistudio/workspace/research/flask_test/images/ori/1120042403528716.png", sources="upload", height=512)
                    with gr.Column(scale=1):
                        caption = gr.Textbox(type="text", label="预测", value=None, lines=4)
                        gt = gr.Textbox(type="text", label="GT", value=None, lines=4)
                        diff = gr.HighlightedText(
                                        label="差异【 + : gt有预测无; - :gt无预测有】",
                                        combine_adjacent=True,
                                        show_legend=True,
                                        color_map={"+": "red", "-": "green"})
            submit_btn = gr.Button("运行")
            dd = gr.Examples(
                examples=examples,
                inputs=image,
                outputs=[caption, gt],
                fn=None,
                cache_examples=False,
                examples_per_page=36
            )
            submit_btn.click(predict, [image], [caption, gt], show_progress=True).then(
                    diff_texts, [caption, gt], diff)
            

    """   
    demo = gr.Interface(
        fn=predict, 
        title="户型基础描述",
        inputs=image_prompt,
        outputs=caption,
        allow_flagging="never",
        examples=examples
    )
    """
    gr.close_all()
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )

def main():
    args = get_args()
    print(args)
    launch_demo(args)

if __name__ == "__main__":
    main()
