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

def diff_texts(text1, text2, text3):
    d = Differ()
    diff_12 = [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text2, text1)
    ]
    diff_13 = [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text3, text1)
    ]
    return diff_12, diff_13

def launch_demo(args):
    demo = gr.Interface(
        fn=diff_texts,
        inputs=[
            gr.Textbox(
                label="GT",
                lines=3,
                value="",
            ),
            gr.Textbox(
                label="v0.1.0",
                lines=3,
                value="",
            ),
            gr.Textbox(
                label="v0.1.1",
                lines=3,
                value="",
            ),

        ],
        outputs=[
            gr.HighlightedText(
                label="GT与v0.1.0 差异【 + : gt有预测无; - :gt无预测有】",
                combine_adjacent=True,
                show_legend=True,
                color_map={"+": "red", "-": "green"}
            ),
            gr.HighlightedText(
                label="GT与v0.1.1 差异【 + : gt有预测无; - :gt无预测有】",
                combine_adjacent=True,
                show_legend=True,
                color_map={"+": "red", "-": "green"}
            ),

        ],

        title="文本diff",
        allow_flagging="never",
    )

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
