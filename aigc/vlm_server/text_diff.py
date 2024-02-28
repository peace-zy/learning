from difflib import Differ, HtmlDiff
import sys
import os

color = {"+": "\033[32m{}\033[0m",
         "-": "\033[31m{}\033[0m"}
def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]

def color_text(tokens):
    text = ""
    for t in tokens:
        if t[1] in color:
            text += color[t[1]].format(t[0])
        else:
            text += t[0]
    return text


def diff_texts_to_html(text1, text2):
    d = HtmlDiff()
    html_content = d.make_file(text1, text2)
    with open("d.html", "w") as f:
        f.write(html_content)
def main():
    infile = sys.argv[1]
    with open(infile, "r") as f:
        for line in f:
            fields = line.strip().split("\t")
            gt, v_0_1_0, v_0_1_1 = fields[0], fields[1], fields[2]
            #diff_texts_to_html(gt, v_0_1_0)
            tokens = diff_texts(gt, v_0_1_0)
            print("gt vs v_0_1_0")
            print(color_text(tokens))
            tokens = diff_texts(gt, v_0_1_1)
            print("gt vs v_0_1_1")
            print(color_text(tokens))
    return

if __name__ == "__main__":
    main()
