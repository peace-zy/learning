# 判断文本是否是英文，去除标点和空格后，帮段是否是alpha
import string
translator = str.maketrans('', '', string.punctuation + string.whitespace)
question.translate(translator).encode("utf-8").isalpha()
