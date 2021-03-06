import os
import sys
from PIL import Image
import numpy as np

import torch
import clip
###########1. 加载测试图像--转换需要
image = Image.open("image/book.jpg")
img = image.resize((224,224))
img = np.array(img).astype("float32") / 255.0
img -= [0.48145466, 0.4578275, 0.4082107]
img /= [0.26862954, 0.26130258, 0.27577711]
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)

###########2. 加载pytorch模型
pretrained_weights = 'only_image.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(pretrained_weights)
model.eval()
#input_tensor = preprocssor(image).unsqueeze(0).to(device)

input_tensor = torch.tensor(img).to(device)

#print(abs(input_tensor.cpu().numpy() - input_tensor_1.cpu().numpy()).max(),  abs(input_tensor.cpu().numpy() - input_tensor_1.cpu().numpy()).min())

#input_tensor = input_tensor.type(model.visual.conv1.weight.dtype)
model = model.float()
image_features = model(input_tensor)
print(image_features.detach().cpu().numpy())
np.savetxt('py_out.npy', image_features.detach().cpu().numpy())

#torch.save(model.visual.float(), 'only_image.pth')
#model.visual()

'''
###########3. 先转换为onnx
jit_type = "trace"    #转换类型
export_onnx_file = "test_new.onnx"  #输出文件
torch.onnx.export(model,
                    input_tensor,
                    export_onnx_file,
                    opset_version=9,   #opset_version 9不支持多输出
                    verbose=True,
                    training=False,
                    do_constant_folding=False,   # 是否执行常量折叠优化
                    input_names=["input"],  # 输入名
                    output_names=["output"],  # 输出名
                    )


import onnxruntime as ort
ort_session = ort.InferenceSession(export_onnx_file)
o_outputs = ort_session.run(None, {'input': img.astype(np.float32)})
#o_outputs = ort_session.run(None, {'input': img.astype(np.float16)})

print(image_features.detach().cpu().numpy())
print(o_outputs[0])
print('torch VS onnx diff ----', 'max: ', abs(image_features.detach().cpu().numpy() - o_outputs[0]).max(), \
      abs(image_features.detach().cpu().numpy() - o_outputs[0]).min())

'''
