import os
import sys
from PIL import Image
import numpy as np
import json

import torch
import clip
###########1. 加载测试图像--转换需要
image = Image.open("image/book.jpg")
img = image.resize((224,224), resample=Image.BICUBIC)
img = np.array(img).astype("float32") / 255.0
print('img shape', img.shape)
img -= [0.48145466, 0.4578275, 0.4082107]
img /= [0.26862954, 0.26130258, 0.27577711]
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)
print('img shape', img.shape)
batch_size = 1
inputs = np.expand_dims(
    img, axis=0).repeat(
    batch_size, axis=0).copy()

###########2. 加载pytorch模型
pretrained_weights = '/home/zhangyan75/.cache/clip/ViT-B-32.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocssor = clip.load('ViT-B/32', device)
import pdb
pdb.set_trace()
model, preprocssor = clip.load('ViT-B/16', device)
model.eval()
torch.save(model.visual.float().state_dict(), 'only_image.pth')
'''
for para in model.parameters():
    print(para.data.shape)
'''
out = {}
for param_tensor in model.state_dict():
    param = model.state_dict()[param_tensor].cpu().numpy()
    out[param_tensor] = param
    print(param_tensor,'\t',model.state_dict()[param_tensor].size())
np.save('weight.npy', out)
input_tensor = preprocssor(image).unsqueeze(0).to(device)

input_tensor_1 = torch.tensor(img).to(device)
print('input_tensor_1 shape', input_tensor_1.shape)

print("input_tensor, input_tensor_1", abs(input_tensor.cpu().numpy() - input_tensor_1.cpu().numpy()).max(),  abs(input_tensor.cpu().numpy() - input_tensor_1.cpu().numpy()).min())

print('in', input_tensor.shape)
#input_tensor = input_tensor.type(model.visual.conv1.weight.dtype)
model = model.float()
image_features = model.encode_image(input_tensor_1)
image_features_ = model.encode_image(input_tensor)
#sys.exit(1)
from torch import nn
class CLIP_VIS(nn.Module):
    def __init__(self, vis):
        super().__init__()
        self.vis = vis
    def forward(self, image):
        return self.vis(image)

clip_vis = CLIP_VIS(model.visual.float())
torch.save(clip_vis.state_dict(), 'only_image_branch.pth')
np.savetxt('py_out_b_16.npy', image_features.detach().cpu().numpy())
np.savetxt('py_out_b_16_.npy', image_features_.detach().cpu().numpy())
diff = image_features_.detach().cpu().numpy() - image_features.detach().cpu().numpy()
print(diff.min(), diff.max())

#sys.exit(1)
#model.visual()

###########3. 先转换为onnx
jit_type = "trace"    #转换类型
save_path = 'pt_2_onnx_out'
if not os.path.exists(save_path):
    os.makedirs(save_path)
export_onnx_file = os.path.join(save_path, 'ViT-B-16-pt.onnx')  #输出文件
'''
torch.onnx.export(model.visual.float(),
                    input_tensor,
                    export_onnx_file,
                    opset_version=9,   #opset_version 9不支持多输出
                    verbose=True,
                    training=False,
                    do_constant_folding=True,   # 是否执行常量折叠优化
                    input_names=["input"],  # 输入名
                    output_names=["output"],  # 输出名
                    dynamic_axes={'input':[0],'output': [0]},
                    )

torch.onnx.export(model.visual.float(),
                    input_tensor,
                    export_onnx_file,
                    opset_version=11,   #opset_version 9不支持多输出
                    input_names=["input"],  # 输入名
                    output_names=["output"],  # 输出名
                    dynamic_axes={'input':[0],'output': [0]},
                    )
'''

torch.onnx.export(model.visual.float(),
                    input_tensor,
                    export_onnx_file,
                    opset_version=11,   #opset_version 9不支持多输出
                    input_names=["images"],  # 输入名
                    output_names=["output"],  # 输出名
                    dynamic_axes={'images':[0],'output': [0]},
                    )



print(model.visual.conv1.weight.dtype)

import onnxruntime as ort
import onnx
set_batch = -1
net = onnx.load(export_onnx_file)
in_batch = net.graph.input[0].type.tensor_type.shape.dim[0]
print(in_batch)
in_batch.dim_value = set_batch
print(net.graph.input[0].type.tensor_type.shape.dim[0])

out_batch = net.graph.output[0].type.tensor_type.shape.dim[0]
print(out_batch)
out_batch.dim_value = set_batch
print(net.graph.output[0].type.tensor_type.shape.dim[0])

print(onnx.helper.printable_graph(net.graph))  # 输出onnx的计算图
onnx.save(net, export_onnx_file)

ort_session = ort.InferenceSession(export_onnx_file)
o_outputs = ort_session.run(None, {'images': img.astype(np.float32)})


print(image_features.detach().cpu().numpy())
print(o_outputs[0])
np.savetxt('py_out_b_16_onnx.npy', o_outputs[0])
print('torch VS onnx diff ----', 'max: ', abs(image_features.detach().cpu().numpy() - o_outputs[0]).max(), \
      abs(image_features.detach().cpu().numpy() - o_outputs[0]).min())
