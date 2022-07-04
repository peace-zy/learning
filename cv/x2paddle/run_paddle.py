import os
import sys
from PIL import Image
import numpy as np
import paddle
from pd_model_pd2_0.x2paddle_code import ONNXModel

###########1. 加载测试图像--转换需要
image = Image.open("image/book.jpg")
img = image.resize((224,224))
img = np.array(img).astype("float32") / 255.0
img -= [0.48145466, 0.4578275, 0.4082107]
img /= [0.26862954, 0.26130258, 0.27577711]
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, 0)

###########2. 加载paddle模型
#paddle.disable_static()
params = paddle.load('pd_model_pd2_0/model.pdparams')
#device = "cuda" if torch.cuda.is_available() else "cpu"

p_model = ONNXModel()
p_model.set_dict(params, use_structured_name=True)
p_model.eval()
p_out = p_model(paddle.to_tensor(img, dtype='float32'))
print(p_out[0].cpu().numpy())
np.savetxt('pd.npy', p_out[0].cpu().numpy())

#print('torch VS paddle diff ----', 'max: ', abs(torch_output[0].detach().numpy()-p_out[0].cpu().numpy()).max(), 'min: ', abs((torch_output[0].detach().numpy()-p_out[0].cpu().numpy()).min()))
