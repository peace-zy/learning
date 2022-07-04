# pytorch_1.7.1  
# clip https://github.com/openai/CLIP. 
paddle_v1.8.5

https://github.com/PaddlePaddle/X2Paddle  
pip install x2paddle==0.8.9
pip install onnx==1.6.0

x2paddle --framework=onnx --model=ViT-B-32-pt_to_pd_2_0.onnx --save_dir=pd_model --params_merge  



https://zhuanlan.zhihu.com/p/396948150

[1]  CUDA_VISIBLE_DEVICES=2 python convert_pytorch_to_onnx.py
[2] x2paddle --framework=onnx --model=ViT-B-32-pt_to_pd_2_0.onnx --save_dir=ViT-B-32_pd_model --params_merge

notice： 不同版本的Pillow读取结果不一定一致，保证版本一致后，参看转换前后差异。（libjpeg的）。
