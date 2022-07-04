# pytorch_1.7.1  
# clip https://github.com/openai/CLIP. 
paddle_v1.8.5

pip install x2paddle==0.8.9
pip install onnx==1.6.0

x2paddle --framework=onnx --model=test_new.onnx.1.8.1 --save_dir=pd_model --params_merge  


https://zhuanlan.zhihu.com/p/396948150

[1]  CUDA_VISIBLE_DEVICES=2 python convert_pytorch_to_onnx.py
[2] x2paddle --framework=onnx --model=ViT-B-32-pt_to_pd_2_0.onnx --save_dir=ViT-B-32_pd_model --params_merge. 
