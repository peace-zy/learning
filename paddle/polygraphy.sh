# 生成测试脚本
polygraphy run --gen a.txt modified_network/conv_nhwc_to_nchw/in_onnx/online_clip_model.onnx.simple --trt --onnxrt --model-type onnx --save-engine trt_out --trt-min-shapes images:[1,3,224,224] --trt-opt-shapes images:[16,3,224,224] --trt-max-shapes images:[32,3,224,224] --iterations 50 --input-shapes images:[32,3,224,224] --rtol 1e-4 --atol 1e-4
https://github.com/NVIDIA/TensorRT/tree/a912a045c32a2247b0d1d669962706d28cac7e40/tools/Polygraphy/examples/api
