# environment
 - paddle 1.8.5
# 离线调用clip
  infer_for_clip_offline.sh
# 修改paddle模型conv dataformat NHWC to NHWC
 python paddle_add_op.py  
 out: modified_network
# paddle模型转onnx
 paddl2_to_onnx.sh
# onnx网络精简
 onnx-simplifier.sh
# onnx转trt
 convert_onnx_to_trt.sh
# 使用trt测试
 python infer_for_trt.py -e modified_network/conv_nhwc_to_nchw/out_trt/clip_simple.fp32.trt -b 32


# 使用trt官方接口测试trt
 polygraphy run --gen - model.onnx --trt --onnxrt --iterations 50 --save-engine trt_out > run_polygraphy.py
