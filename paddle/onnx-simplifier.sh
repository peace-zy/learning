#!/bin/bash
python -m onnxsim modified_network/conv_nhwc_to_nchw/in_onnx/online_clip_model.onnx modified_network/conv_nhwc_to_nchw/in_onnx/online_clip_model.onnx.simple
