inpath="modified_network/conv_nhwc_to_nchw"
save_path=${inpath}/in_onnx
if [ ! -d ${save_path} ];then
  mkdir ${save_path}
fi
paddle2onnx --model_dir modified_network/conv_nhwc_to_nchw \
            --model_filename __model__ \
            --params_filename __params__ \
            --save_file ${save_path}/online_clip_model.onnx \
            --enable_dev_version True
