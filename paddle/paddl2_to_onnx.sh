paddle2onnx --model_dir conv_nhwc_to_nchw \
            --model_filename __model__ \
            --params_filename __params__ \
            --save_file online_clip_model_20221227.onnx \
            --enable_dev_version True
