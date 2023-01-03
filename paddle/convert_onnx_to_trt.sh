ENV_PATH="env"

export LD_LIBRARY_PATH=${ENV_PATH}/cuda-10.2/lib64/:${ENV_PATH}/cudnn-8.0/cuda/lib64/:${ENV_PATH}/TensorRT-7.1.3.4/lib/:$LD_LIBRARY_PATH

ONNX_MODEL_PATH="modified_network/conv_nhwc_to_nchw/in_onnx"
#ONNX_MODEL_NAME="online_clip_model.onnx"
#ONNX_MODEL_NAME="online_clip_model.onnx.simple"
ONNX_MODEL_NAME="online_clip_model.onnx.simple"
SAVE_ENGINE_PATH="modified_network/conv_nhwc_to_nchw/out_trt"

if [ ! -d ${SAVE_ENGINE_PATH} ]; then
    mkdir -p ${SAVE_ENGINE_PATH}
fi

function convert_trt_in_fp32_mode()
{
    #TRT_ENGINE_NAME="clip.fp32.trt"
    TRT_ENGINE_NAME="clip_simple.fp32.trt"
    #TRT_ENGINE_NAME="clip_simple.fp32.trt.opt"
    #TRT_ENGINE_NAME="clip_simple.fp32.trt.opt.fp16"
    /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib64/:$LD_LIBRARY_PATH ${ENV_PATH}/TensorRT-7.1.3.4/bin/trtexec \
        --onnx=${ONNX_MODEL_PATH}/${ONNX_MODEL_NAME} \
        --saveEngine=${SAVE_ENGINE_PATH}/${TRT_ENGINE_NAME} \
        --minShapes=images:1x3x224x224,save_infer_model/scale_0.tmp_0:1x512 \
        --optShapes=images:16x3x224x224,save_infer_model/scale_0.tmp_0:16x512 \
        --maxShapes=images:32x3x224x224,save_infer_model/scale_0.tmp_0:32x512 \
        --explicitBatch \
        --device=0 \
        --verbose \
        --buildOnly \
        --workspace=4000
}

function convert_trt_in_int8_mode()
{
    TRT_ENGINE_NAME="clip.int8.trt"
    CALIB="caches/clip.cache"
    /opt/compiler/gcc-4.8.2/lib64/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-4.8.2/lib64/:$LD_LIBRARY_PATH ${ENV_PATH}/TensorRT-7.1.3.4/bin/trtexec \
        --onnx=${ONNX_MODEL_PATH}/${ONNX_MODEL_NAME} \
        --saveEngine=${SAVE_ENGINE_PATH}/${TRT_ENGINE_NAME} \
        --minShapes=image:1x3x224x224,softmax_0.tmp_0:1x512 \
        --optShapes=image:8x3x224x224,softmax_0.tmp_0:8x512 \
        --maxShapes=image:16x3x224x224,softmax_0.tmp_0:16x512 \
        --explicitBatch \
        --device=0 \
        --verbose \
        --buildOnly \
        --int8 \
        --calib=${CALIB}\
        --workspace=4000
}

function usage()
{
    echo -e """usage:\e[32m
         convert onnx model to trt in [fp32] mode:
         please set ONNX_MODEL_PATH
         sh convert_onnx_to_trt.sh 0

         convert onnx model to trt in [int8] mode:
         please set ONNX_MODEL_PATH
         sh convert_onnx_to_trt.sh 1\e[0m"""
}

if [ $# -eq 0 ];then
    usage
    exit
fi

mode=$1
echo "mode="${mode}
if [ ${mode} == 0 ];then
    convert_trt_in_fp32_mode
elif [ ${mode} == 1 ];then
    convert_trt_in_int8_mode
else
    echo "invalid mode ["${mode}"] support 0/1 mode"
fi
