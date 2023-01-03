#-*-coding: utf-8-*-
本文件实现了paddle1.8.5调用模型。
"""


import argparse
import utils
import numpy as np
import logging
import time
import os
from tqdm import tqdm
import traceback
from PIL import Image
import hashlib

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """parse_args"""
    def str2bool(v):
        """str2bool"""
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)
    parser.add_argument("-p", "--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_trt", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str)

    return parser.parse_args()


def create_predictor(args):
    """create_predictor"""
    config = AnalysisConfig(args.model_file, args.params_file)
    '''
    if args.use_trt:
        config.enable_tensorrt_engine(workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=AnalysisConfig.Precision.Float32,
            use_static=False,
            use_calib_mode=False)
    '''

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=AnalysisConfig.Precision.Half
            if args.use_fp16 else AnalysisConfig.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)

    return predictor



def main():
    """main"""
    args = parse_args()

    if not args.enable_benchmark:
        assert args.batch_size == 1
        assert args.use_fp16 is False
    else:
        assert args.use_gpu is True
        assert args.model_name is not None
        #assert args.use_tensorrt is True
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    predictor = create_predictor(args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])

    test_num = 50
    test_time = 0.0
    wlines = []
    if not args.enable_benchmark:
        if args.image_file is not None:
            #image = Image.open("book.jpg")
            image = Image.open(args.image_file)
            img = image.resize((224, 224), resample=Image.BICUBIC)
            img = img.convert('RGB')
            img = np.array(img).astype("float32") / 255.0
            img -= [0.48145466, 0.4578275, 0.4082107]
            img /= [0.26862954, 0.26130258, 0.27577711]
            print(img.shape)
            # nchw
            #inputs = np.transpose(img, (2, 0, 1))
            #print(inputs.shape)
            # nhwc
            inputs = img

            inputs = np.expand_dims(
                inputs, axis=0).repeat(
                    args.batch_size, axis=0).copy()
            input_tensor.copy_from_cpu(inputs)

            predictor.zero_copy_run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            save_path = 'infer_out'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            out_file = os.path.join(save_path, 'nhwc_online_clip_pd8.5.1.npy')
            np.savetxt(out_file, output)
            logger.info(output)
            cls = np.argmax(output)
            score = output[cls]
            logger.info("class: {0}".format(cls))
            logger.info("score: {0}".format(score))
    else:
        for i in range(0, test_num + 10):
            # nchw
            #inputs = np.random.rand(args.batch_size, 3, 224,
            #                        224).astype(np.float32)
            # nhwc
            inputs = np.random.rand(args.batch_size, 224,
                                    224, 3).astype(np.float32)
            '''
            from PIL import Image
            image = Image.open("../../CLIP-main/image/book.jpg")
            img = image.resize((224,224))
            img = np.array(img).astype("float32") / 255.0
            img -= [0.48145466, 0.4578275, 0.4082107]
            img /= [0.26862954, 0.26130258, 0.27577711]
            inputs = np.transpose(img, (2, 0, 1))
            preprocess_time = time.time()
            logger.info("preprocess time {0}(ms)".format((preprocess_time - start_time) * 1000))

            inputs = np.expand_dims(
                inputs, axis=0).repeat(
                    args.batch_size, axis=0).copy()
            '''

            start_time = time.time()
            input_tensor.copy_from_cpu(inputs)

            predictor.zero_copy_run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            if i >= 10:
                test_time += time.time() - start_time

        fp_message = "FP16" if args.use_fp16 else "FP32"
        logger.info("{0}\t{1}\tbatch size: {2}\ttime(ms): {3}\tper_image_time(ms): {4}".format(
            args.model_name, fp_message, args.batch_size, 1000 * test_time / test_num,
            1000 * test_time / test_num / args.batch_size))


if __name__ == "__main__":
    main()
