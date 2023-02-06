#!/usr/bin/env python3
"""
inference script
"""
import argparse
from typing import Tuple, List

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import cv2
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#batch = 1
filename = 'book.jpg'

def get_img_np_nchw(filename):
    """生成网络输入数据
    Args:
        filename: 本地图像路径名称
    Returns:
        numpy: [1, c, h, w] rgb格式
    """
    """
    image = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    print(image.shape)
    image_cv = cv2.resize(image, (224, 224))
    #image_cv = cv2.resize(image, (416, 416))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    """


    from PIL import Image
    image = Image.open(filename)
    img = image.resize((224,224), resample=Image.BICUBIC)
    img = img.convert('RGB')
    img = np.array(img).astype("float32") / 255.0
    img -= [0.48145466, 0.4578275, 0.4082107]
    img /= [0.26862954, 0.26130258, 0.27577711]
    print(img.shape)
    # nchw
    inputs = np.transpose(img, (2, 0, 1))
    #print(inputs.shape)
    # nhcw

    img_np_nchw = np.expand_dims(inputs, axis=0)
    return img_np_nchw

def is_fixed(shape: Tuple[int]):
    """判断是否固定shape"""
    return not is_dynamic(shape)


def is_dynamic(shape: Tuple[int]):
    """判断是否动态shape"""
    return any(dim is None or dim < 0 for dim in shape)


def setup_binding_shapes(
    batch,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    host_inputs: List[np.ndarray],
    input_binding_idxs: List[int],
    output_binding_idxs: List[int],
):
    """设置binding形状"""
    # Explicitly set the dynamic input shapes, so the dynamic output
    # shapes can be computed internally
    for host_input, binding_index in zip(host_inputs, input_binding_idxs):
        context.set_binding_shape(binding_index, host_input.shape)

    assert context.all_binding_shapes_specified
    '''
    '''

    host_outputs = []
    device_outputs = []
    profile_index = context.active_optimization_profile
    profile_shapes = engine.get_profile_shape(profile_index, 0)
    for binding_index in output_binding_idxs:
        output_shape = context.get_binding_shape(binding_index)
        output_shape = [batch] + list(output_shape[1:])
        print(output_shape)
        # Allocate buffers to hold output results after copying back to host
        buffer = np.empty(output_shape, dtype=np.float32)
        host_outputs.append(buffer)
        # Allocate output buffers on device
        device_outputs.append(cuda.mem_alloc(buffer.nbytes))

    return host_outputs, device_outputs


def get_binding_idxs(engine: trt.ICudaEngine, profile_index: int):
    """获取binding索引"""
    # Calculate start/end binding indices for current context's profile
    num_bindings_per_profile = engine.num_bindings // engine.num_optimization_profiles
    start_binding = profile_index * num_bindings_per_profile
    end_binding = start_binding + num_bindings_per_profile
    print("Engine/Binding Metadata")
    print("\tNumber of optimization profiles: {}".format(engine.num_optimization_profiles))
    print("\tNumber of bindings per profile: {}".format(num_bindings_per_profile))
    print("\tFirst binding for profile {}: {}".format(profile_index, start_binding))
    print("\tLast binding for profile {}: {}".format(profile_index, end_binding-1))

    # Separate input and output binding indices for convenience
    input_binding_idxs = []
    output_binding_idxs = []
    for binding_index in range(start_binding, end_binding):
        if engine.binding_is_input(binding_index):
            input_binding_idxs.append(binding_index)
        else:
            output_binding_idxs.append(binding_index)

    return input_binding_idxs, output_binding_idxs


def load_engine(filename: str):
    """加载trt引擎"""
    # Load serialized engine file into memory
    with open(filename, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def get_random_inputs(
    batch,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    input_binding_idxs: List[int],
    seed: int = 42,
):
    """获取输入"""
    # Input data for inference
    host_inputs = []
    print("Generating Random Inputs")
    print("\tUsing random seed: {}".format(seed))
    np.random.seed(seed)
    for binding_index in input_binding_idxs:
        # If input shape is fixed, we'll just use it
        input_shape = context.get_binding_shape(binding_index)
        input_shape = [batch] + list(input_shape[1:])
        input_name = engine.get_binding_name(binding_index)
        print("\tInput [{}] shape: {}".format(input_name, input_shape))
        # If input shape is dynamic, we'll arbitrarily select one of the
        # the min/opt/max shapes from our optimization profile
        if is_dynamic(input_shape):
            profile_index = context.active_optimization_profile
            profile_shapes = engine.get_profile_shape(profile_index, binding_index)
            #profile_shapes = [(8, 3, 224, 224), (8, 3, 224, 224), (8, 3, 224, 224)]
            print("\tProfile Shapes for [{}]: [kMIN {} | kOPT {} | kMAX {}]".format(input_name, *profile_shapes))
            # 0=min, 1=opt, 2=max, or choose any shape, (min <= shape <= max)
            input_shape = profile_shapes[1]
            print("\tInput [{}] shape was dynamic, setting inference shape to {}".format(input_name, input_shape))

        img_np_nchw = get_img_np_nchw(filename)
        #img_np_nchw = get_img_np_nchw_new(filename)
        img_np_nchw = img_np_nchw.astype(dtype=np.float32)
        img_np_nchw = img_np_nchw.repeat(batch, 0)
        print(img_np_nchw.shape)
        '''
        img_np_nchw = np.load('image.npy')
        img_np_nchw = img_np_nchw.repeat(batch, 0)
        '''
        host_inputs.append(img_np_nchw)
        #host_inputs.append(np.random.random(input_shape).astype(np.float32))

    return host_inputs


def main():
    """main主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, type=str,
                        help="Path to TensorRT engine file.")
    parser.add_argument("-b", "--batch", type=int, default=1,
                        help="Path to TensorRT engine file.")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    global batch
    batch = args.batch

    # Load a serialized engine into memory
    engine = load_engine(args.engine)
    print("Loaded engine: {}".format(args.engine))

    # Create context, this can be re-used
    context = engine.create_execution_context()
    # Profile 0 (first profile) is used by default
    context.active_optimization_profile = 0
    print("Active Optimization Profile: {}".format(context.active_optimization_profile))

    # These binding_idxs can change if either the context or the
    # active_optimization_profile are changed
    input_binding_idxs, output_binding_idxs = get_binding_idxs(
        engine, context.active_optimization_profile
    )
    input_names = [engine.get_binding_name(binding_idx) for binding_idx in input_binding_idxs]

    # Generate random inputs based on profile shapes
    while True:
        batch = int(input('batch='))
        host_inputs = get_random_inputs(batch, engine, context, input_binding_idxs, seed=args.seed)

        # Allocate device memory for inputs. This can be easily re-used if the
        # input shapes don't change
        device_inputs = [cuda.mem_alloc(h_input.nbytes) for h_input in host_inputs]
        # Copy host inputs to device, this needs to be done for each new input
        for h_input, d_input in zip(host_inputs, device_inputs):
            cuda.memcpy_htod(d_input, h_input)

        print("Input Metadata")
        print("\tNumber of Inputs: {}".format(len(input_binding_idxs)))
        print("\tInput Bindings for Profile {}: {}".format(context.active_optimization_profile, input_binding_idxs))
        print("\tInput names: {}".format(input_names))
        print("\tInput shapes: {}".format([inp.shape for inp in host_inputs]))

        # This needs to be called everytime your input shapes change
        # If your inputs are always the same shape (same batch size, etc.),
        # then you will only need to call this once
        host_outputs, device_outputs = setup_binding_shapes(
            batch, engine, context, host_inputs, input_binding_idxs, output_binding_idxs,
        )
        output_names = [engine.get_binding_name(binding_idx) for binding_idx in output_binding_idxs]

        print("Output Metadata")
        print("\tNumber of Outputs: {}".format(len(output_binding_idxs)))
        print("\tOutput names: {}".format(output_names))
        print("\tOutput shapes: {}".format([out.shape for out in host_outputs]))
        print("\tOutput Bindings for Profile {}: {}".format(context.active_optimization_profile, output_binding_idxs))

        # Bindings are a list of device pointers for inputs and outputs
        bindings = device_inputs + device_outputs

        # Inference
        elaps = 0
        test_num = 100
        #test_num = 1
        d = 0
        warmup = 10
        for i in range(0, test_num + warmup):
            start = time.time()
            context.execute_v2(bindings)

            # Copy outputs back to host to view results
            for h_output, d_output in zip(host_outputs, device_outputs):
                cuda.memcpy_dtoh(h_output, d_output)
            #out = host_outputs[0]
            #print('match', host_outputs == out)

            end = time.time()
            if i >= warmup:
                elaps += (end - start)
        print('\033[32minference time:\033[0m {}ms'.format(1000 * elaps / test_num / batch))

        # View outputs
        print("Inference Outputs:", host_outputs)
        out = host_outputs[0]
        print('match', host_outputs == out)
        print(np.isclose(host_outputs, host_outputs[0]).all())
        """
        out_nchw = np.loadtxt('modified_network/nchw_output_online_clip_pd8.5.1.npy')
        out_nhwc = np.loadtxt('modified_network/nhwc_output_online_clip_pd8.5.1.npy')
        out_online_nhwc = np.loadtxt('online_clip_pd8.5.1.npy')
        print('nchw vs nhwc')
        diff = abs(out_nchw - out_nhwc)
        print(diff.max(), diff.min())
        print('nchw vs trt_nchw')
        diff = abs(out_nchw - out)
        print(diff.max(), diff.min())
        print('nchw vs online_nhwc')
        diff = abs(out_nchw - out_online_nhwc)
        print(diff.max(), diff.min())
        print('trt_nchw vs online_nhwc')
        diff = abs(out - out_online_nhwc)
        print(diff.max(), diff.min())
        """

        '''
        print(np.max(np.max(out[0], axis=1)))
        print(np.argmax(np.max(out[0], axis=1)))
        indices = np.argsort(out[0,:])[::-1]
        sum = np.sum(np.exp(out[0, :]))
        with open('labels/imagenet1k_labels.txt', 'r') as f:
            names = [line.rstrip() for line in f.readlines()]

        print([(idx, names[idx], 100 * np.exp(out[0, idx])/sum) for idx in indices[:5]])
        print([(idx, names[idx], 100 * np.exp(out[2, idx])/sum) for idx in indices[:5]])
        '''
        np.savetxt('trt_out.npy', out)

    # Cleanup (Can also use context managers instead)
    del context
    del engine

if __name__ == "__main__":
    main()
