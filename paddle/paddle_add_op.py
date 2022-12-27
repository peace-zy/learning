from paddle import fluid
from PIL import Image
import numpy as np
import traceback
import sys
import hashlib
import copy

def get_nhwc_conv_res(inputs, weights):
    import paddle.fluid as fluid
    import numpy as np
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name="images", shape=[-1, 224, 224, 3], dtype='float32', append_batch_size=False)
        conv = fluid.layers.conv2d(input=data, num_filters=768, filter_size=16, stride=16, bias_attr=False, data_format='NHWC')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    for block in main_prog.blocks:
        for param in block.all_parameters():
            pd_var = fluid.global_scope().find_var(param.name)
            pd_param = pd_var.get_tensor()
            print("load: {}, shape: {}".format(param.name, param.shape))
            print("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            pd_param.set(weights, place)
            print("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            diff = weights - np.array(pd_param)
            print('min={}, max={}'.format(diff.min(), diff.max()))

    results = exe.run(main_prog,
                  feed={'images': inputs},
                  fetch_list=[conv])
    return results

def get_nchw_conv_res(inputs, weights):
    import paddle.fluid as fluid
    import numpy as np
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name="images", shape=[-1, 2, 224, 224], dtype='float32', append_batch_size=False)
        conv = fluid.layers.conv2d(input=data, num_filters=768, filter_size=16, stride=16, bias_attr=False, data_format='NCHW')

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    for block in main_prog.blocks:
        for param in block.all_parameters():
            pd_var = fluid.global_scope().find_var(param.name)
            pd_param = pd_var.get_tensor()
            print("load: {}, shape: {}".format(param.name, param.shape))
            print("Before setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            pd_param.set(weights, place)
            print("After setting the numpy array value: {}".format(np.array(pd_param).ravel()[:5]))
            diff = weights - np.array(pd_param)
            print('min={}, max={}'.format(diff.min(), diff.max()))

    results = exe.run(main_prog,
                  feed={'images': inputs},
                  fetch_list=[conv])
    return results

def count_params(inference_program):
    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 获取参数情况
    for block in inference_program.blocks:
        for var in block.vars:
            param = block.vars[var]
            try:
                if 'tmp' in var:
                    continue
                param = block.vars[var]
                shape = param.shape
                array = np.asarray(shape)  # 转换为numpy数组，方便后续计算
                array[array == -1] = 1
                mulValue = np.prod(array)  # 使用numpy prod接口计算数组所有元素之积

                Total_params += mulValue  # 总参数量
                if param.persistable and not param.stop_gradient:
                    Trainable_params += mulValue  # 可训练参数量
                else:
                    NonTrainable_params += mulValue  # 非可训练参数量
            except Exception as e:
                traceback.print_exc()
                continue

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def compare_conv(nchw_inputs, nhwc_inputs,
                 path='./online_clip/cspd_image_similarity', model_filename='__model__', params_filename='__params__'):
    exe = fluid.Executor(fluid.CPUPlace())
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path,
                                                                                          executor=exe,
                                                                                          model_filename=model_filename,
                                                                                          params_filename=params_filename)


    tensor = fluid.global_scope().find_var('patch_embedding.w_0').get_tensor()
    '''
    if tensor._is_initialized():
        print(tensor.shape())
        print(np.array(tensor))
    '''
    nhwc_conv = get_nhwc_conv_res(nhwc_inputs, np.array(tensor))

    nchw_conv = get_nchw_conv_res(nchw_inputs, tensor)
    print('nchw shape={}\tnhwc shape={}'.format(nchw_conv[0].shape, nhwc_conv[0].shape))
    #nchw shape=(1, 768, 14, 14) nhwc shape=(1, 14, 14, 768)
    #(1, 768, 14, 14) (1, 14, 14, 768)
    diff = np.transpose(nchw_conv[0], (0, 2, 3, 1)) - nhwc_conv[0]
    print('conv diff min={}, max={}'.format(diff.min(), diff.max()))


def get_nchw_res(nchw_inputs,
                 path='./online_clip/cspd_image_similarity', model_filename='__model__', params_filename='__params__'):
    exe = fluid.Executor(fluid.CPUPlace())
    #exe = fluid.Executor(fluid.CUDAPlace(0))
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path,
                                                                                          executor=exe,
                                                                                          model_filename=model_filename,
                                                                                          params_filename=params_filename)

    export_program = inference_program.clone()
    global_block = inference_program.global_block()
    export_global_block = export_program.global_block()
    '''
    global_block.vars.pop('images')
    global_block.desc._remove_var(b'images')
    global_block.create_var(name='images', shape=[-1, 3, 224, 224], dtype='float32')
    '''
    global_block.desc.var(b'images').set_shape([-1, 3, 224, 224])
    export_global_block.desc.var(b'images').set_shape([-1, 3, 224, 224])
    conv_op = None
    ops = []
    for idx, op in enumerate(global_block.ops):
        if op.type == 'conv2d':
            print(op.type)
            print(op)
            conv_op = op

        if op.type == 'reshape2' and 'patch_embedding.tmp_0' in op.input_arg_names:
            print(op.type)
            print(op)
            #transpose_to_nhwc = fluid.layers.transpose(global_block.var(conv_op.output('Output')[0]), perm=[0, 2, 3, 1])
            perm = [0, 2, 3, 1]
            out = global_block.create_var(name='trans_to_nhwc', shape=[-1, 14, 14, 768], dtype='float32')
            x_shape = global_block.create_var(name='trans_to_nhwc_shape', shape=[-1, 14, 14, 768], dtype='float32')
            x = global_block.var(conv_op.output('Output')[0])
            temp = global_block.desc.op(2)
            global_block._insert_op(2,
                type='transpose2',
                inputs={'X': [x]},
                outputs={'Out': [out],
                         'XShape': [x_shape]},
                attrs={'axis': perm})
            op.desc.set_input('X', ['trans_to_nhwc'])

            for ii in range(global_block.desc.op_size()):
                pp = global_block.desc.op(ii)
                if pp.type() == 'transpose2' and 'patch_embedding.tmp_0' in pp.input_arg_names():
                    print('[debug] transpose2 has insert in desc')



    '''
    for idx in range(0, len(global_block.ops) - 1):
        op = global_block.ops[idx]
        if op.type == 'reshape2' and 'trans_to_nhwc' in op.input_arg_names:
            ops.append(global_block.ops[-1])
            ops.append(op)
        else:
            ops.append(op)
        print(idx, len(ops))
    '''

    patch_embedding = global_block.ops[1]
    patch_embedding._set_attr('data_format', 'NCHW') # ori NHWC


    model_str = inference_program.to_string(throw_on_error=False, with_details=False)
    for param in export_global_block.all_parameters():
        print(param)

    model_str_file = 'model_str.txt'
    with open(model_str_file, 'w') as f:
        f.write(model_str)

    export_exe = fluid.Executor(fluid.CPUPlace())
    path = 'conv_nhwc_to_nchw'
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['images'],
        target_vars=fetch_targets, executor=export_exe, main_program=inference_program, params_filename=params_filename)

    nchw_exe = fluid.Executor(fluid.CPUPlace())
    results = nchw_exe.run(inference_program,
                  feed={feed_target_names[0]: nchw_inputs},
                  fetch_list=fetch_targets)

    #output = results.copy_to_cpu()
    output = results[0]
    nchw_output = output.flatten()
    np.savetxt('nchw_output_online_clip_pd8.5.1.npy', nchw_output)
    return nchw_output


def get_nhwc_res(nhwc_inputs,
                 path='./online_clip/cspd_image_similarity', model_filename='__model__', params_filename='__params__'):
    exe = fluid.Executor(fluid.CPUPlace())
    #exe = fluid.Executor(fluid.CUDAPlace(0))
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path,
                                                                                          executor=exe,
                                                                                          model_filename=model_filename,
                                                                                          params_filename=params_filename)

    results = exe.run(inference_program,
                  feed={feed_target_names[0]: nhwc_inputs},
                  fetch_list=fetch_targets)

    nhwc_output = results[0].flatten()
    np.savetxt('nhwc_output_online_clip_pd8.5.1.npy', nhwc_output)


    export_program = inference_program.clone()
    global_block = inference_program.global_block()
    global_block.desc.var(b'scale_0.tmp_0').set_shape([-1, 1, 197, 197])
    export_global_block = export_program.global_block()
    matmul_op = None
    matmul_pos = -1
    ops = []
    for idx, op in enumerate(global_block.ops):
        if op.type == 'matmul' and 'fill_constant_2.tmp_0' in op.input_arg_names:
            print(op.type)
            print(op)
            matmul_op = op
            matmul_pos = idx

        if op.type == 'scale' and "matmul_0.tmp_0" in op.input_arg_names:
            print(op.type)
            print(op)
            #transpose_to_nhwc = fluid.layers.transpose(global_block.var(conv_op.output('Output')[0]), perm=[0, 2, 3, 1])
            out = global_block.create_var(name='unsqueeze_to_scale', shape=[-1, 1, 197, 197], dtype='float32')
            x_shape = global_block.create_var(name='unsqueeze_to_scale_shape', shape=[-1, 1, 197, 197], dtype='float32')
            x = global_block.var(matmul_op.output('Out')[0])
            global_block._insert_op(idx,
                type='unsqueeze',
                inputs={'X': [x]},
                outputs={'Out': [out],
                         'XShape': [x_shape]},
                attrs={'axis': 0})
            op.desc.set_input('X', ['unsqueeze_to_scale'])

            for ii in range(global_block.desc.op_size()):
                pp = global_block.desc.op(ii)
                if pp.type() == 'scale' and 'matmul_0.tmp_0' in pp.input_arg_names():
                    print('[debug] scale has insert in desc')



    '''
    for idx in range(0, len(global_block.ops) - 1):
        op = global_block.ops[idx]
        if op.type == 'reshape2' and 'trans_to_nhwc' in op.input_arg_names:
            ops.append(global_block.ops[-1])
            ops.append(op)
        else:
            ops.append(op)
        print(idx, len(ops))
    '''


    model_str = inference_program.to_string(throw_on_error=False, with_details=False)
    for param in export_global_block.all_parameters():
        print(param)

    model_str_file = 'model_str.txt'
    with open(model_str_file, 'w') as f:
        f.write(model_str)

    export_exe = fluid.Executor(fluid.CPUPlace())
    path = 'add_unsqueeze_for_scale'
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['images'],
        target_vars=fetch_targets, executor=export_exe, main_program=inference_program, params_filename=params_filename)

    nchw_exe = fluid.Executor(fluid.CPUPlace())
    results = nchw_exe.run(inference_program,
                  feed={feed_target_names[0]: nhwc_inputs},
                  fetch_list=fetch_targets)

    #output = results.copy_to_cpu()
    output = results[0]
    unsqueeze_nhwc_output = output.flatten()
    np.savetxt('unsqueeze_nhwc_output_online_clip_pd8.5.1.npy', unsqueeze_nhwc_output)
    diff = nhwc_output - unsqueeze_nhwc_output
    print('unsqueeze_to_scale  min={}\tmax={}'.format(diff.min(), diff.max()))



    return nhwc_output

def compare_nchw_and_nhwc_infer(nchw_inputs, nhwc_inputs,
                 path='./online_clip/cspd_image_similarity', model_filename='__model__', params_filename='__params__'):
    nhwc_output = get_nhwc_res(nhwc_inputs, path=path, model_filename=model_filename, params_filename=params_filename)
    nchw_output = get_nchw_res(nchw_inputs, path=path, model_filename=model_filename, params_filename=params_filename)

    print('min={}, max={}'.format(np.abs((nhwc_output - nchw_output).min()), np.abs((nhwc_output - nchw_output).max())))

def main():
    path = "./online_clip/cspd_image_similarity"
    model_filename = '__model__'
    params_filename = '__params__'

    image = Image.open("../../CLIP-main/image/book.jpg")
    img = image.resize((224,224), resample=Image.BICUBIC)
    img = img.convert('RGB')
    img = np.array(img).astype("float32") / 255.0
    img -= [0.48145466, 0.4578275, 0.4082107]
    img /= [0.26862954, 0.26130258, 0.27577711]
    print(img.shape)
    nchw_inputs = np.transpose(img, (2, 0, 1))
    print(nchw_inputs.shape)
    batch_size = 1

    nchw_inputs = np.expand_dims(
        nchw_inputs, axis=0).repeat(
            batch_size, axis=0).copy()

    nhwc_inputs = np.expand_dims(
        img, axis=0).repeat(
            batch_size, axis=0).copy()
    compare_conv(nchw_inputs, nhwc_inputs)

    compare_nchw_and_nhwc_infer(nchw_inputs, nhwc_inputs)
    return



if __name__ == '__main__':
    main()
    '''
    path = 'ddfdfdfd'
    debug_fetch_targets = [inference_program.global_block().var('patch_embedding.tmp_0')]
    print(debug_fetch_targets)
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['images'],
        target_vars=debug_fetch_targets, executor=exe, main_program=inference_program, params_filename=params_filename)
    sys.exit(0)


    export_program = inference_program.clone()
    block = export_program.global_block()
    block.vars.pop('images')
    block.desc._remove_var(b'images')
    block.create_var(name='new_images', shape=[-1, 3, 224, 224], dtype='float32')
    export_exe = fluid.Executor(fluid.CPUPlace())

    path = 'ddfdfdfd'
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['new_images'],
        target_vars=fetch_targets, executor=export_exe, main_program=export_program, params_filename=params_filename)

    sys.exit(0)
    '''
