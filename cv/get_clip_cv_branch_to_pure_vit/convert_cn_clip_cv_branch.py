import torch
import sys
import numpy as np
from PIL import Image
from src.models.vit_backbones.vit import VisionTransformer
import torchvision as tv
import json
import os
from tqdm import tqdm
import torch
import collections

model_type = "sup_vitb16"
img_size = 224
num_classes = 512

infile = sys.argv[1]


device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionTransformer(model_type=model_type, img_size=img_size, num_classes=num_classes, pre_norm=True)
state_dict = torch.load(infile)
#print(state_dict['state_dict'].keys())
new_state_dict = {}

if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
torch_np_ckpt = {}
print('=vit=')
for k, v in model.state_dict().items():
    print(k, v.shape)
for k, v in state_dict.items():
    if 'visual' not in k:
        continue
    k = k.replace('module.', '')
    k = k.replace('visual.', '')
    v = v.cpu().detach().numpy()
    v_shape = v.shape
    if 'in_proj_weight' in k:
        kk_q = k.replace('in_proj_weight', 'query.weight')
        kk_k = k.replace('in_proj_weight', 'key.weight')
        kk_v = k.replace('in_proj_weight', 'value.weight')
        vv = np.array_split(v, 3, axis=0)
        print('v split shape {} -> {}'.format(v_shape, np.array(vv).shape))
        kk_q_val, kk_k_val, kk_v_val = np.array(vv)
        torch_np_ckpt[kk_q] = kk_q_val
        torch_np_ckpt[kk_k] = kk_k_val
        torch_np_ckpt[kk_v] = kk_v_val

        print('trans {} shape {} -> {} {}; {} {}; {} {}'.format(
            k, v.shape, kk_k, kk_k_val.shape, kk_q, kk_q_val.shape, kk_v, kk_v_val.shape))
        continue
    elif 'in_proj_bias' in k:
        kk_q = k.replace('in_proj_bias', 'query.bias')
        kk_k = k.replace('in_proj_bias', 'key.bias')
        kk_v = k.replace('in_proj_bias', 'value.bias')
        vv = np.array_split(v, 3)
        print('v spllit shape {} -> {}'.format(v_shape, np.array(vv).shape))
        kk_q_val, kk_k_val, kk_v_val = np.array(vv)

        torch_np_ckpt[kk_q] = kk_q_val
        torch_np_ckpt[kk_k] = kk_k_val
        torch_np_ckpt[kk_v] = kk_v_val

        print('trans {} shape {} -> {} {}; {} {}; {} {}'.format(
            k, v.shape, kk_k, kk_k_val.shape, kk_q, kk_q_val.shape, kk_v, kk_v_val.shape))
        continue
    torch_np_ckpt[k] = v
clip_cv_to_vit = {
    'class_embedding': 'transformer.embeddings.cls_token',
    'positional_embedding': 'transformer.embeddings.position_embeddings',
    'conv1.weight': 'transformer.embeddings.patch_embeddings.weight',
    'ln_pre.weight': 'transformer.encoder.pre_encoder_norm.weight',
    'ln_pre.bias': 'transformer.encoder.pre_encoder_norm.bias',
    'ln_post.weight': 'transformer.encoder.encoder_norm.weight',
    'ln_post.bias': 'transformer.encoder.encoder_norm.bias',
    }

transformer_part = {
    'transformer.resblocks.{}.ln_1.weight': 'transformer.encoder.layer.{}.attention_norm.weight',
    'transformer.resblocks.{}.ln_1.bias': 'transformer.encoder.layer.{}.attention_norm.bias',
    'transformer.resblocks.{}.attn.key.weight': 'transformer.encoder.layer.{}.attn.key.weight',
    'transformer.resblocks.{}.attn.key.bias': 'transformer.encoder.layer.{}.attn.key.bias',
    'transformer.resblocks.{}.attn.query.weight': 'transformer.encoder.layer.{}.attn.query.weight',
    'transformer.resblocks.{}.attn.query.bias': 'transformer.encoder.layer.{}.attn.query.bias',
    'transformer.resblocks.{}.attn.value.weight': 'transformer.encoder.layer.{}.attn.value.weight',
    'transformer.resblocks.{}.attn.value.bias': 'transformer.encoder.layer.{}.attn.value.bias',
    'transformer.resblocks.{}.attn.out_proj.weight': 'transformer.encoder.layer.{}.attn.out.weight',
    'transformer.resblocks.{}.attn.out_proj.bias': 'transformer.encoder.layer.{}.attn.out.bias',
    'transformer.resblocks.{}.ln_2.weight': 'transformer.encoder.layer.{}.ffn_norm.weight',
    'transformer.resblocks.{}.ln_2.bias': 'transformer.encoder.layer.{}.ffn_norm.bias',
    'transformer.resblocks.{}.mlp.c_fc.weight': 'transformer.encoder.layer.{}.ffn.fc1.weight',
    'transformer.resblocks.{}.mlp.c_fc.bias': 'transformer.encoder.layer.{}.ffn.fc1.bias',
    'transformer.resblocks.{}.mlp.c_proj.weight': 'transformer.encoder.layer.{}.ffn.fc2.weight',
    'transformer.resblocks.{}.mlp.c_proj.bias': 'transformer.encoder.layer.{}.ffn.fc2.bias',
    }
for i in range(12):
    for k, v in transformer_part.items():
        k = k.format(i)
        v = v.format(i)
        clip_cv_to_vit[k] = v
clip_cv_to_vit['proj'] = 'head.weight'
print(clip_cv_to_vit)

print('=clip=')
out_torch_np_ckpt = collections.OrderedDict()
for k, v in torch_np_ckpt.items():
    v = v.astype('float32')
    if k in clip_cv_to_vit.keys():
        name = clip_cv_to_vit[k]
        if 'transformer.embeddings.position_embeddings' == name:
            out_torch_np_ckpt[name] = np.expand_dims(v, axis=0)
        elif 'transformer.embeddings.cls_token' == name:
            out_torch_np_ckpt[name] = np.expand_dims(np.expand_dims(v, axis=0), axis=0)
        elif 'head.weight' == name:
            out_torch_np_ckpt[name] = np.transpose(v, (1, 0))
        else:
            out_torch_np_ckpt[name] = v
        print('trans {} shape {} -> {}'.format(k, v.shape, clip_cv_to_vit[k]))
    else:
        print('[{}] no found'.format(k))
out_torch_np_ckpt['transformer.embeddings.patch_embeddings.bias'] = np.zeros((768), dtype=np.float32)
out_torch_np_ckpt['head.bias'] = np.zeros((num_classes), dtype=np.float32)

out_torch_np_ckpt_tensor = collections.OrderedDict()
for k, v in out_torch_np_ckpt.items():
    out_torch_np_ckpt_tensor[k] = torch.tensor(v)

model.load_state_dict(out_torch_np_ckpt_tensor)
torch.save(out_torch_np_ckpt_tensor, 'vit_b16_from_cn_clip.pt')
print(len(torch_np_ckpt))
sys.exit(0)

'''
for k, v in state_dict.items():
    k = k.replace('enc.', '')
    k = k.replace('last_layer.', '')
    new_state_dict[k] = v
#state_dict = {k.replace('enc.', ''):v for k,v in state_dict.items()}
for k, v in state_dict['state_dict'].items():
    if 'module.visual' in k:
        k = k.replace('module.visual.', '')
        new_state_dict[k] = v
'''

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()


json_file = 'data/stanfordcars/val.json'
with open(json_file, 'r') as f:
    test_data = json.load(f)
#image_file = 'data/stanfordcars/car_ims/016173.jpg'
num = len(test_data)
count = 0
for image_file, label in tqdm(test_data.items(), total=num):
    image_file = os.path.join('data/stanfordcars', image_file)
    resize_dim = 256
    crop_dim = 224
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(resize_dim),
            tv.transforms.CenterCrop(crop_dim),
            tv.transforms.ToTensor(),
            normalize,
        ]
    )

    #image = Image.open(image_file)
    image = tv.datasets.folder.default_loader(image_file)
    '''
    img = image.resize((224,224), resample=Image.BICUBIC)
    img = np.array(img).astype("float32") / 255.0
    print('img shape', img.shape)
    img -= [0.48145466, 0.4578275, 0.4082107]
    img /= [0.26862954, 0.26130258, 0.27577711]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    print('img shape', img.shape)
    '''
    img = transform(image)
    #print('img shape', img.shape)
    input_tensor_1 = img.unsqueeze(0).to(device)
    #input_tensor_1 = torch.tensor(img).to(device)
    #print('input_tensor_1 shape', input_tensor_1.shape)

    res = model.forward(input_tensor_1)
    prob = torch.softmax(res, axis=-1)
    #print('score={}, pred_id={}, label={}, match={}'.format(prob.max(), prob.argmax() + 1, label, (prob.argmax() + 1) == label))
    count += int((prob.argmax() + 1) == label)
print('precision={}'.format(count / float(num)))
