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

model_type = "sup_vitb16"
img_size = 224
num_classes = 196

infile = sys.argv[1]


device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionTransformer(model_type=model_type, img_size=img_size, num_classes=num_classes)
state_dict = torch.load(infile)
new_state_dict = {}
for k, v in state_dict.items():
    k = k.replace('enc.', '')
    k = k.replace('last_layer.', '')
    new_state_dict[k] = v
#state_dict = {k.replace('enc.', ''):v for k,v in state_dict.items()}

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
