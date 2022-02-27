from __future__ import absolute_import, division, print_function
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

import networks
from utils import download_model_if_doesnt_exist
from layers import disp_to_depth

from tqdm import tqdm
import pickle as pkl


model_name = "mono+stereo_640x192"

download_model_if_doesnt_exist(model_name)
encoder_path = os.path.join("models", model_name, "encoder.pth")
depth_decoder_path = os.path.join("models", model_name, "depth.pth")

# LOADING PRETRAINED MODEL
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
depth_decoder.load_state_dict(loaded_dict)

encoder.eval()
depth_decoder.eval();


# Loading the test image and preprocessing

depth_list=[]
for i in tqdm(range(486)):
    image_path = "../../data/height/"+str(i)+"_600.png"
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_image_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)
    scaled_disp, depth = disp_to_depth(disp_resized, 0.1, 10)
    STEREO_SCALE_FACTOR = 5.4
    metric_depth = STEREO_SCALE_FACTOR * depth.squeeze().cpu().numpy()

    depth_list.append(metric_depth)


# In[103]:


with open('height_depth.pkl', 'wb') as f:
    pkl.dump(depth_list, f)