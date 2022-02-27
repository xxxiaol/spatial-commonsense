# Adapted from code of Ryan Murdoch, @advadnoun on Twitter.

import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import kornia

import PIL
import matplotlib.pyplot as plt

import os
import random
import imageio
from IPython import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pickle as pkl
import json

clip_path=''
taming_path=''

import sys
sys.path.append(".")
sys.path.append(clip_path)
sys.path.append(taming_path)

with open('../../data/posrel/data.json') as f:
    data = []
    for row in f.readlines():
        data.append(json.loads(row))
        
text_all=[i['text'] for i in data]
idx_all=np.arange(len(text_all))

# text_input = "A toy accordian" #@param {type:"string"}
w0 = 1 #@param {type:"slider", min:-5, max:5, step:0.1}
text_to_add = "" #@param {type:"string"}
w1 = 0 #@param {type:"slider", min:-5, max:5, step:0.1}
img_enc_path = "" #@param {type:"string"}
w2 = 1 #@param {type:"slider", min:-5, max:5, step:0.1}
ne_img_enc_path = "" #@param {type:"string"}
w3 = 0 #@param {type:"slider", min:-5, max:5, step:0.1}

# How to weight the 2 texts (w0 and w1) and the images (w3 & w3)

text_other = '''incoherent, confusing, cropped, watermarks'''


im_shape = [512, 512, 3]
sideX, sideY, channels = im_shape
batch_size = 3

from CLIP import clip
perceptor, preprocess = clip.load('ViT-B/32', jit=False)
perceptor.eval()

clip.available_models()

perceptor.visual.input_resolution

scaler = 1

def displ(img, img_name, pre_scaled=True):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    imageio.imwrite('../../data/posrel/images/' + img_name + '.png', np.array(img))
    return

# os.chdir('content/taming-transformers')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec

config16384 = load_config(taming_path+"logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
model16384 = load_vqgan(config16384, ckpt_path=taming_path+"logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)


torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()


        # self.normu = torch.nn.Parameter(o_i2.cuda().clone())

        self.normu = .5*torch.randn(len(text_input), 256, sideX//16, sideY//16).cuda()
        
        self.normu = torch.nn.Parameter(torch.sinh(1.9*torch.arcsinh(self.normu)))

    def forward(self):


        return self.normu.clip(-6, 6)
      

def model(x):
    o_i2 = x
    o_i3 = model16384.post_quant_conv(o_i2)
    i = model16384.decoder(o_i3)
    return i

 
def augment(into, cutn=32):

    into = torch.nn.functional.pad(into, (sideX//2, sideX//2, sideX//2, sideX//2), mode='constant', value=0)


    into = augs(into)

    p_s = []
    for ch in range(cutn):
        size = int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * sideX)

        if ch > cutn - 4:
            size = int(sideX*1.4)
        offsetx = torch.randint(0, int(sideX*2 - size), ())
        offsety = torch.randint(0, int(sideX*2 - size), ())
        apper = into[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(apper, (int(224*scaler), int(224*scaler)), mode='bilinear', align_corners=True)
        p_s.append(apper)
    into = torch.cat(p_s, 0)

    into = into + up_noise*torch.rand((into.shape[0], 1, 1, 1)).cuda()*torch.randn_like(into, requires_grad=False)
    return into

def checkin(loss, itt):
    global up_noise

    with torch.no_grad():

        alnot = model(lats()).float()
        alnot = augment((((alnot).clip(-1, 1) + 1) / 2), cutn=1)

        alnot = (model(lats()).cpu().clip(-1, 1) + 1) / 2

        for i, allls in enumerate(alnot.cpu()):
            displ(allls, str(batch_idx[i])+'_'+str(itt) )


def ascend_txt():
    global up_noise
    out = model(lats())

    into = augment((out.clip(-1, 1) + 1) / 2)  # [96, 3, 224, 224]
    into = nom(into)  # [96, 3, 224, 224]

    iii = perceptor.encode_image(into)
    iii = iii.reshape(32, len(text_input), iii.shape[-1]).permute(1, 0, 2)

    q = w0*t + w1*text_add + w2*img_enc + w3*ne_img_enc
    q = q / q.norm(dim=-1, keepdim=True)
    q = q.unsqueeze(1)
    
#     print(iii.shape, q.shape)  # torch.Size([3, 32, 512]) torch.Size([3, 1, 512])

    all_s = torch.cosine_similarity(q, iii, -1)

    return [0, -10*all_s + 5 * torch.cosine_similarity(t_not, iii, -1)]
  
def train(itt):
    global dec
    global up_noise

    loss1 = ascend_txt()
    loss = loss1[0] + loss1[1]
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if torch.abs(lats()).max() > 5:
        for g in optimizer.param_groups:
            g['weight_decay'] = dec
    else:
        for g in optimizer.param_groups:
            g['weight_decay'] = 0

    if itt % 100 == 0:
        checkin(loss1, itt)

#     print('up_noise', up_noise)
#     for g in optimizer.param_groups:
#         print(g['lr'], 'lr', g['weight_decay'], 'decay')


def loop():
    global itt
    for asatreat in range(601):
        train(itt)
        itt+=1

        
n_batch=(len(text_all)+batch_size-1)//batch_size
for i in range(n_batch):
    text_input=text_all[i*batch_size:(i+1)*batch_size]
    batch_idx=idx_all[i*batch_size:(i+1)*batch_size]
    print(text_input, batch_idx)

    dec = .1

    lats = Pars().cuda()
    mapper = [lats.normu]
    optimizer = torch.optim.AdamW([{'params': mapper, 'lr': .5}], weight_decay=dec)
    eps = 0

    t = 0
    if text_input != '':
        tx = clip.tokenize(text_input)
        t = perceptor.encode_text(tx.cuda()).detach().clone()

    text_add = 0
    if text_to_add != '':
        text_add = clip.tokenize(text_to_add)
        text_add = perceptor.encode_text(text_add.cuda()).detach().clone()

    t_not = clip.tokenize(text_other)
    t_not = perceptor.encode_text(t_not.cuda()).detach().clone()


    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    img_enc = 0
    if img_enc_path != '':
        img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
        img_enc = nom(img_enc)
        img_enc = perceptor.encode_image(img_enc.cuda()).detach().clone()

    ne_img_enc = 0
    if ne_img_enc_path != '':
        ne_img_enc = (torch.nn.functional.interpolate(torch.tensor(imageio.imread(ne_img_enc_path)).unsqueeze(0).permute(0, 3, 1, 2), (224, 224)) / 255).cuda()[:,:3]
        ne_img_enc = nom(ne_img_enc)
        ne_img_enc = perceptor.encode_image(ne_img_enc.cuda()).detach().clone()



    augs = torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(24, (.1, .1))
    ).cuda()


    up_noise = .11



    itt = 0


    with torch.no_grad():
        al = (model(lats()).cpu().clip(-1, 1) + 1) / 2

    loop()
