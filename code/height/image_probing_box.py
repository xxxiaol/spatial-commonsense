import json
import base64
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.metrics import f1_score, accuracy_score

# load all the objects detected by VinVL in each image
df_pred=pd.read_csv('predictions.tsv',header=None,sep='\t')
json.loads(df_pred.loc[0,1])['objects'][0].keys()

objects_all={}
for i in range(df_pred.shape[0]):
    k=df_pred.loc[i, 0]
    v=json.loads(df_pred.loc[i,1])["objects"]
    objects_all[k]=v

with open('../../data/height/data.json') as f:
    height_data = []
    for row in f.readlines():
        height_data.append(json.loads(row))

depth=pkl.load(open("height_depth.pkl", "rb"))

# evaluate the images with the height of bounding boxes
pred=[]
gold=[]
pred_all=[]
box_idt_dict={}
for idx, i in enumerate(height_data):
    height_a=0
    height_b=0
    conf_a=0
    conf_b=0
    for j in objects_all[str(idx)+'_600']:
        if j['class']==i['obj_a'] and j['conf']>conf_a:
            conf_a=j['conf']
            x1, y1, x2, y2=j['rect']
            height_a=y2-y1
            coordinate_a=[int(x1), int(y1), int(x2), int(y2)]

        if j['class']==i['obj_b'] and j['conf']>conf_b:
            conf_b=j['conf']
            x1, y1, x2, y2=j['rect']
            height_b=y2-y1
            coordinate_b=[int(x1), int(y1), int(x2), int(y2)]

    if height_a>0 and height_b>0:
        x1, y1, x2, y2=coordinate_a
        depth_a=np.mean(depth[idx][y1:y2, x1:x2])
        x1, y1, x2, y2=coordinate_b
        depth_b=np.mean(depth[idx][y1:y2, x1:x2])

        height_a=height_a*depth_a
        height_b=height_b*depth_b
        if height_a>height_b:
            pred.append(1)
            pred_all.append(1)
        else:
            pred.append(0) 
            pred_all.append(0)
        gold.append(i['label'])

        if not i['obj_a'] in box_idt_dict:
            box_idt_dict[i['obj_a']]=set()
        box_idt_dict[i['obj_a']].add(i['obj_b'])
    else:
        pred_all.append('n')

print('identified:', len(pred)/len(height_data))
print('f1:', f1_score(gold, pred, average='macro'))
print('acc:', accuracy_score(gold, pred))

with open('box_idt.pkl', 'wb') as f:
    pkl.dump(box_idt_dict, f)