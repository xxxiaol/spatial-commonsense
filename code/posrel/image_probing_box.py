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

with open('../../data/posrel/data.json') as f:
    data = []
    for row in f.readlines():
        data.append(json.loads(row))
        
labels={'inside':0, 'above':1, 'below':2, 'beside':3}

# evaluate the images with the position of bounding boxes
pred=[]
gold=[]
pred_all=[]
box_idt=[]
for idx, i in enumerate(data):
    box_a=None
    box_b=None
    conf_a=0
    conf_b=0
    for j in objects_all[str(idx)+'_600']:
        if j['class']==i['obj_a'] and j['conf']>conf_a:
            conf_a=j['conf']
            box_a=j['rect']
        if j['class']==i['obj_b'] and j['conf']>conf_b:
            conf_b=j['conf']
            box_b=j['rect']
    if box_a is not None and box_b is not None:
        ax1, ay1, ax2, ay2=box_a
        bx1, by1, bx2, by2=box_b
        cx1=(ax1+ax2)/2
        cy1=(ay1+ay2)/2
        cx2=(bx1+bx2)/2
        cy2=(by1+by2)/2
        f=True  # identifiable        
        if (ax1>=bx1 and ax2<=bx2) and (ay1>=by1 and ay2<=by2):
            pred.append(0)  # inside
        elif cy1>cy2 and cy1-cy2>=np.abs(cx1-cx2):
            pred.append(1)  # above
        elif cy1<cy2 and cy2-cy1>=np.abs(cx1-cx2):
            pred.append(2)  # below
        elif np.abs(cy2-cy1)<np.abs(cx1-cx2):
            pred.append(3)  # beside
        else:
            f=False

        if f:
            gold.append(i['label'])
            box_idt.append(data[idx]['text'])
            pred_all.append(pred[-1])
        else:
            pred_all.append('n')


print('identified:', len(pred)/len(data))
print('f1:', f1_score(gold, pred, average='macro'))
print('acc:', accuracy_score(gold, pred))

with open('box_idt.pkl', 'wb') as f:
    pkl.dump(box_idt, f)