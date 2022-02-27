import json
import base64
import numpy as np
import pandas as pd
import torch
import pickle as pkl

df=pd.read_csv('feature.tsv',header=None,sep='\t')
features_np={}
for i in range(df.shape[0]):
    k=df.loc[i, 0]
    v=np.frombuffer(base64.b64decode(json.loads(df.loc[i,1])['features']),np.float32).reshape(
        json.loads(df.loc[i,1])['num_boxes'],-1)
    features_np[k]=torch.Tensor(v)
    
torch.save(features_np, "feats.pt")

df_pred=pd.read_csv('predictions.tsv',header=None,sep='\t')
objects={}
for i in range(df_pred.shape[0]):
    k=df.loc[i, 0]
    v=" ".join([i["class"] for i in json.loads(df_pred.loc[i,1])["objects"]])
    objects[k]=v
    
with open('../../data/posrel/data_qa.json') as f:
    data = []
    for row in f.readlines():
        data.append(json.loads(row))
ans2label=pkl.load(open("../../data/trainval_ans2label.pkl", "rb"))

qla=[]
answer_dict={0: 'no', 1: 'yes'}
for idx, i in enumerate(data):
    qla.append({'q': i['question'].split('.')[1], 'o': objects[str(idx//2)+'_600'], 'an': [ans2label[answer_dict[i['label']]]], 'img_id': str(idx//2)+'_600', 's': [1.0]})
    
json.dump(qla, open("oscar_data.json", "w"))