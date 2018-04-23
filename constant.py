import numpy as np
import pandas as pd
import torch
import json
import os

HOME = os.environ['HOME']+"/"

# Terminus
if HOME == "/home/zhangxiaochen/":
    DATA = "/terminus/coco/"
    IMG = DATA+"train2017/"
    ANN = DATA+"annotations/instances_train2017.json"
# Macbook
elif HOME == "/Users/zhangxiaochen/":
    DATA = "/data/coco/"
    IMG = DATA+"val2017/"
    ANN = DATA+"annotations/instances_val2017.json"
# Jupiter AWS
elif HOME == "/home/ubuntu/":
    DATA = "/data/"
    IMG = DATA+"val2017/"
    ANN = DATA+"annotations/instances_val2017.json"

# --------------------------------------------------

SIZE = 416 # 13 * 32

HEIGHT = 416
WIDTH = 416

SCALE =32

BOX = 5
CLS_LEN = 80
VEC_LEN = BOX + CLS_LEN

FEAT_W = int(HEIGHT/SCALE)
FEAT_H = int(WIDTH/SCALE)

# --------------------------------------------------


ANCHORS = [0.57273, 0.677385, 
           1.87446, 2.06253, 
           3.33843, 5.47434, 
           7.88282, 3.52778, 
           9.77052, 9.16828]

DN121 = HOME+".torch/models/dn121.pkl"

anchor = torch.from_numpy(np.array(ANCHORS,dtype=np.float)).view(1,1,1,5,2).type(torch.FloatTensor)

ANC_ARR = np.array(ANCHORS).reshape(5,2)

GRID_MAP=np.concatenate([
    np.tile(np.arange(FEAT_W).reshape(1,-1,1,1,1),[1,1,FEAT_H,5,1]),
    np.tile(np.arange(FEAT_H).reshape(1,1,-1,1,1),[1,FEAT_W,1,5,1]),]
    ,axis=-1)


jsfile = open(ANN).read()

jsdict = json.loads(jsfile)

jsdict.keys()

cat_df = pd.DataFrame(jsdict["categories"])
idx2name = dict(zip(cat_df["id"],cat_df["name"]))
id2idx = dict(enumerate(cat_df["id"]))
idx2id = dict((v,k) for k,v in enumerate(cat_df["id"]))