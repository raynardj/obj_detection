import numpy as np
import pandas as pd
import torch
import json
import os
from constant_char import *

HOME = os.environ['HOME']+"/"

# Terminus
if HOME == "/home/zhangxiaochen/":
    DATA = "/data/forge/"
    IMG = DATA+"char_detect/"
    # ANN = DATA+"annotations/instances_train2017.json"
# Macbook
elif HOME == "/Users/zhangxiaochen/":
    DATA = "/data/forge/"
    IMG = DATA+"char_detect/"
    # ANN = DATA+"annotations/instances_val2017.json"
# Jupiter AWS
elif HOME == "/home/ubuntu/":
    DATA = "/data/forge/"
    IMG = DATA+"char_detect/"
    # ANN = DATA+"annotations/instances_val2017.json"

# --------------------------------------------------

SIZE = 320 # 13 * 32

HEIGHT = SIZE
WIDTH = SIZE

SCALE =32

BOX = 2
CLS_LEN = len(chars)
VEC_LEN = 5 + CLS_LEN

FEAT_W = int(HEIGHT/SCALE)
FEAT_H = int(WIDTH/SCALE)

# --------------------------------------------------


ANCHORS = [0.57273, 0.677385, 
           1.87446, 2.06253,]

DN121 = HOME+".torch/models/dn121_4.pkl"

anchor = torch.from_numpy(np.array(ANCHORS,dtype=np.float)).view(1,1,1,BOX,2).type(torch.FloatTensor)

ANC_ARR = np.array(ANCHORS).reshape(BOX,2)

GRID_MAP=np.concatenate([
    np.tile(np.arange(FEAT_W).reshape(1,-1,1,1,1),[1,1,FEAT_H,BOX,1]),
    np.tile(np.arange(FEAT_H).reshape(1,1,-1,1,1),[1,FEAT_W,1,BOX,1]),]
    ,axis=-1)

idx2name = IDX2CHARS
id2idx = dict(enumerate(range(CLS_LEN)))
idx2id = dict(enumerate(range(CLS_LEN)))