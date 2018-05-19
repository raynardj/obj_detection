import numpy as np
import pandas as pd
import torch
import json
import os
from constant_char import *


TRAIN_CLS = True
REBUILD_DATA = False
EXPERIMENT = False

# dir and locations

HOME = os.environ['HOME']+"/"

DATA = "/data/forge/"

# Terminus
if HOME == "/home/zhangxiaochen/":
    DATA = "/data/forge/"
    IMG_EPT = "/data/coco/train2017/"

# Macbook
elif HOME[:6]=="/Users":
    EXPERIMENT = True
    DATA = "/data/forge/"
    IMG_EPT = "/data/coco/val2017/"

# Jupiter AWS
elif HOME == "/home/paperspace/":
    EXPERIMENT = False
    DATA = "/data/forge/"
    IMG_EPT = "/data/train2017/"

IMG = DATA+"char_detect/"
IMG_CLS = DATA+"char_detect_cn/"

# ANN = DATA+"annotations/instances_val2017.json"

ANN = DATA+"char_lbl.csv"
ANN_CLS = DATA+"char_lbl_cn.csv"

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

LBD_COORD=1
LBD_OBJ=5
LBD_NOOBJ=1
LBD_CLS=1

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