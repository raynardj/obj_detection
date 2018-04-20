import numpy as np
import torch

# HOME = "/home/zhangxiaochen/"
# DATA = "/terminus/coco/"
# IMG = DATA+"train2017/"
# ANN = DATA+"annotations/instances_train2017.json"


HOME = "/Users/zhangxiaochen/"
DATA = "/data/coco/"
IMG = DATA+"val2017/"
ANN = DATA+"annotations/instances_val2017.json"
# --------------------------------------------------

SIZE = 416 # 13 * 32

HEIGHT = 416
WIDTH = 416

SCALE =32

BOX = 5
VEC_LEN = 85

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