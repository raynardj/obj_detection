from matplotlib import pyplot as plt
import matplotlib.patches as patches

from datetime import datetime
from constant import *
import matplotlib.text as text
import torch
import pandas as pd
import numpy as np

COLORS = ["#ff0000","#ffff00","#ff00ff","#00ffff","#00ff00","#0000ff","#ffffff"]
bx_grid=torch.arange(0,BOX).unsqueeze(0).unsqueeze(0).repeat(1,FEAT_W,FEAT_H,1)

def plot_bb(img,bbdf):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    if type(bbdf).__name__=="DataFrame":
        for row_ in bbdf.iterrows():
            row = row_[1]
            # format of the bb: x, y, width, height
            c = np.random.choice(COLORS)
            rect = patches.Rectangle((row["x"],row["y"]),row["w"],row["h"],linewidth=1,edgecolor=c,facecolor='none')
    
            ax.add_patch(rect)
            # format of bb 
            ax.text(row["x"],
                    row["y"],
                    idx2name[id2idx[int(row["cate"])]]+"%.3f"%row["conf"],
                    dict({"color":c}))
    save = fig.savefig("/data/bbsample/%s.jpg"%(datetime.now().strftime("%H%M%S")))
    
def data_to_df(y_pred,head=10):
    c=y_pred[...,4:5].repeat(1,1,1,CLS_LEN)*y_pred[...,5:]
    val_max, idx_max = torch.max(c.view(-1,CLS_LEN), dim=-1)

    df_lbl = pd.DataFrame({"conf":val_max.data.numpy(),"cate":idx_max.data.numpy()})

    bbmax = y_pred[...,:4].contiguous().view(-1,4).data.numpy()*32
    df_bbox = pd.DataFrame(bbmax,columns=["x","y","w","h"])
    
    df_bbox["x"]=df_bbox["x"]-(df_bbox["w"]/2)
    df_bbox["y"]=df_bbox["y"]-(df_bbox["h"]/2)
    df_bbox["x"][df_bbox["x"]<0]=0.
    df_bbox["y"][df_bbox["y"]<0]=0.
    
    return pd.concat([df_lbl,df_bbox],axis=1).sort_values(by="conf",ascending=False).head(head).reset_index()

def data_to_df_bmark(y_pred,head=5,bm=.50):
    bs = y_pred.size()[0]
    val, idx = torch.max(y_pred[..., 4], dim=-1)
    pick = (idx.unsqueeze(-1) == bx_grid.repeat(bs, 1, 1, 1).long())

    cls_val, cls_idx = torch.max(y_pred[..., 5:], dim=-1)
    cls_ = cls_idx[pick].view(-1, 1).float()

    conf_ = y_pred[..., 4][pick].view(-1, 1)
    box_ = y_pred[..., :4][pick.unsqueeze(-1).repeat(1, 1, 1, 1, 4)].view(-1, 4) * 32

    combine = torch.cat([box_, conf_, cls_], dim=-1).numpy()

    df_bbox = pd.DataFrame(combine[combine[..., 4] > bm], columns=["x", "y", "w", "h", "conf", "cate"])

    df_bbox["x"] = df_bbox["x"] - (df_bbox["w"] / 2)
    df_bbox["y"] = df_bbox["y"] - (df_bbox["h"] / 2)
    df_bbox["x"][df_bbox["x"] < 0] = 0.
    df_bbox["y"][df_bbox["y"] < 0] = 0.

    return df_bbox