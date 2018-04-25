from matplotlib import pyplot as plt
import matplotlib.patches as patches

from datetime import datetime
from constant import *
import matplotlib.text as text
import torch
import pandas as pd
import numpy as np

COLORS = ["#ff0000","#ffff00","#ff00ff","#00ffff","#00ff00","#0000ff","#ffffff"]

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
    y_pred=y_pred.detach()
    conf_, idx = torch.max(y_pred[..., 4], dim=-1)
    conf_b = (conf_>bm)
    if conf_b.sum().data[0]==0:
        return None
    else:
        conf = y_pred[...,4][conf_b]

        conf_cls = y_pred[...,5:][conf_b].view(-1,CLS_LEN)

        val_max, idx_max = torch.max(conf_cls, dim=-1)
    
        df_lbl = pd.DataFrame({"conf":conf.data.numpy(),"cate":idx_max.data.numpy()})

        bbmax = y_pred[...,:4][conf_b].contiguous().view(-1,4).data.numpy()*32
        df_bbox = pd.DataFrame(bbmax,columns=["x","y","w","h"])
    
        df_bbox["x"]=df_bbox["x"]-(df_bbox["w"]/2)
        df_bbox["y"]=df_bbox["y"]-(df_bbox["h"]/2)
        df_bbox["x"][df_bbox["x"]<0]=0.
        df_bbox["y"][df_bbox["y"]<0]=0.
    
        return pd.concat([df_lbl,df_bbox],axis=1) # .sort_values(by="conf",ascending=False).head(head).reset_index()