from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import torch
import pandas as pd
from datetime import datetime
from constant import *

def plot_bb(img,bbdf):
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    for row_ in bbdf.iterrows():
        row = row_[1]
        # format of the bb: x, y, width, height
        rect = patches.Rectangle((row["x"],row["y"]),row["w"],row["h"],linewidth=1,edgecolor='g',facecolor='none')

        ax.add_patch(rect)
        # format of bb 
        ax.text(row["x"],row["y"],idx2name[id2idx[int(row["cate"])]]+"%.3f"%row["conf"],dict({"color":"#ff0000"}))
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

#plot_bb(img,data_to_df(loss.t2b(y_pred[0])))