from torch.utils.data import DataLoader,dataset
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from constant import *
import numpy as np

from PIL import Image
from torchvision import transforms

class odData(dataset.Dataset):
    def __init__(self,*args,**kwargs):
        """Object detection data generatorcd """
        super(odData,self).__init__()
        for k,v in kwargs.items():
            setattr(self,k,v)
        self.anc = np.tile(ANC_ARR[np.newaxis,np.newaxis,:,:],[FEAT_W,FEAT_H,1,1])
        
    def __len__(self):
        return len(self.urllist)
        
    def __getitem__(self,idx):
        img=Image.open(self.urllist[idx]).convert("RGB")

        if self.transform:
            sample = self.transform(img)
        
        original = self.trans_origin(img)
        true_lbl,mask,vec_loc = self.true_label(self.true_adj_expand(self.true_adj[idx]),self.vec_loc[idx])
        
        t_xy = self.b2t_xy(self.true_adj[idx])
        t_wh = self.b2t_wh(self.true_adj[idx])
        
        return sample,true_lbl,original,mask,vec_loc,t_xy,t_wh
    
    def true_label(self,true_adj,vec_loc):
        """
        Create true label for training
        """
        mask = self.mask_from_loca(vec_loc)
        true_adj *= np.tile(mask[:,:,:,np.newaxis],[1,1,1,true_adj.shape[-1]])
        return true_adj,mask,vec_loc
    
    def b2t_xy(self,x):
        x = x[...,:2]-np.floor(x[...,:2])
        x = np.clip(x,1e-4,(1.-1e-4))
        x = -np.log(1/x-1)
        return x
    
    def b2t_wh(self,x):
        x = np.clip(x[...,2:4],1e-2,12.999)
        x = x/self.anc
        x = np.log(x)
        return x
    
    def true_adj_expand(self,true_adj):
        return  np.tile(true_adj[np.newaxis,np.newaxis,np.newaxis,:],[FEAT_W,FEAT_H,BOX,1])
    
    def mask_from_loca(self,vec_loc):
        """
        return mask tensor [batch_size,grid_w,grid_h,box] according to grid location
        """
        mask=np.eye(BOX)[vec_loc[:,2:]]
        mask_w=np.eye(FEAT_W)[vec_loc[:,:1]].reshape(FEAT_H,1,1)
        mask_h=np.eye(FEAT_H)[vec_loc[:,1:2]].reshape((1,FEAT_H,1))
        
        mask = np.tile(mask,[FEAT_W,FEAT_H,1])
        mask *= np.tile(mask_w,[1,FEAT_H,BOX])
        mask *= np.tile(mask_h,[FEAT_W,1,BOX])
        return mask