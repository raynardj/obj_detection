from torch.utils.data import dataset

from constant import *
import numpy as np

from PIL import Image
from torchvision import transforms

class odData(dataset.Dataset):
    def __init__(self,testing=False,*args,**kwargs):
        """Object detection data generatorcd """
        super(odData,self).__init__()
        for k,v in kwargs.items():
            setattr(self,k,v)
        self.testing=testing
        self.anc = np.tile(ANC_ARR[np.newaxis,np.newaxis,:,:],[FEAT_W,FEAT_H,1,1])
        
    def __len__(self):
        return len(self.urllist)
        
    def __getitem__(self,idx):
        img=Image.open(self.urllist[idx]).convert("RGB")

        sample = self.transform(img)
        
        original = self.trans_origin(img)
        
        true_lbl,mask,vec_loc,t_xy,t_wh = self.true_label(self.true_adj_expand(self.true_adj[idx]),
                                                          self.vec_loc[idx])
        if self.testing:
            slicing = vec_loc[...,0],vec_loc[...,1],vec_loc[...,2]
            print("b",true_lbl[vec_loc[...,0],vec_loc[...,1],vec_loc[...,2],:5],"\ttxy",t_xy[slicing],"\ttwh",t_wh[slicing])
        return sample,true_lbl,original,mask,vec_loc,t_xy,t_wh
    
    def true_label(self,true_adj,vec_loc):
        """
        Create true label for training
        """
        mask = self.mask_from_loca(vec_loc)
        true_adj_t = self.b2t_wh(self.b2t_xy(true_adj))
        true_adj *= np.tile(mask[:,:,:,np.newaxis],[1,1,1,true_adj.shape[-1]])
        
        true_adj_t *= np.tile(mask[:,:,:,np.newaxis],[1,1,1,true_adj_t.shape[-1]])
        t_xy,t_wh = true_adj_t[...,:2],true_adj_t[...,2:4]
        
        return true_adj,mask,vec_loc,t_xy,t_wh
    
    def b2t_xy(self,x):
        x[...,:2] = x[...,:2]-np.floor(x[...,:2])
        x[...,:2] = np.clip(x[...,:2],1e-4,(1.-1e-4))
        x[...,:2] = -np.log(1/x[...,:2]-1)
        return x
    
    def b2t_wh(self,x):
        x[...,2:4] = np.clip(x[...,2:4],1e-2,12.999)
        x[...,2:4] = x[...,2:4]/self.anc
        x[...,2:4] = np.log(x[...,2:4])
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