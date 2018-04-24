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
            print("b",true_lbl[vec_loc[...,0],
                               vec_loc[...,1],
                               vec_loc[...,2],:5],
                  "\ttxy",t_xy[slicing],"\ttwh",t_wh[slicing])
        return sample,true_lbl,original,mask,vec_loc,t_xy,t_wh
    
    def true_label(self,true_adj,vec_loc):
        """
        Create true label for training
        """
        mask = self.mask_from_loca(vec_loc)
        true_adj_t = self.b2t_wh(self.b2t_xy(true_adj.copy()))
        true_adj *= np.tile(mask[:,:,:,np.newaxis],[1,1,1,true_adj.shape[-1]])
        
        true_adj_t *= np.tile(mask[:,:,:,np.newaxis],[1,1,1,true_adj_t.shape[-1]])
        t_xy,t_wh = true_adj_t[...,:2],true_adj_t[...,2:4]
        
        return true_adj,mask,vec_loc,t_xy,t_wh
    
    def b2t_xy(self,x):
        x[...,:2] = x[...,:2]-np.floor(x[...,:2])
#         x[...,:2] = np.clip(x[...,:2],1e-4,(1.-1e-4))
#         x[...,:2] = -np.log(1/x[...,:2]-1)
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

class Data_Multi(dataset.Dataset):
    def __init__(self, data_df, testing=False, *args, **kwargs):
        """
        Object detection data generatorcd
        """
        super(Data_Multi, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data_df = data_df
        self.img_ids = list(set(list(data_df["image_id"])))
        self.ids2fn = dict((k, v) for k, v in zip(self.data_df["image_id"], self.data_df["file_name"]))
        self.testing = testing
        self.anc = ANC_ARR

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = Image.open(self.id2url(self.img_ids[idx])).convert("RGB")

        sample = self.transform(img)

        original = self.trans_origin(img)

        img_df = self.data_df[self.data_df.image_id == self.img_ids[idx]].head(50)

        b_xywh = img_df[["true_bb_x", "true_bb_y", "true_bb_w", "true_bb_h"]].as_matrix()
        posi = img_df[["true_grid_x", "true_grid_y", "best_anchor"]].as_matrix().astype(int)
        cls_id = img_df["cate_id_oh"].as_matrix()
        t_xywh = self.b2t_xy(b_xywh)
        t_xywh = self.b2t_wh(t_xywh, posi)

        N = t_xywh.shape[0]

        t_box = np.zeros((FEAT_W, FEAT_H, BOX, 4))
        b_box = np.zeros((FEAT_W, FEAT_H, BOX, 4))
        conf_ = np.zeros((FEAT_W, FEAT_H, BOX, 1))
        cls_ = np.zeros((FEAT_W, FEAT_H, BOX, 1))
        mask = np.zeros((FEAT_W, FEAT_H, BOX, 1))
        cls_mask = np.zeros((FEAT_W, FEAT_H, BOX, 1))

        for i_lbl in range(N):
            t_box[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = t_xywh[i_lbl]
            b_box[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = b_xywh[i_lbl]
            conf_[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1
            cls_[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = cls_id[i_lbl]
            mask[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1
            cls_mask[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1

        if self.testing:
            for i in sample, t_box, conf_, cls_, mask, cls_mask, b_box:
                print(i.shape)

        return sample,original, t_box, conf_, cls_, mask, cls_mask, b_box

    def get_id(self, url):
        return int(url.split("/")[-1].split(".")[0])

    def id2url(self, image_id):
        return IMG + self.ids2fn[image_id]

    def b2t_xy(self, x):
        x[..., :2] = x[..., :2] - np.floor(x[..., :2])
        return x

    def b2t_wh(self, x, posi):
        x[..., 2:4] = np.clip(x[..., 2:4], 1e-2, 12.999)
        lb_s = x.shape[0]
        anc_tile = np.tile(self.anc[np.newaxis, :, :], [lb_s, 1, 1])
        # print(anc_map.shape)
        x[..., 2:4] = x[..., 2:4] / anc_tile[np.eye(5)[posi[:, 2]] == 1]
        x[..., 2:4] = np.log(x[..., 2:4])
        return x

    def true_adj_expand(self, true_adj):
        return np.tile(true_adj[np.newaxis, np.newaxis, np.newaxis, :], [FEAT_W, FEAT_H, BOX, 1])