import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from constant import *

AVG=False
mse = nn.MSELoss(size_average=AVG)
ce = nn.CrossEntropyLoss(size_average=AVG)

class t2b(nn.Module):
    def __init__(self):
        super(t2b,self).__init__()
        self.grid=nn.Parameter(data=torch.from_numpy(GRID_MAP,).type(torch.FloatTensor),
                               requires_grad=False)
        self.anc=nn.Parameter(torch.from_numpy(ANC_ARR).type(torch.FloatTensor),
                              requires_grad=False)
        
    def forward(self,x,pred=False):
        bs=x.size()[0]
        x=x.float()
        xy=F.sigmoid(x[...,:2])+self.grid.repeat(bs,1,1,1,1)
        wh=torch.exp(x[...,2:4])*self.anc.repeat(bs,FEAT_W,FEAT_H,1,1)
        others=F.sigmoid(x[...,4:])
        
        x = torch.cat([xy,wh,others],dim=-1)
        return x
    
class yloss_basic(nn.Module):
    def __init__(self,lbd_coord,lbd_noobj,lbd_cls,testing):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yloss_basic,self).__init__()
        self.t2b=t2b()
        self.lbd_coord = lbd_coord
        self.lbd_noobj = lbd_noobj
        self.lbd_cls = lbd_cls
        self.testing = testing
        
    def calc_iou(self,y_true,y_pred):
        """
        IOU score:Intersection over union
        return: IOU score on each grid(width,height) and anchor box
        """
        left_up = torch.max(y_true[...,:2]-y_true[...,2:4]/2,y_pred[...,:2]-y_pred[...,2:4]/2)
        right_down = torch.min(y_true[...,:2]+y_true[...,2:4]/2,y_pred[...,:2]+y_pred[...,2:4]/2)
    
        inter = torch.clamp(right_down-left_up,0,None).prod(dim=-1)
        union = y_true[...,2:4].prod(dim=-1) + y_pred[...,2:4].prod(dim=-1)-inter
    
        return (inter/union).unsqueeze(-1)
    
    def loss_mask(self,y_true,y_pred,lbl_mask):
        ioumap = self.calc_iou(y_true,y_pred)

        mask = lbl_mask.unsqueeze(-1).repeat(1,1,1,1,VEC_LEN)
    
        mask2 = (ioumap<0.5).unsqueeze(-1).repeat(1,1,1,1,VEC_LEN).float()
        mask2[...,:4]=0
        mask2[...,5:]=0
        mask2-=mask
        mask2=torch.clamp(mask2,0,1)
    
        return mask.detach(),mask2.detach(),ioumap.detach()
    
class yolo3_loss_on_t(yloss_basic):
    def __init__(self,lbd_coord=5,lbd_noobj=.5,lbd_cls=1,testing=False):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yolo3_loss_on_t,self).__init__(lbd_coord,lbd_noobj,lbd_cls,testing)
    
    def forward(self,y_pred,t_box, conf_, cls_, mask, cls_mask, b_box):
        bs = t_box.size()[0]

        t_box = t_box.float()
        b_box = b_box.float()
        conf_=conf_.float()

        y_pred_b = self.t2b(y_pred.float())
        ioumap = self.calc_iou(b_box,y_pred_b).detach()

        y_pred = y_pred.float()
        
        y_pred[...,4:5] = F.sigmoid(y_pred[...,4:5])

        mask2 = (mask==0)*(ioumap<.5)
        
        y_pred_xy = F.sigmoid(y_pred[...,:2])
        y_pred_wh = y_pred[...,2:4]
        y_pred_conf = y_pred[...,4:5]
        y_pred_cls = y_pred[...,5:]

        if self.testing:
            pass
        
        mask_slice = mask.float()
        mask2_slice = mask2.float()
        oh_slice = (cls_mask.repeat(1, 1, 1, 1, CLS_LEN) == 1)
        idx_slice = (cls_mask == 1)

        loss_noobj = mse(y_pred_conf*mask2_slice,conf_*mask2_slice)/2.0 * self.lbd_noobj

        loss_obj = mse(y_pred_conf*mask_slice, conf_*mask_slice)/2.0

        loss_x = mse(y_pred_xy[...,0:1]*mask_slice,t_box[...,0:1]*mask_slice)/2.0 * self.lbd_coord
        loss_y = mse(y_pred_xy[...,1:2]*mask_slice,t_box[...,1:2]*mask_slice)/2.0 * self.lbd_coord
        loss_w = mse(y_pred_wh[...,0:1]*mask_slice,t_box[...,2:3]*mask_slice)/2.0 * self.lbd_coord
        loss_h = mse(y_pred_wh[...,1:2]*mask_slice,t_box[...,3:4]*mask_slice)/2.0 * self.lbd_coord

        loss_cls = ce((y_pred_cls[oh_slice]).view(-1,CLS_LEN),
                      cls_[idx_slice].view(-1).long())* self.lbd_cls
        
        loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_noobj + loss_cls
        
        return loss,loss_x,loss_y,loss_w,loss_h,loss_obj,loss_noobj,loss_cls

    
class yolo3_loss_on_b(yloss_basic):
    def __init__(self,lbd_coord=5,lbd_noobj=.1,lbd_cls=1,testing=False):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yolo3_loss_on_b,self).__init__(lbd_coord,lbd_noobj,lbd_cls,testing)
    
    def forward(self,y_pred,y_true,lbl_mask,vec_loc,t_xy,t_wh):
        bs = y_true.size()[0]
        y_pred = self.t2b(y_pred.float())
        y_true = y_true.float()
        lbl_mask = lbl_mask.float()
        
        mask,mask2,ioumap = self.loss_mask(y_true,y_pred,lbl_mask)
        
        y_true = y_true.float()
        y_pred = y_pred.float()
        
#         y_true = (y_true * mask).float()
        y_pred = (y_pred * mask).float()        
        y_true_noobj = (y_true * mask2).float()
        y_pred_noobj = (y_pred * mask2).float()
        
        y_pred_xy = y_pred[...,:2]
        y_true_xy = y_true[...,:2]
        
        y_pred_wh=torch.sqrt(F.relu(y_pred[...,2:4]))
        y_true_wh=torch.sqrt(y_true[...,2:4])
        # y_pred_wh=y_pred[...,2:4]
        # y_true_wh=y_true[...,2:4]
        
        y_pred_conf = y_pred[...,4]
#         y_true_conf = ioumap * y_true[...,4]
        y_true_conf = y_true[...,4]
        
        y_pred_cls = y_pred[...,5:]
        y_true_cls = y_true[...,5:]
        
        if self.testing:
            idxw,idxh,idxb = vec_loc[0,...,0].data[0],vec_loc[0,...,1].data[0],vec_loc[0,...,2].data[0]
            print("bb",y_pred[0,idxw,idxh,idxb,:4].view(-1).data.cpu().numpy(),
              "\t",y_true[0,idxw,idxh,idxb,:4].view(-1).data.cpu().numpy())
            print("conf",y_pred_conf[0,idxw,idxh,idxb].data[0],
                  "\t",y_true_conf[0,idxw,idxh,idxb].data[0])
            print("cls",torch.max(y_pred_cls[0,idxw,idxh,idxb])[0].data[0],
                  "\t",torch.max(y_true_cls[0,idxw,idxh,idxb])[0].data[0])
        
        loss_noobj = F.mse_loss(y_pred_noobj[...,4],y_true_noobj[...,4]) * self.lbd_noobj
        loss_obj = F.mse_loss(y_pred_conf,y_true_conf)
        
        loss_xy = F.mse_loss(y_pred_xy,y_true_xy) * self.lbd_coord
        loss_wh = F.mse_loss(y_pred_wh,y_true_wh) * self.lbd_coord
        
        loss_cls = F.binary_cross_entropy(y_pred_cls,y_true_cls) * self.lbd_cls
        loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls
        
        return loss,loss_xy,loss_wh,loss_obj,loss_noobj,loss_cls
