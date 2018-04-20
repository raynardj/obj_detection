import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from constant import *

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
        x[...,:2]=F.sigmoid(x[...,:2].clone())+self.grid.repeat(bs,1,1,1,1)
        x[...,2:4]=torch.exp(x[...,2:4].clone())*self.anc.repeat(bs,FEAT_W,FEAT_H,1,1)
        x[...,4:]=F.sigmoid(x[...,4:])
        return x
    
class yloss_basic(nn.Module):
    def __init__(self,lbd_coord=5,lbd_noobj=.1,lbd_cls=1,testing=False):
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
    
        return inter/union
    
    def loss_mask(self,y_true,y_pred,lbl_mask):
        ioumap = self.calc_iou(y_true,y_pred)
    
#         mask = ioumap.eq(torch.max(ioumap,dim=-1)[0].view(-1,FEAT_W,FEAT_H,1).repeat(1,1,1,BOX))
#         mask = mask.view(-1,FEAT_W,FEAT_H,BOX,1).repeat(1,1,1,1,VEC_LEN)
        mask = lbl_mask.unsqueeze(-1).repeat(1,1,1,1,VEC_LEN)
    
        mask2 = (ioumap<0.5).unsqueeze(-1).repeat(1,1,1,1,VEC_LEN).float()
        mask2[...,:4]=0
        mask2[...,5:]=0
        mask2-=mask
        mask2=torch.clamp(mask2,0,1)
    
        return mask.detach(),mask2.detach(),ioumap.detach()
    
class yolo3_loss_on_b(yloss_basic):
    def __init__(self,lbd_coord=5,lbd_noobj=.1,lbd_cls=1,testing=False):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yolo3_loss_on_b,self).__init__()
    
#     def forward(self,y_pred,y_true,lbl_mask):
#         y_pred = self.t2b(y_pred.float())
#         y_true = y_true.float()
#         lbl_mask = lbl_mask.float()
        
#         mask,mask2,ioumap = self.loss_mask(y_true,y_pred,lbl_mask)
        
#         y_true = y_true.float()
#         y_pred = y_pred.float()
        
#         y_true_noobj = (y_true * mask2).float()
#         y_pred_noobj = (y_pred * mask2).float()
        
#         y_pred_conf = y_pred[...,4]
#         y_true_conf = ioumap * y_true[...,4]
        
#         loss_noobj = F.binary_cross_entropy(y_pred_noobj[...,4],y_true_noobj[...,4]) * self.lbd_noobj
        
#         y_true = (y_true * mask).float()
#         y_pred = (y_pred * mask).float()

#         loss_bb = F.mse_loss(y_pred[...,:4],y_true[...,:4]) * self.lbd_coord
#         loss_obj = F.binary_cross_entropy(y_pred_conf,y_true_conf) 
#         loss_cls = F.binary_cross_entropy(y_pred[...,5:],y_true[...,5:]) * self.lbd_cls
#         loss = loss_bb + loss_obj + loss_noobj + loss_cls
        
#         return loss,loss_bb,loss_obj,loss_noobj,loss_cls
    
    def forward(self,y_pred,y_true,lbl_mask,vec_loc):
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
        
        y_pred_conf = y_pred[...,4]
#         y_true_conf = ioumap * y_true[...,4]
        y_true_conf = y_true[...,4]
        
        y_pred_cls = y_pred[...,5:]
        y_true_cls = y_true[...,5:]
        
        idxw,idxh,idxb = vec_loc[0,...,0].data[0],vec_loc[0,...,1].data[0],vec_loc[0,...,2].data[0]
        
        if self.testing:
            print("bb",y_pred_wh[0,idxw,idxh,idxb].view(-1).data.cpu().numpy(),
              "\t",y_true_wh[0,idxw,idxh,idxb].view(-1).data.cpu().numpy())
            print("conf",y_pred_conf[0,idxw,idxh,idxb].data[0],
                  "\t",y_true_conf[0,idxw,idxh,idxb].data[0])
            print("cls",torch.max(y_pred_cls[0,idxw,idxh,idxb])[0].data[0],
                  "\t",torch.max(y_true_cls[0,idxw,idxh,idxb])[0].data[0])
        
        loss_noobj = (torch.pow(y_pred_noobj[...,4]-y_true_noobj[...,4],2).sum() * self.lbd_noobj)/bs
        
        loss_xy = (torch.pow(y_pred_xy-y_true_xy,2).sum() * self.lbd_coord)/bs
        loss_wh = (torch.pow(y_pred_xy-y_true_wh,2).sum() * self.lbd_coord)/bs
        loss_obj = (torch.pow(y_pred_conf-y_true_conf,2).sum())/bs
        loss_cls = (torch.pow(y_pred_cls-y_true_cls,2).sum() * self.lbd_cls)/bs
        loss = loss_xy + loss_wh + loss_obj + loss_noobj + loss_cls
        
        return loss,loss_xy,loss_wh,loss_obj,loss_noobj,loss_cls
