from constant import *
from torch import nn
from torch.nn import functional as F
import torch

AVG = False
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
        conf=F.sigmoid(x[...,4:5])
        cls_=F.softmax(x[...,5:],dim=-1)
        # cls_=x[...,5:]
        
        x = torch.cat([xy,wh,conf,cls_],dim=-1)
        return x
    
class yloss_basic(nn.Module):
    def __init__(self,lbd_coord,lbd_obj,lbd_noobj,lbd_cls,testing,train_all):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yloss_basic,self).__init__()
        self.t2b=t2b()
        self.resume_full()
        self.testing = testing
        self.train_all = train_all
        
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

    def only_cls(self):
        """train only classification"""
        self.lbd_coord = 0
        self.obj = 0
        self.noobj = 0
        self.lbd_cls = 1

    def resume_full(self):
        self.lbd_obj = LBD_OBJ
        self.lbd_coord = LBD_COORD
        self.lbd_noobj = LBD_NOOBJ
        self.lbd_cls = LBD_CLS
    
class yolo3_loss_on_t(yloss_basic):
    def __init__(self,lbd_coord=1,lbd_obj=5,lbd_noobj=1,lbd_cls=1,testing=False,train_all=True):
        """
        lbd_coord: lambda_coordinate
        lbd_noobj: lambda_no_object
        """
        super(yolo3_loss_on_t,self).__init__(lbd_coord,lbd_obj,lbd_noobj,lbd_cls,testing,train_all)
    
    def forward(self,y_pred,t_box, conf_, cls_, mask, cls_mask, b_box):
        """
        Forward Calculation of loss function
        :param y_pred: bs * grid_width * grid_height * boxes * (4+1+classes)
        :param t_box: bounding box $t$ t is easy for training
        :param conf_:  confidence, objectiveness score
        :param cls_:  class index
        :param mask: mask of the seen object
        :param cls_mask:
        :param b_box: bounding box $b$ b * downsampling scale(32) is the pixel size of resized image
        :return: loss,loss_x,loss_y,loss_w,loss_h,loss_obj,loss_noobj,loss_cls
        """
        t_box = t_box.float()
        b_box = b_box.float()
        conf_=conf_.float()

        y_pred_b = self.t2b(y_pred.float())
        ioumap = self.calc_iou(b_box,y_pred_b).detach()

        y_pred = y_pred.float()

        mask2 = (mask==0)*(ioumap<.5)
        
        y_pred_xy = F.sigmoid(y_pred[...,:2])
        y_pred_wh = y_pred[...,2:4]
        y_pred_conf = F.sigmoid(y_pred[...,4:5])
        
        mask_slice = mask.float()
        mask2_slice = mask2.float()

        loss_noobj = self.lbd_noobj * mse(y_pred_conf*mask2_slice,conf_*mask2_slice)/2.0
        loss_obj = mse(y_pred_conf*mask_slice, self.lbd_obj * ioumap * mask_slice)/2.0

        loss_x = self.lbd_coord * mse(y_pred_xy[...,0:1]*mask_slice,t_box[...,0:1]*mask_slice)/2.0
        loss_y = self.lbd_coord * mse(y_pred_xy[...,1:2]*mask_slice,t_box[...,1:2]*mask_slice)/2.0
        loss_w = self.lbd_coord * mse(y_pred_wh[...,0:1]*mask_slice,t_box[...,2:3]*mask_slice)/2.0
        loss_h = self.lbd_coord * mse(y_pred_wh[...,1:2]*mask_slice,t_box[...,3:4]*mask_slice)/2.0
        
        y_pred_cls = F.softmax((y_pred[...,5:][(mask==1).repeat(1,1,1,1,CLS_LEN)]).view(-1,CLS_LEN),dim=-1)

        if self.testing:
            print( y_pred_cls[0],cls_[mask==1].view(-1).long()[0])
        loss_cls = self.lbd_cls * ce(y_pred_cls,
                                     cls_[mask==1].view(-1).long())
        
        loss = loss_x*self.train_all + loss_y*self.train_all + loss_w*self.train_all + loss_h*self.train_all + loss_obj*self.train_all + loss_noobj*self.train_all + loss_cls
        
        return loss,loss_x,loss_y,loss_w,loss_h,loss_obj,loss_noobj,loss_cls