from torch.utils.data import dataset
from constant import *
import numpy as np

from PIL import Image

class Data_Multi(dataset.Dataset):
    def __init__(self, data_df, train_cls=False, testing=False, *args, **kwargs):
        """
        Object detection data generatorcd
        """
        super(Data_Multi, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data_df = data_df
        self.train_cls = train_cls # then the df will have a field called only_cls: 1 for training cls only
        self.img_ids = list(set(list(data_df["image_id"])))
        self.ids2fn = dict((k, v) for k, v in zip(self.data_df["image_id"], self.data_df["file_name"]))
        self.testing = testing
        self.anc = ANC_ARR

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_df = self.data_df[self.data_df.image_id == self.img_ids[idx]].head(50)

        if self.train_cls:
            only_cls = list(img_df["only_cls"])[0]

        if only_cls:
            img = Image.open(self.id2url(self.img_ids[idx])).convert("RGB")

        else:
            img = Image.open(self.id2url_cls(self.img_ids[idx])).convert("RGB")

        sample = self.transform(img)

        original = self.trans_origin(img)



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
        # cls_mask = np.zeros((FEAT_W, FEAT_H, BOX, 1))

        for i_lbl in range(N):
            t_box[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = t_xywh[i_lbl]
            b_box[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = b_xywh[i_lbl]
            conf_[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1
            cls_[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = cls_id[i_lbl]
            mask[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1
            # cls_mask[posi[i_lbl, 0], posi[i_lbl, 1], posi[i_lbl, 2]] = 1

        if self.testing:
            for i in sample, t_box, conf_, cls_, mask, b_box:
                print(i.shape)
        if self.train_cls:
            return sample,original, t_box, conf_, cls_, mask, b_box, only_cls
        else:
            return sample, original, t_box, conf_, cls_, mask, b_box

    def get_id(self, url):
        return int(url.split("/")[-1].split(".")[0])

    def id2url(self, image_id):
        return IMG + self.ids2fn[image_id]

    def id2url_cls(self, image_id):
        return IMG_CLS + self.ids2fn[image_id]

    def b2t_xy(self, x):
        x=x.copy()
        x[..., :2] = x[..., :2] - np.floor(x[..., :2])
        return x

    def b2t_wh(self, x, posi):
        x=x.copy()
        x[..., 2:4] = np.clip(x[..., 2:4], 1e-2, 12.999)
        lb_s = x.shape[0]
        anc_tile = np.tile(self.anc[np.newaxis, :, :], [lb_s, 1, 1])
        # print(anc_map.shape)
        x[..., 2:4] = x[..., 2:4] / anc_tile[np.eye(BOX)[posi[:, 2]] == 1]
        x[..., 2:4] = np.log(x[..., 2:4])
        return x

    def true_adj_expand(self, true_adj):
        return np.tile(true_adj[np.newaxis, np.newaxis, np.newaxis, :], [FEAT_W, FEAT_H, BOX, 1])




