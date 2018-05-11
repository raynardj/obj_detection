from torch.utils.data import dataset
from constant import *
import numpy as np

from PIL import Image

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
        x=x.copy()
        x[..., :2] = x[..., :2] - np.floor(x[..., :2])
        return x

    def b2t_wh(self, x, posi):
        x=x.copy()
        x[..., 2:4] = np.clip(x[..., 2:4], 1e-2, 12.999)
        lb_s = x.shape[0]
        anc_tile = np.tile(self.anc[np.newaxis, :, :], [lb_s, 1, 1])
        # print(anc_map.shape)
        x[..., 2:4] = x[..., 2:4] / anc_tile[np.eye(5)[posi[:, 2]] == 1]
        x[..., 2:4] = np.log(x[..., 2:4])
        return x

    def true_adj_expand(self, true_adj):
        return np.tile(true_adj[np.newaxis, np.newaxis, np.newaxis, :], [FEAT_W, FEAT_H, BOX, 1])


from glob import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

fonts = glob("/data/fonts_cn/*")


def rd(scale, start=0):
    return int(np.random.rand() * scale) + start

def rd_font(size):
    return ImageFont.FreeTypeFont(fonts[rd(len(fonts))], size)

class Make_Char(dataset.Dataset):
    def __init__(self, **kwargs):
        """
        kwargs
        img_dir
        forge_dir
        """
        super(Make_Char, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.img_dir[-1] != "/":
            self.img_dir += "/"
        if self.forge_dir[-1] != "/":
            self.forge_dir += "/"
        self.img_list = glob(self.img_dir + "*")
        self.df_dicts = [dict()] * len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fn_id = self.img_list[idx].split("/")[-1]
        fn = self.id2url(fn_id)
        img = Image.open(fn).convert("RGB").resize((WIDTH, HEIGHT))
        draw = ImageDraw.Draw(img)
        nb_ttl = rd(10, 15)
        bbox = list()
        category_id = list()
        for nb, rg in zip([.4, .2, .2, .2], [rg_n, rg_l, rg_u, rg_c]):
            nb_ = int(nb * nb_ttl)
            bbox_, cateid_ = self.write_char(draw, nb_, rg)
            bbox += bbox_
            category_id += cateid_
        filename = [fn_id] * len(category_id)
        self.df_dicts[idx] = pd.DataFrame(
            {"bbox": bbox, "filename": filename, "image_id": filename, "category_id": category_id})
        img.save(self.forge_dir + fn_id)
        return 0

    def id2url(self, filename):
        return self.img_dir + filename

    def write_char(self, draw, nb_c, rg):
        bbox = list()
        category_id = list()
        for i in range(nb_c):
            scale = rd(30, 20)
            x, y = rd(WIDTH - scale), rd(HEIGHT - scale)
            f_color = (rd(255), rd(255), rd(255), rd(200, 50))
            ft = rd_font(rd(30))  # font
            cate_id = np.random.choice(range(rg[0], rg[1]))
            txt = chars[cate_id]
            f_size = ft.getsize(text=txt)

            draw.text((x, y), txt, f_color, font=ft)

            bbox.append([x, y, f_size[0], f_size[1]])
            category_id.append(cate_id)

        return bbox, category_id
