from constant import *
from torch.utils.data import dataset
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from constant_char import *

fonts = glob("/data/fonts_cn/*")


def rd(scale, start=0):
    return int(np.random.rand() * scale) + start

def rd_font(size):
    return ImageFont.FreeTypeFont(fonts[rd(len(fonts))], size)

class Make_Char(dataset.Dataset):
    def __init__(self, **kwargs):
        """
        The data generator to generate iamge loaded with texts
        kwargs:
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
            {"bbox": bbox, "file_name": filename, "image_id": filename, "category_id": category_id})
        img.save(self.forge_dir + fn_id)
        return 0

    def id2url(self, filename):
        return self.img_dir + filename

    def write_char(self, draw, nb_c, rg):
        bbox = list()
        category_id = list()
        for i in range(nb_c):
            scale = rd(30,15)

            f_color = (rd(255), rd(255), rd(255), rd(200, 50))
            ft = rd_font(scale)  # font
            cate_id = np.random.choice(range(rg[0], rg[1]))
            txt = chars[cate_id]
            f_size = ft.getsize(text=txt)
            x, y = rd(WIDTH - f_size[0]), rd(HEIGHT - f_size[1])

            draw.text((x, y), txt, f_color, font=ft)

            bbox.append([x, y, f_size[0], f_size[1]])
            category_id.append(cate_id)

        return bbox, category_id

class Make_Char_cn(dataset.Dataset):
    def __init__(self, **kwargs):
        """
        The data generator to generate iamge loaded with texts
        kwargs:
            img_dir
            forge_dir
        """
        super(Make_Char_cn, self).__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.img_dir[-1] != "/":
            self.img_dir += "/"
        if self.forge_dir[-1] != "/":
            self.forge_dir += "/"
        self.grid_size = int(SIZE / SCALE)
        self.img_list = glob(self.img_dir + "*")
        self.pairs = np.concatenate([np.tile(range(self.grid_size), [self.grid_size, 1])[:, :, np.newaxis],
                                     np.tile(range(self.grid_size), [self.grid_size, 1]).T[:, :, np.newaxis]],
                                    axis=-1).reshape(-1, 2) * SCALE
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
        rg = rg_c
        bbox_, cateid_ = self.write_char_cls(draw, rg)
        bbox += bbox_
        category_id += cateid_
        filename = [fn_id] * len(category_id)
        self.df_dicts[idx] = pd.DataFrame(
            {"bbox": bbox, "file_name": filename, "image_id": filename, "category_id": category_id})
        img.save(self.forge_dir + fn_id)
        return 0

    def id2url(self, filename):
        return self.img_dir + filename

    def write_char(self, draw, nb_c, rg):
        bbox = list()
        category_id = list()
        for i in range(nb_c):
            scale = rd(30, 15)

            f_color = (rd(255), rd(255), rd(255), rd(200, 50))
            ft = rd_font(scale)  # font
            cate_id = np.random.choice(range(rg[0], rg[1]))
            txt = chars[cate_id]
            f_size = ft.getsize(text=txt)
            x, y = rd(WIDTH - f_size[0]), rd(HEIGHT - f_size[1])

            draw.text((x, y), txt, f_color, font=ft)

            bbox.append([x, y, f_size[0], f_size[1]])
            category_id.append(cate_id)

        return bbox, category_id

    def write_char_cls(self, draw, rg):
        bbox = list()
        category_id = list()
        for x, y in self.pairs[np.random.choice(range(100), 50, replace=False)]:
            scale = rd(SCALE-15, 15)
            f_color = (rd(255), rd(255), rd(255), rd(200, 50))

            ft = rd_font(scale)  # font

            cate_id = np.random.choice(range(rg[0], rg[1]))
            txt = chars[cate_id]
            f_size = ft.getsize(text=txt)

            draw.text((x, y), txt, f_color, font=ft)

            bbox.append([x, y, f_size[0], f_size[1]])
            category_id.append(cate_id)

        return bbox, category_id

