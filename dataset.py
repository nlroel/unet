from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
from PIL import Image
import torch
import os

def Nomalize2(pil_img):
    np_img = np.array(pil_img)
    return Image.fromarray(np_img.transpose())


def make_dataset(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(n):
        img=os.path.join(root,"%03d.mat"%i)
        mask=os.path.join(root,"%03d_mask.png"%i)
        imgs.append((img,mask))
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = scio.loadmat(x_path)['scan_lines']
        img_y = Image.open(y_path).convert('L')
        tmp_y = np.array(img_y)
        tmp_x = img_x
        # tmp_x = tmp_x / 32768.0
        # tmp_y = tmp_y / 127.5 - 1

        img_x = Image.fromarray(tmp_x.transpose())
        img_y = Image.fromarray(tmp_y.transpose())
        # img_x = Nomalize2(img_x)
        # img_y = Nomalize2(img_y)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
