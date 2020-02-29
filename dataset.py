from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
from PIL import Image
import torch
import os
from torchvision.transforms import transforms

def Nomalize2(pil_img):
    np_img = np.array(pil_img)
    return Image.fromarray(np_img.transpose())

def list_clean(list, str):
    nlist = []
    for i in list:
        if str in i:
            nlist.append(i)
    return nlist

def make_dataset(root):
    imgs=[]
    scan_lines = os.listdir(os.path.join(root, 'scan_lines'))
    medium = os.listdir(os.path.join(root, 'medium'))
    n_scan_lines = list_clean(scan_lines, "scan_lines")
    n_scan_lines.sort()
    n_medium = list_clean(medium, "meidium")
    n_medium.sort()
    for i in range(len(n_scan_lines)):
        _in = os.path.join(root, 'scan_lines', n_scan_lines[i])
        _out = os.path.join(root, 'medium', n_medium[i])
        imgs.append((_in, _out))
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
        img_y = scio.loadmat(y_path)['medium_p']["sound_speed_map"][0,0][:,54:-54,53]
        tmp_x = img_x
        tmp_y = img_y
        tmp_x[:,0:100]=0
        # img_x = Image.fromarray(tmp_x.transpose())
        # img_y = Image.fromarray(tmp_y.transpose())
        img_x = Image.fromarray(tmp_x)
        img_y = Image.fromarray(tmp_y.transpose())

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

