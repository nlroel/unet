import scipy.io as scio
import numpy as np
from PIL import Image
import torch
import os


imgs=[]
scan_lines = os.listdir('data/train_2/scan_lines')
medium = os.listdir('data/train_2/medium')
for i in range(len(scan_lines)):
    _in = os.path.join('data/train_2','scan_lines',scan_lines[i])
    _out = os.path.join('data/train_2','medium',medium[i])
    imgs.append((_in, _out))

print(imgs)