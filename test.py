import scipy.io as scio
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt

data = scio.loadmat("/Users/lroel/PycharmProjects/unet/data/val_2/medium/meidium_0217175617.mat")['medium_p']
img = data["sound_speed_map"][0, 0][:,:,53]
dx = 1.851851851851852e-04
center_x = int(data["x_pos"].item(0)[0,0] /dx) - 1
center_y = int(data["y_pos"].item(0)[0,0] /dx) - 1
radius = int(data["radius"].item(0)[0,0] /dx)
print(img.shape)
plt.figure()
plt.imshow(img[:,54:-54])
h = img.shape[0]
w = img.shape[1]
X, Y = np.ogrid[:h, :w]
dist_from_center = np.sqrt((X - center_x)**2 + (Y-center_y)**2)
mask = dist_from_center <= radius
maskimg = np.zeros(img.shape)
maskimg[~mask] = 1540
maskimg[mask] = 1600
plt.figure()
plt.imshow(maskimg[:,54:-54])
plt.show()
print(maskimg[11,11])
print(img[11,11])