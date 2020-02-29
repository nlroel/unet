import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from torchvision import models

import matplotlib.pyplot as plt


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize((128, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (327680.0,)),
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((128, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((1540,), (1600-1540,)),
])

def train_model(model, criterion, optimizer, dataload, num_epochs=30):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            # plt.ion()
            # img_y = torch.squeeze(y).numpy()
            # plt.imshow(img_y, aspect='auto', interpolation='none', cmap=plt.get_cmap('gray'))
            # plt.show()
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
def train(args):
    model = Unet(1, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    liver_dataset = LiverDataset("/gs/home/majg/liupeng/code",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(1, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    liver_dataset = LiverDataset("data/val_2", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, yy in dataloaders:
            y=model(x)
            l1loss = nn.L1Loss()
            loss = l1loss(y, yy)
            print(loss.item())
            img_y=torch.squeeze(y).numpy()
            img_yy = torch.squeeze(yy).numpy()
            # img_y = (img_y + 1) * 127.5
            plt.figure()
            plt.subplot(121)
            plt.imshow(img_y.transpose(),aspect = 'auto', interpolation = 'none', cmap = plt.get_cmap('gray'))
            plt.subplot(122)
            plt.imshow(img_yy.transpose(),aspect = 'auto', interpolation = 'none', cmap = plt.get_cmap('gray'))
            plt.pause(0.01)
            # plt.waitforbuttonpress()
        plt.show()


if __name__ == '__main__':
    #参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)


