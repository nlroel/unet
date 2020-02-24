import torch
from torch import nn

class conv_1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_1d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        return self.conv(input)


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        return self.conv(input)


class up_1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_1d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        return self.conv(input)


class up_2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
        )

    def forward(self, input):
        return self.conv(input)



class Unet(nn.Module):
    def __init__(self,in_ch,out_ch, nf=32):
        super(Unet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 1*nf, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
        )
        self.conv2 = conv_1d(1*nf, 2*nf)
        self.conv3 = conv_2d(2*nf, 4*nf)
        self.conv4 = conv_1d(4*nf, 8*nf)
        self.conv5 = conv_1d(8*nf, 8*nf)
        self.conv6 = conv_2d(8*nf, 8*nf)
        self.conv7 = conv_1d(8*nf, 8*nf)
        self.conv8 = conv_2d(8*nf, 8*nf)
        self.conv9 = conv_2d(8*nf, 8*nf)
        # self.conv10 = conv_1d(8*nf, 8*nf)
        # self.conv11 = conv_2d(8*nf, 8*nf)
        self.up1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*nf, out_ch, kernel_size=(1,4), stride=(1,2), padding=(0,1)),
            nn.Tanh(),
        )
        self.up2 = up_1d(4*nf, 1*nf)
        self.up3 = up_2d(8*nf, 2*nf)
        self.up4 = up_1d(16*nf, 4*nf)
        self.up5 = up_1d(16*nf, 8*nf)
        self.up6 = up_2d(16*nf, 8*nf)
        self.up7 = up_1d(16*nf, 8*nf)
        self.up8 = up_2d(16*nf, 8*nf)
        self.up9 = up_2d(8*nf, 8*nf)
        # self.up10 = up_1d(16*nf, 8*nf)
        # self.up11 = up_2d(8*nf, 8*nf)

    def forward(self,x):
        c1=self.conv1(x)
        c2=self.conv2(c1)
        c3=self.conv3(c2)
        c4=self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)
        c9 = self.conv9(c8)
        # c10 = self.conv10(c9)
        # c11 = self.conv11(c10)
        # u11 = self.up11(c11)
        # m11 = torch.cat([u11, c10], dim=1)
        # u10 = self.up10(m11)
        # m10 = torch.cat([u10, c9], dim=1)
        u9 = self.up9(c9)
        m9 = torch.cat([u9, c8], dim=1)
        u8 = self.up8(m9)
        m8 = torch.cat([u8, c7], dim=1)
        u7 = self.up7(m8)
        m7 = torch.cat([u7, c6], dim=1)
        u6 = self.up6(m7)
        m6 = torch.cat([u6, c5], dim=1)
        u5 = self.up5(m6)
        m5 = torch.cat([u5, c4], dim=1)
        u4 = self.up4(m5)
        m4 = torch.cat([u4, c3], dim=1)
        u3 = self.up3(m4)
        m3 = torch.cat([u3, c2], dim=1)
        u2 = self.up2(m3)
        m2 = torch.cat([u2, c1], dim=1)
        u1 = self.up1(m2)

        return u1







