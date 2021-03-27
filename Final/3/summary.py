from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    #input (N, in_dim)
    #output (N, 3, 64, 64)
    def __init__(self, in_dim, dim=32):
        super(Generator, self).__init__()
        self.fc = nn.Linear(in_dim, dim*8*5*5)
        self.fc_bn = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, 3, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 128, 3, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d(128,  64, 3, 2, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(64 ,   3, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = self.fc(input).reshape((-1, 256, 5, 5))
        x = self.relu(self.fc_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.relu(self.deconv5_bn(self.deconv5(x)))
        x = self.relu(self.deconv6_bn(self.deconv6(x)))
        x = self.tanh(self.deconv7(x))
        return x



class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.25))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8))
        self.fc = nn.Linear(128*dim,1)
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        y = y.view(-1)
        return y

G = Generator(in_dim=100)
D = Discriminator(3)
summary(G, (100,))
summary(D, (3,64,64))

