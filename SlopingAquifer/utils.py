import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, nt0=30):
        self.nfeatures = os.listdir('data/features')
        self.nt0 = nt0

    def __len__(self):
        return len(self.nfeatures)

    def __getitem__(self, idx):
        self.features = np.load('data/features/features_{}.npy'.format(idx))
        self.targets  = np.load('data/targets/targets_{}.npy'.format(idx))
        self.targets1 = self.targets[..., :self.nt0]
        self.targets2 = self.targets[-1, ..., self.nt0:]
        return self.features, self.targets1, self.targets2

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, drop=0):
        super(Unet, self).__init__()
        self.conv1   = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=drop)
        self.conv2   = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=drop)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=drop)
        self.conv3   = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate=drop)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate=drop)
        
        self.deconv2 = self.deconv(input_channels,   output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv3 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv3), 1)

        out_deconv2 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv2), 1)

        out_deconv1 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv1), 1)

        return self.output_layer(concat0)

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(in_planes, output_channels, 
                      kernel_size = kernel_size,
                      stride      = stride,
                      padding     = 1, 
                      bias        = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, 
                               kernel_size = 4,
                               stride      = 2, 
                               padding     = 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride):
        return nn.Conv3d(input_channels, output_channels, 
                         kernel_size = kernel_size,
                         stride      = stride, 
                         padding     = (kernel_size - 1) // 2)
    
class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels:int=5, out_channels:int=2, mlp_size:int=128):
        super(SimpleBlock3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width  = width

        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv4 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv5 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.unet3 = Unet(self.width, self.width, kernel_size=3)
        self.unet4 = Unet(self.width, self.width, kernel_size=3)
        self.unet5 = Unet(self.width, self.width, kernel_size=3)
        self.fc1 = nn.Linear(self.width, mlp_size)
        self.fc2 = nn.Linear(mlp_size, out_channels)

    def forward(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet3(x) 
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

class NeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, width, mlp_size:int=128):
        super(NeuralOperator, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.model = SimpleBlock3d(modes1, modes2, modes3, width, 
                                   in_channels=in_channels, out_channels=out_channels, mlp_size=mlp_size)

    def forward(self, x):
        batchsize, size_x, size_y, size_t = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = self.model(x)
        x = x.view(batchsize, size_x, size_y, size_t, self.out_channels)
        return x #.squeeze() #.permute(0, 4, 1, 2, 3)

class DualFNO(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, 
                 modes1:int=10, modes2:int=10, modes3:int=10, width:int=16, mlp_size:int=128):
        super(DualFNO, self).__init__()
        self.op1 = NeuralOperator(in_channels,                out_channels_1, modes1, modes2, modes3, width, mlp_size=mlp_size)
        self.op2 = NeuralOperator(in_channels+out_channels_1, out_channels_2, modes1, modes2, modes3, width, mlp_size=mlp_size)
    
    def forward(self, x):
        y1 = self.op1(x)
        y2 = self.op2(torch.cat([x, y1], dim=-1))
        return y1, y2