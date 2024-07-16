import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('-'*60)
        device = torch.device('cpu')
        return device
    
###########################################################################################################
class NeuralPix2Vid(nn.Module):
    def __init__(self, 
                 encoder_hidden_sizes:list=[16,64,128],
                 onet_num_layers=5, 
                 branch_size:int=512,
                 transformer_num_layers=6,
                 transformer_dim_feedforward:int=1024, 
                 transformer_nhead:int=4, transformer_activation=F.gelu):
        
        super(NeuralPix2Vid, self).__init__()
        self.c1, self.c2, self.c3 = encoder_hidden_sizes
        self.nc = branch_size // self.c3

        self.encoder = SpatialEncoder(in_channels=3, 
                                      hidden_channels=encoder_hidden_sizes, 
                                      return_hidden_states=True)
        
        self.lift_w = LiftingLayer(2*5, branch_size)
        self.lift_c = LiftingLayer(5*33, branch_size)
        self.lift_t = LiftingLayer(33, branch_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=branch_size, 
                                       nhead=transformer_nhead, 
                                       dim_feedforward=transformer_dim_feedforward,
                                       activation=transformer_activation,
                                       batch_first=True),
            num_layers=transformer_num_layers)
        
        self.dense_block = DenseBlock(branch_size, branch_size, num_layers=onet_num_layers)

        self.decoder1 = ConvLSTM3D(input_channels=self.c3,
                                   hidden_channels=[self.c3,self.c3,self.c3], 
                                   kernel_size=3)

        self.distributed_deconv1 = TimeDistributed(
            nn.ConvTranspose3d(in_channels=self.c3, out_channels=self.c3, kernel_size=(3,3,1), 
                               stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0)))
        
        self.conv1 = TimeDistributed(
            nn.Conv3d(in_channels=self.c3, out_channels=self.c2, kernel_size=3, padding=1))

        self.decoder2 = ConvLSTM3D(input_channels=self.c2,
                                   hidden_channels=[self.c2,self.c2,self.c2],
                                   kernel_size=3)
        
        self.distributed_deconv2 = TimeDistributed(
            nn.ConvTranspose3d(in_channels=self.c2, out_channels=self.c2, kernel_size=(3,3,1), 
                               stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0)))
        
        self.conv2 = TimeDistributed(
            nn.Conv3d(in_channels=self.c2, out_channels=self.c1, kernel_size=3, padding=1))
        
        self.decoder3 = ConvLSTM3D(input_channels=self.c1,
                                   hidden_channels=[self.c1,self.c1,self.c1],
                                   kernel_size=3)
        
        self.distributed_deconv3 = TimeDistributed(
            nn.ConvTranspose3d(in_channels=self.c1, out_channels=self.c1, kernel_size=(3,3,1), 
                               stride=(2,2,1), padding=(1,1,0), output_padding=(1,1,0)))
        
        self.conv3 = TimeDistributed(
            nn.Conv3d(in_channels=self.c1, out_channels=2, kernel_size=3, padding=1))
        
        self.fno = FNO(n_modes=(4,4,1), in_channels=128, hidden_channels=256, out_channels=128,
                       lifting_channels=256, projection_channels=256, n_layers=3)

    def forward(self, xm, xw, xc, xt):

        zt = self.dense_block(self.transformer(self.lift_t(xt))).reshape(-1,self.c3,self.nc)

        zm, hm = self.encoder(xm)
        zm1, zm2, zm3 = hm

        zw = self.dense_block(self.transformer(self.lift_w(xw)))
        zc = self.dense_block(self.transformer(self.lift_c(xc)))
        zb = torch.einsum('bi,bi->bi', zw, zc).reshape(-1,self.c3,self.nc)
        zb = torch.einsum('bcijk,bcp->bcijk', zm, zb)
        zx = torch.einsum('bcijk,bcp->bcijk', zb, zt)

        zx = self.fno(zx)

        zy1, _ = self.decoder1(zx.reshape(-1, 1, self.c3, 12, 12, 5))
        zy1 = torch.tile(zy1[0], (1, 33, 1, 1, 1, 1))
        zy1 = self.distributed_deconv1(zy1)
        zy1 = nn.ZeroPad3d((0,0,0,1,0,1))(zy1)
        zy1 = torch.einsum('btcijk,bcijk->btcijk', zy1, zm3)
        zy1 = self.conv1(zy1)

        zy2 = self.distributed_deconv2(zy1)
        zy2 = torch.einsum('btcijk,bcijk->btcijk', zy2, zm2)
        zy2 = self.conv2(zy2)

        zy3, _ = self.decoder3(zy2)
        zy3 = self.distributed_deconv3(zy3[0])
        zy3 = torch.einsum('btcijk,bcijk->btcijk', zy3, zm1)
        zy3 = self.conv3(zy3)

        return zy3

###########################################################################################################

class SeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv3d, self).__init__()

        self.depthwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class SqueezeExcite3d(nn.Module):
    def __init__(self, channels, ratio=4):
        super(SqueezeExcite3d, self).__init__()
        self.ratio = ratio
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite1 = nn.Linear(channels, channels//ratio)
        self.excite2 = nn.Linear(channels//ratio, channels)

    def forward(self, x):
        b, c, h, w, d = x.size()
        se_tensor = self.squeeze(x).view(b,c)
        se_tensor = F.relu(self.excite1(se_tensor))
        se_tensor = torch.sigmoid(self.excite2(se_tensor)).view(b,c,1,1,1)
        scaled_inputs = x * se_tensor.expand_as(x)
        return x + scaled_inputs
    
class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels:list, return_hidden_states:bool=False,
                 kernel_size=3, stride=1, padding=1, dilation=1, bias=True, 
                 pool_size=(2,2,1), dropout_rate=0.1):
        super(SpatialEncoder, self).__init__()
        assert len(hidden_channels) == 3, 'Hidden channels must be a list of 3 integers'
        c1, c2, c3 = hidden_channels
        self.return_hidden_states = return_hidden_states
        self.conv1 = SeparableConv3d(in_channels, c1, kernel_size, stride, padding, dilation, bias)
        self.sae1  = SqueezeExcite3d(c1)
        self.norm1 = nn.GroupNorm(c1, c1)
        self.conv2 = SeparableConv3d(c1, c2, kernel_size, stride, padding, dilation, bias)
        self.sae2  = SqueezeExcite3d(c2)
        self.norm2 = nn.GroupNorm(c2, c2)
        self.conv3 = SeparableConv3d(c2, c3, kernel_size, stride, padding, dilation, bias)
        self.sae3  = SqueezeExcite3d(c3)
        self.norm3 = nn.GroupNorm(c3, c3)
        self.pool = nn.MaxPool3d(pool_size)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout3d(dropout_rate)

    def forward(self, x):
        x = self.sae1(self.conv1(x))
        x1 = x
        x = self.drop(self.pool(self.gelu(self.norm1(x))))
        x = self.sae2(self.conv2(x))
        x2 = x
        x = self.drop(self.pool(self.gelu(self.norm2(x))))
        x = self.sae3(self.conv3(x))
        x3 = x
        x = self.drop(self.pool(self.gelu(self.norm3(x))))
        if self.return_hidden_states:
            return x, (x1,x2,x3)
        else:
            return x
        
class LiftingLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LiftingLayer, self).__init__()
        self.fc   = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(self.norm(x))
        return x
    
class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([LiftingLayer(in_features, out_features) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTM3DCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv3d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        depth, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, depth, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, depth, height, width, device=self.conv.weight.device))


class ConvLSTM3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM3D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(self.hidden_channels)
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]
            cell_list.append(ConvLSTM3DCell(input_channels=cur_input_channels,
                                            hidden_channels=self.hidden_channels[i],
                                            kernel_size=self.kernel_size,
                                            bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4, 5)
        b, _, _, d, h, w = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b,image_size=(d, h, w))
        layer_output_list = []
        last_state_list = []
        seq_len = x.size(1)
        cur_layer_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :, :], h=h, c=c)
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        if self.num_layers == 1:
            layer_output_list = layer_output_list[0]
            last_state_list = last_state_list[0]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps, C, D, H, W = x.size()
        x_reshaped = x.contiguous().view(batch_size * time_steps, C, D, H, W)
        y = self.module(x_reshaped)
        _, C_out, D_out, H_out, W_out = y.size()
        y = y.view(batch_size, time_steps, C_out, D_out, H_out, W_out)
        return y
    
