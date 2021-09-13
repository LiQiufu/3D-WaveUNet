"""
这个脚本描述用于训练神经元图像的网络结构
"""

import torch
from torch import nn
from torch.nn import Module
from torch.nn.parallel import DataParallel
from datetime import datetime
import torch.nn.functional as F
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D, DWT_3D_tiny


class Cov3x3_BN(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1,
                 padding = 1, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', with_BN = True):
        super(Cov3x3_BN, self).__init__()
        self.with_BN = with_BN
        self.cov = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
        self.bn = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.with_BN:
            return self.relu(self.bn(self.cov(input)))
        else:
            return self.relu(self.cov(input))


class Threshold_HFC(Module):
    def __init__(self, la = 0.25):
        super(Threshold_HFC, self).__init__()
        self.threshold = nn.Hardshrink(lambd = la)
    def forward(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        return self.threshold(LLH), self.threshold(LHL), self.threshold(LHH), \
               self.threshold(HLL), self.threshold(HLH), self.threshold(HHL), self.threshold(HHH)


class Neuron_SegNet(Module):
    """
    构建用于神经元图像分割的 3D SegNet
    它的编解码器两端进行下采样和上采样分别使用 max-pooling 和 max-unpooling
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4):
        super(Neuron_SegNet, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = nn.MaxPool3d(kernel_size = 2, return_indices = True)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = nn.MaxPool3d(kernel_size = 2, return_indices = True)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = nn.MaxPool3d(kernel_size = 2, return_indices = True)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = nn.MaxPool3d(kernel_size = 2, return_indices = True)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = nn.MaxUnpool3d(kernel_size = 2, stride = 2)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = nn.MaxUnpool3d(kernel_size = 2, stride = 2)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = nn.MaxUnpool3d(kernel_size = 2, stride = 2)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = nn.MaxUnpool3d(kernel_size = 2, stride = 2)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output = self.cov3d_12_en(self.cov3d_11_en(input))
        output, indices_1 = self.downsampling_1(output)

        output = self.cov3d_22_en(self.cov3d_21_en(output))
        output, indices_2 = self.downsampling_2(output)

        output = self.cov3d_32_en(self.cov3d_31_en(output))
        output, indices_3 = self.downsampling_3(output)

        output = self.cov3d_42_en(self.cov3d_41_en(output))
        output, indices_4 = self.downsampling_4(output)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(input = output, indices = indices_4)
        output = self.cov3d_42_de(self.cov3d_41_de(output))

        output = self.upsampling_3(input = output, indices = indices_3)
        output = self.cov3d_32_de(self.cov3d_31_de(output))

        output = self.upsampling_2(input = output, indices = indices_2)
        output = self.cov3d_22_de(self.cov3d_21_de(output))

        output = self.upsampling_1(input = output, indices = indices_1)
        output = self.cov3d_12_de(self.cov3d_11_de(output))

        output = self.cov_final(output)

        return output


class Neuron_UNet_V1(Module):
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 max-pooling、解码器上采样使用 反卷积
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4):
        super(Neuron_UNet_V1, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = nn.MaxPool3d(kernel_size = 2)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = nn.MaxPool3d(kernel_size = 2)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = nn.MaxPool3d(kernel_size = 2)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = nn.MaxPool3d(kernel_size = 2)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = nn.ConvTranspose3d(in_channels = 8 * channel_width, out_channels = 8 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = nn.ConvTranspose3d(in_channels = 4 * channel_width, out_channels = 4 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = nn.ConvTranspose3d(in_channels = 2 * channel_width, out_channels = 2 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = nn.ConvTranspose3d(in_channels = 1 * channel_width, out_channels = 1 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(input = output)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        output = self.upsampling_3(input = output)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        output = self.upsampling_2(input = output)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        output = self.upsampling_1(input = output)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


class Neuron_UNet_V2(Module):
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 步长卷积，解码器上采样使用 线性插值
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4):
        super(Neuron_UNet_V2, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = nn.Conv3d(in_channels = 1 * channel_width, out_channels = 1 * channel_width,
                                        kernel_size = 1, stride = 2)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = nn.Conv3d(in_channels = 2 * channel_width, out_channels = 2 * channel_width,
                                        kernel_size = 1, stride = 2)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = nn.Conv3d(in_channels = 4 * channel_width, out_channels = 4 * channel_width,
                                        kernel_size = 1, stride = 2)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = nn.Conv3d(in_channels = 8 * channel_width, out_channels = 8 * channel_width,
                                        kernel_size = 1, stride = 2)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


class Neuron_WaveSNet_V1(Module):
    """
    构建集成小波变换用于神经元分割的 3D UNet
    它的编码器下采样使用 3D-DWT，解码器上采样使用 线性插值
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4, wavename = 'haar'):
        super(Neuron_WaveSNet_V1, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D_tiny(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D_tiny(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D_tiny(wavename = wavename)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D_tiny(wavename = wavename)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        _, _, D, H, W = output.size()
        output = F.interpolate(output, size = (2 * D, 2 * H, 2 * W), mode = 'trilinear', align_corners = True)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


class Neuron_WaveSNet_V2(Module):
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 3D DWT、解码器上采样使用 反卷积
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4, wavename = 'haar'):
        super(Neuron_WaveSNet_V2, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D_tiny(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D_tiny(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D_tiny(wavename = wavename)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D_tiny(wavename = wavename)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = nn.ConvTranspose3d(in_channels = 8 * channel_width, out_channels = 8 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = nn.ConvTranspose3d(in_channels = 4 * channel_width, out_channels = 4 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = nn.ConvTranspose3d(in_channels = 2 * channel_width, out_channels = 2 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = nn.ConvTranspose3d(in_channels = 1 * channel_width, out_channels = 1 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(input = output)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        output = self.upsampling_3(input = output)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        output = self.upsampling_2(input = output)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        output = self.upsampling_1(input = output)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


class Neuron_WaveSNet_V3(Module):
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 3D DWT、解码器上采样使用 3D IDWT
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4, wavename = 'haar'):
        super(Neuron_WaveSNet_V3, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D(wavename = wavename)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D(wavename = wavename)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = IDWT_3D(wavename = wavename)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = IDWT_3D(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        output = self.upsampling_3(output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        output = self.upsampling_2(output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        output = self.upsampling_1(output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


class Neuron_WaveSNet_V4(Module):
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 3D DWT、解码器上采样使用 3D IDWT, 高频分量经过滤波处理
    """
    def __init__(self, num_class = 2, with_BN = True, channel_width = 4, wavename = 'haar', la = 0.25):
        super(Neuron_WaveSNet_V4, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = 1, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D(wavename = wavename)
        self.threshold_1 = Threshold_HFC(la = la)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D(wavename = wavename)
        self.threshold_2 = Threshold_HFC(la = la)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D(wavename = wavename)
        self.threshold_3 = Threshold_HFC(la = la)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D(wavename = wavename)
        self.threshold_4 = Threshold_HFC(la = la)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = IDWT_3D(wavename = wavename)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = IDWT_3D(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.downsampling_1(output_1)
        LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.threshold_1(LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.downsampling_2(output_2)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.threshold_2(LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.downsampling_3(output_3)
        LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.threshold_3(LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.downsampling_4(output_4)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.threshold_4(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        output = self.upsampling_3(output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        output = self.upsampling_2(output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        output = self.upsampling_1(output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)

        return output


if __name__ == '__main__':
    import numpy as np
    from pprint import pprint
    input = torch.rand(size = (1, 1, 32, 128, 128)).float()
    model = Neuron_WaveSNet_V4(channel_width = 8)
    start = datetime.now()
    output = model(input)
    stop = datetime.now()
    print('tooking {} secs'.format(stop - start))