"""
这个脚本对超大神经元图像进行划分、合并处理
处理的重点在于记录各个数据块的信息
"""

import os, cv2, sys
import numpy as np
from constant import ImageSize_NoMoreThan, ImageSize_PartitionTo, ImageSize_Overlap
from constant import Block_Size
from constant import ImageSuffixes
from constant import Image2DName_Length, Image3DName_Length
from datetime import datetime
from tools.printer import print_my

class NeuronImage_Huge():
    """
    这个类型定义一种超大的神经元图像，希望能在一般的配置条件下对其进行划分、分块分割、合并、组装等处理
    """
    def __init__(self, neuron_name, neuron_image_root):
        """
        这个超大的神经元图像由其根目录 root 唯一指代，
        root目录中直接保存文件夹 image (保存神经元二维图像序列)和信息文件 source.info (其中保存这个超大神经元图像的一些基本信息，如图像尺寸等)
        :param neuron_name: 这个超大尺寸神经元的名称
        :param root: 这个超大神经元图像的唯一标识
        """
        raise NotImplementedError
