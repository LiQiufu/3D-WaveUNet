import torch
import os
from torch.utils.data import Dataset
from postprocess.divide_neuron_image import Image3D_PATH
from constant import Mean_TrainData, Std_TrainData
import numpy as np

class SingleNeuron_Dataset(Dataset):
    """
    将单个神经元切割得到的图像块包装成 torch 数据集
    """
    def __init__(self, neuron_image_path):
        self.neuron_image_path = neuron_image_path
        self.neuron_image = Image3D_PATH(image_path = self.neuron_image_path)
        self.neuron_image.divide_regular()
        self.neuron_blocks = self.neuron_image.block_list
        self.neuron_block_locations = self.neuron_image.location_list
        self.neuron_block_sizes = self.neuron_image.size_list
        self.shape = self.neuron_image.shape()
        super(SingleNeuron_Dataset, self).__init__()

    def __getitem__(self, item):
        block = self.neuron_blocks[item]
        block = np.array(block).astype(np.float32)
        block /= 255.0
        block -= Mean_TrainData
        block /= Std_TrainData
        return torch.tensor(block).float()

    def __len__(self):
        assert len(self.neuron_blocks) == len(self.neuron_block_locations)
        return len(self.neuron_blocks)