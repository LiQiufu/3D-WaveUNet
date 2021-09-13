"""
这个脚本对单个神经元进行分割
"""

import os, sys
import torch
import copy
import numpy as np
from constant import Block_Size, RESULT_SAVE_PATH, NEURON_NAME_TEST, DataNeuron_Root, MODEL_ROOT
from constant import RESULT_SAVE_PATH_HUST, NEURON_NAME_TEST_HUST, DataNeuron_Root_HUST
from tools.image_3D_io import save_image_3d
from skimage.measure import label as Label
from postprocess_huge_image.huge_image_partition_assemble import NeuronImage_Huge

from torch.utils.data.dataloader import DataLoader
from postprocess.dataloader_neuron_image import SingleNeuron_Dataset
from postprocess.network_pretrained import Network_Pretrained
from constant import Image2DName_Length, Image3DName_Length
from datetime import datetime
from tools.printer import print_my

class Segmentation_SingleNeuron():
    """
    对单个神经元图像进行 切分、逐块分割、拼接、保存 等操作
    """
    def __init__(self, neuron_name, neuron_image_path, net_pretrained, batchsize = 8):
        """
        :param neuron_image_path: 神经元图像保存路径，它的终端子目录名称通常是其指代的神经元名称；其中包含一个子目录，如 'image'
        :param net_pretrained: 已经加载了预训练模型的 pytorch 网络
        :param batchsize: 批次大小
        """
        self.neuron_name = neuron_name
        self.num_class = 2
        self._data(neuron_image_path = neuron_image_path, batchsize = batchsize)
        self.net_pretrained = net_pretrained

    def _data(self, neuron_image_path, batchsize):
        # 将神经元图像切分并生成 pytorch 能处理的数据
        self.singleneuron_dataset = SingleNeuron_Dataset(neuron_image_path = neuron_image_path)
        self.image_3d = self.singleneuron_dataset.neuron_image.image_3d
        self.block_locations = self.singleneuron_dataset.neuron_block_locations     #获取规则切割神经元图像后的每个图像块起始坐标列表
        self.block_sizes = self.singleneuron_dataset.neuron_block_sizes             #获取规则切割时候每个图像块的实际尺寸
        self.image_shape = self.singleneuron_dataset.shape                          #获取神经元尺寸
        self.singleneuron_loader = DataLoader(dataset = self.singleneuron_dataset, batch_size = batchsize, shuffle = False) #不能修改shuffle值

    def segment_block(self):
        """
        逐块分割
        :return:
        """
        self.block_label_pre = None   # 保存每个图像块的分割结果
        for index, data in enumerate(self.singleneuron_loader):
            #if index % 10 == 0:
            time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
            sys.stdout.write('\r{}-------- processing the {}th ({}) batch for neuron image "{}"'.
                  format(time, str(index).rjust(4), len(self.singleneuron_loader), self.neuron_name))
            sys.stdout.flush()
            block_data = data
            block_data = block_data.unsqueeze(dim = 1)
            block_data = block_data.cuda()
            output = self.net_pretrained(block_data)
            _, pre_segmentation = output.topk(1, dim = 1)
            pre_segmentation = pre_segmentation.squeeze(dim = 1)
            self.block_label_pre = pre_segmentation if self.block_label_pre is None \
                else torch.cat((self.block_label_pre, pre_segmentation), dim = 0)
            #pre_segmentation = pre_segmentation.cpu().numpy()
            #for ii in range(pre_segmentation.shape[0]):
            #    block_pre = pre_segmentation[ii, :, :, :]
            #    self.block_label_pre.append(block_pre)
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish segmenting for the neuron image "{}"\n'.format(time, self.neuron_name))

    def merge(self):
        """
        将图像块的分割结果按照位置坐标进行合并
        :return:
        """
        print_my('-------- merging data ....')
        self.label_pre = torch.zeros(size = self.image_shape)
        self.label_pre = self.label_pre.cuda()
        assert len(self.block_label_pre) == len(self.block_locations) == len(self.block_sizes)
        for index in range(self.block_label_pre.shape[0]):
            label_pre = self.block_label_pre[index, :, :, :]
            time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
            sys.stdout.write('\r{}-------- merging {} - {} block '.format(time, index, len(self.block_label_pre)))
            sys.stdout.flush()
            z_start, y_start, x_start = self.block_locations[index]
            depth, height, width = self.block_sizes[index]
            z_stop = z_start + depth
            y_stop = y_start + height
            x_stop = x_start + width
            if (depth, height, width) != Block_Size:
                self.label_pre[z_start:z_stop, y_start:y_stop, x_start:x_stop] += label_pre[:depth, :height, :width]
            else:
                self.label_pre[z_start:z_stop, y_start:y_stop, x_start:x_stop] += label_pre
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish merging - 0...'.format(time))
        self.label_pre[self.label_pre != 0] = 255
        self.label_pre = torch.tensor(self.label_pre, dtype = torch.uint8)
        self.label_pre = self.label_pre.cpu().numpy()
        time = '[{}]'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ' ' * 4
        sys.stdout.write('\n{}-------- finish merging - 1...\n'.format(time))

    def segment(self, final_process = False):
        """
        按照神经元图像的分割结果提取原图像中的神经元部分
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除；对于尺寸比较大的图像，这个操作速度很慢，不建议使用
        :return:
        """
        print_my('-------- in segmenting - 0')
        self.image_pre = torch.tensor(self.image_3d).cuda()
        print_my('-------- in segmenting - 1')
        for i in range(self.image_pre.shape[0]):
            self.image_pre[i][self.label_pre[i] != 255] = 0
        self.image_pre = self.image_pre.cpu().numpy()
        print_my('-------- in segmenting - 2')
        if final_process:
            self._get_theMax_connectedRegion()
            self.image_final = copy.copy(self.image_3d)
            self.image_final[self.label_pre_final != 1] = 0

    def run(self, final_process = False):
        """
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除
        :return:
        """
        self.segment_block()      #对每个图像块进行图像分割，预测其中属于神经元的像素点
        self.merge()        #将预测结果进行合并处理，获得整个神经元图像的预测结果
        self.segment(final_process = final_process)

    def _get_theMax_connectedRegion(self, connectivity = 3):
        """
        返回预测标签中的最大连通域
        :return:
        """
        raise RuntimeError('这个子函数还是不要用了！')
        print('---------- geting the max connected region ...')
        connected_label, num = Label(self.label_pre, connectivity = connectivity, background = 0, return_num = True)
        max_label = 0
        max_num = 0
        for index in range(1, num+1):
            if np.sum(connected_label == index) > max_num:
                max_num = np.sum(connected_label == index)
                max_label = index
        self.label_pre_final = (connected_label == max_label)

    def save(self, save_path, label = False, final_process = False):
        """
        :param final_process: True: 进行求最大连通域操作，能够将某些孤立点去除，但也会将一些孤立纤维片段去除
        :return:
        """
        print_my('-------- saving image data ... ')
        save_image_3d(self.image_pre, image_save_root = os.path.join(save_path, 'image_pre_init'))
        if final_process:
            save_image_3d(self.image_final, image_save_root = os.path.join(save_path, 'image_pre_final'))
        if label:
            print_my('\n-------- saving label data ... ')
            save_image_3d(self.label_pre, image_save_root = os.path.join(save_path, 'label_pre_init'))
            if final_process:
                self.label_pre_final = np.array(self.label_pre_final * 255, dtype = np.uint8)
                save_image_3d(self.label_pre_final, image_save_root = os.path.join(save_path, 'label_pre_final'))


class Segmentation_HugeNeuron():
    """
    对单个超大尺寸神经元图像进行 划分、逐子图像分割、整合、保存 等操作
    混合、融合、配合、结合、组合、整合
    """
    def __init__(self, neuron_name, neuron_image_root, net_pretrained, batchsize=8):
        """
        :param neuron_name: 超大神经元名称
        :param neuron_image_root: 神经元图像保存路径，这个路径的终端目录通常是当前神经元名称 neuron_name
        :param net_pretrained: 已经加载了预训练模型的 pytorch 网络
        :param batchsize: 批次大小
        """
        self.num_class = 2
        self.neuron_name = neuron_name
        self.neuron_image_root = neuron_image_root
        self.huge_neuron = NeuronImage_Huge(neuron_name = self.neuron_name,
                                            neuron_image_root = self.neuron_image_root)
        self.net_pretrained = net_pretrained
        self.batchsize = batchsize

    def run(self, save_path, label = False):
        """
        对超大尺寸的神经元图像进行：
        1.划分
        2.逐个子图分割
        3.合并分割后的子图
        :return:
        """
        self.partition()
        self.segment()
        self.assemble(save_root = save_path, label = label)

    def partition(self):
        self.huge_neuron.partition()
        self.partitioned_image_root = self.huge_neuron.partition_save_root

    def segment(self):
        save_path = self.partitioned_image_root
        for index, neuron_subimage_name in enumerate(self.huge_neuron.subimage_name_list):
            print_my('')
            subimage_path = os.path.join(self.partitioned_image_root, neuron_subimage_name)
            seg_neuron = Segmentation_SingleNeuron(neuron_name = neuron_subimage_name + ' / '
                                                                 + str(self.huge_neuron.image3d_number_partitioned).zfill(Image3DName_Length),
                                                   neuron_image_path = subimage_path,
                                                   net_pretrained = self.net_pretrained,
                                                   batchsize = self.batchsize)
            seg_neuron.run()
            seg_neuron.save(save_path = os.path.join(save_path, neuron_subimage_name), label = True)
            print_my('')

    def assemble(self, save_root, label = True):
        self.huge_neuron.assemble(source_root = self.partitioned_image_root,
                                  save_root = save_root, leaf_path = 'image_pre_init')
        if label:
            self.huge_neuron.assemble(source_root = self.partitioned_image_root,
                                      save_root = save_root, leaf_path = 'label_pre_init')


def denoise_neurons_for_test(net_name, wavename = 'none', model_epoch = 29, batchsize = 8, gpus = (0,)):
    """
    :param net_name:
    :param wavename:
    :param model_epoch:
    :param batchsize:
    :param gpus:
    :return:
    """
    # 预训练模型
    model_subpath = os.path.join('weight_models', 'epoch_{}.pth.tar'.format(model_epoch)) if wavename == 'none' \
        else os.path.join('weight_models_' + wavename, 'epoch_{}.pth.tar'.format(model_epoch))
    model_path = os.path.join(MODEL_ROOT, net_name, model_subpath)
    net_pretrained = Network_Pretrained(net_name = net_name, model_path = model_path, gpus = gpus, wavename = wavename)
    # 保存路径
    subpath = net_name if wavename == 'none' else net_name + '_' + wavename
    save_root = os.path.join(RESULT_SAVE_PATH, subpath, 'epoch_{}'.format(model_epoch))
    run_times = dict()
    for neuron_name in NEURON_NAME_TEST:
        t0 = datetime.now()
        neuron_image_path = os.path.join(DataNeuron_Root, neuron_name)
        print_my('processing {}'.format(neuron_image_path))
        seg_neuron = Segmentation_SingleNeuron(neuron_name = neuron_name, neuron_image_path = neuron_image_path,
                                               net_pretrained = net_pretrained, batchsize = batchsize)
        seg_neuron.run()
        save_path = os.path.join(save_root, neuron_name)
        t1 = datetime.now()
        print('segmentating {} took {} secs'.format(neuron_name, t1 - t0))
        run_times[neuron_name] = t1 - t0
        seg_neuron.save(save_path = save_path, label = True)
        del seg_neuron
    for neuron_name in run_times:
        print('{} ==> {}'.format(neuron_name, run_times[neuron_name]))


def denoise_neurons_for_HUST(net_name, wavename = 'none', model_epoch = 29, batchsize = 8, gpus = (0,)):
    """
    :param net_name:
    :param wavename:
    :param model_epoch:
    :param batchsize:
    :param gpus:
    :return:
    """
    # 预训练模型
    model_subpath = os.path.join('weight_models', 'epoch_{}.pth.tar'.format(model_epoch)) if wavename == 'none' \
        else os.path.join('weight_models_' + wavename, 'epoch_{}.pth.tar'.format(model_epoch))
    model_path = os.path.join(MODEL_ROOT, net_name, model_subpath)
    net_pretrained = Network_Pretrained(net_name = net_name, model_path = model_path, gpus = gpus, wavename = wavename)
    # 保存路径
    subpath = net_name if wavename == 'none' else net_name + '_' + wavename
    save_root = os.path.join(RESULT_SAVE_PATH_HUST, subpath, 'epoch_{}'.format(model_epoch))
    for neuron_name in NEURON_NAME_TEST_HUST:
        neuron_image_path = os.path.join(DataNeuron_Root_HUST, neuron_name)
        print_my('processing {}, using {}_ep{}_{}'.format(neuron_image_path, net_pretrained.net_name, model_epoch, wavename))
        seg_neuron = Segmentation_HugeNeuron(neuron_name = neuron_name, neuron_image_root = neuron_image_path,
                                             net_pretrained = net_pretrained, batchsize = batchsize)
        save_path = os.path.join(save_root, neuron_name)
        seg_neuron.run(save_path = save_path, label = True)
        del seg_neuron
