"""
这个脚本负责对三维神经元图像进行切割
"""

import numpy as np
from tools.image_3D_io import save_image_3d, load_image_3d
from constant import Block_Size
import os
from random import randint

class Image3D():
    """
    这个类型描述一个三维图像，这个三维图像数据的第一个维度描述图像的层数（Z轴，通道数），第二个描述高度（Y轴，行数），第三个描述宽度（X轴，列数）
    """
    def __init__(self, image):
        """
        :param image: 三维数据
        """
        self.image_3d = image            # 三维 np.narray 类型
        self.depth, self.height, self.width = image.shape
        self.divide_flag = True         # 切割图像时候的一个标识符
        self.block_list = []            # 保存图像块的列表
        self.location_list = []         # 图像块的其实点坐标，拼接时候需要
        self.size_list = []             # 图像块实际尺寸，拼接时候需要

    def shape(self):
        return self.depth, self.height, self.width

    def save(self, image_save_root = None, dim = 0):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int, 切片维度
        :return:
        """
        save_image_3d(self.image_3d, image_save_root = os.path.join(image_save_root, 'image'), dim = dim)

    def save_block(self, root, block_size = Block_Size, mode = 'regular', info_file = 'info.txt'):
        """
        保存切分得到的图像块
        :param root:
        :param mode: in ['regular', 'irregular', 'both']
                'regular' 表示将图像拆分为若干不想重叠的固定大小图像块
                'irregular' 表示从图像中随机提取固定大小的图像块
                'both' 表示两种方式都进行
        :return:
        """
        if not self.block_list:
            if mode == 'regular':
                self.divide_regular(block_size = block_size)
            elif mode == 'irregular':
                self.divide_irregular(block_size = block_size)
            elif mode == 'both':
                self.divide_regular(block_size = block_size)
                self.divide_irregular(block_size = block_size)
            else:
                raise ValueError("mode is in ['regular', 'irregular', 'both']")
        info_file_name = os.path.join(root, info_file)
        file = open(info_file_name, 'w')
        for index, block in enumerate(self.block_list):
            image_save_root = os.path.join(root, str(index).zfill(6), 'image')
            print('---- {}'.format(image_save_root))
            z, y, x = self.location_list[index]
            info_line = image_save_root + ' ' + ' '.join([str(z).rjust(6), str(y).rjust(6), str(x).rjust(6)]) + '\n'
            file.write(info_line)
            save_image_3d(block, image_save_root)
        file.close()

    def divide_regular(self, block_size = Block_Size):
        """
        将图像数据拆分为若干三维小块
        :return:
        """
        self.z_cuting = 0
        self.y_cuting = 0
        self.x_cuting = 0
        while self.divide_flag:
            block, size = self._get_block(block_size = block_size)
            self.block_list.append(block)
            self.size_list.append(size)
        self.divide_flag = True

    def _get_block(self, block_size, step_ratio = 3 / 4):
        """
        按照当前切分位置和块大小，从三维图像数据中获取数据
        :param block_size:
        :param step_ratio: 步长比率
        :return:
        """
        block = np.zeros(shape = block_size)
        depth, height, width = block_size

        z_start = self.z_cuting          if self.z_cuting + depth  < self.depth  else (0 if self.depth  < depth  else self.depth  - depth)
        z_stop  = self.z_cuting + depth  if self.z_cuting + depth  < self.depth  else self.depth

        y_start = self.y_cuting          if self.y_cuting + height < self.height else (0 if self.height < height else self.height - height)
        y_stop  = self.y_cuting + height if self.y_cuting + height < self.height else self.height

        x_start = self.x_cuting          if self.x_cuting + width  < self.width  else (0 if self.width  < width  else self.width  - width)
        x_stop  = self.x_cuting + width  if self.x_cuting + width  < self.width  else self.width

        if self.x_cuting + width < self.width:
            self.x_cuting = self.x_cuting + round(width * step_ratio)
        else:
            self.x_cuting = 0
            if self.y_cuting + height < self.height:
                self.y_cuting = self.y_cuting + round(height * step_ratio)
            else:
                self.y_cuting = 0
                if self.z_cuting + depth < self.depth:
                    self.z_cuting = self.z_cuting + round(depth * step_ratio)
                else:
                    self.z_cuting = 0
                    self.divide_flag = False

        block_temp = self.image_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        self.location_list.append((z_start, y_start, x_start))
        if block_temp.shape == block_size:
            return block_temp, block_temp.shape
        else:
            block[:block_temp.shape[0], :block_temp.shape[1], :block_temp.shape[2]] = block_temp
            return block, block_temp.shape

    def divide_irregular(self, block_size = Block_Size, number = 500, number_mode = 'auto'):
        """
        将图像进行随机切块
        :param block_size: 块大小
        :param number: 块数量
        :param number_mode: 规定块数量 in [None, 'auto']
        :return:
        """
        raise NotImplementedError('在模型部署时候，不对神经元图像进行随机切割')


class Image3D_PATH(Image3D):
    """
    将文件路径中保存的一张张二维图像序列转换生成三维图像
    """
    def __init__(self, image_path = None, leaf_path = 'image'):
        """
        :param image_path: string，保存图像数据的路径名，这个路径下保存的图像尺寸必须相同，这个路径下可以有其他类型文件的存在
        """
        #self.image_path = os.path.join(image_path, 'image')
        self.image_path = os.path.join(image_path, leaf_path) if leaf_path != None else image_path
        assert os.path.isdir(self.image_path), self.image_path
        image = load_image_3d(image_root = self.image_path)
        super(Image3D_PATH, self).__init__(image = image)


if __name__ == '__main__':
    image_path = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/Neuron_Denoised/neuron_wavesnet_v3_haar/epoch_29/000020/image_pre_init'
    image3D = Image3D_PATH(image_path = image_path, leaf_path = None)
    root_save = '/home/liqiufu/PycharmProjects/MyDataBase/Vaa3D_picture/neuron_20/denoised_image_block'
    image3D.save_block(root = root_save, block_size = (90, 128, 128))