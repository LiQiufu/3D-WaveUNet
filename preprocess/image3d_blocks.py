"""
基于神经元图像生成图像块
"""

import numpy as np
from tools.image_3D_io import save_image_3d, load_image_3d
from constant import Block_Size, Block_Size_
import os, math
from random import randint

class Image3D():
    """
    这个类型描述一个三维图像，这个三维图像数据的第一个维度描述图像的层数（Z轴，通道数），第二个描述高度（Y轴，行数），第三个描述宽度（X轴，列数）
    """
    def __init__(self, image, label = None):
        """
        :param image: 三维数据
        """
        self.image_3d = image            # 三维 np.narray 类型
        self.label_3d = label
        if label is not None:
            self.label_existing = True
            assert self.image_3d.shape == self.label_3d.shape
        else:
            self.label_existing = False
        self.depth, self.height, self.width = image.shape
        self.divide_flag = True         #
        self.block_list = []
        if self.label_existing:
            self.block_list_label = []

    def shape(self):
        return self.depth, self.height, self.width

    def save(self, image_save_root = None, dim = 0, suffix = '.tiff'):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int, 切片维度
        :return:
        """
        save_image_3d(self.image_3d, image_save_root = os.path.join(image_save_root, 'image'), dim = dim, suffix = suffix)
        if self.label_existing:
            save_image_3d(self.label_3d, image_save_root = os.path.join(image_save_root, 'label'), dim = dim, suffix = suffix)

    def save_block(self, root, block_size = Block_Size, mode = 'both'):
        """
        保存切分得到的图像块
        :param root:
        :param mode: in ['regular', 'irregular', 'both']
                'regular' 表示将图像拆分为若干不想重叠的固定大小图像块
                'irregular' 表示从图像中随机提取固定大小的图像块
                'both' 表示两种方式都进行
        :return:
        """
        self.block_number = 0
        self.label_3d[self.label_3d != 0] = 1
        if not self.block_list:
            if mode == 'regular':
                self.divide_regular(block_size = block_size)
            elif mode == 'irregular':
                self.divide_irregular(block_size = block_size, root = root)
            elif mode == 'both':
                self.divide_regular(block_size = block_size)
                self.divide_irregular(block_size = block_size, root = root)
            else:
                raise ValueError("mode is in ['regular', 'irregular', 'both']")
        #for index, block in enumerate(self.block_list):
        #    image_save_root = os.path.join(root, str(index).zfill(6), 'image')
        #    print('---- {}'.format(image_save_root))
        #    save_image_3d(block, image_save_root)
        #    if self.label_existing:
        #        block_label = self.block_list_label[index]
        #        image_save_root = os.path.join(root, str(index).zfill(6), 'label')
        #        save_image_3d(block_label, image_save_root)

    def divide_regular(self, block_size = Block_Size):
        """
        将图像数据拆分为若干三维小块
        :return:
        """
        self.z_cuting = 0
        self.y_cuting = 0
        self.x_cuting = 0
        while self.divide_flag:
            block, block_label = self._get_block(block_size = block_size)
            self.block_list.append(block)
            if self.label_existing:
                self.block_list_label.append(block_label)
        self.divide_flag = True

    def _get_block(self, block_size):
        """
        按照当前切分位置和块大小，从三维图像数据中获取数据
        :param block_size:
        :return:
        """
        block = np.zeros(shape = block_size)
        if self.label_existing:
            block_label = np.zeros(shape = block_size)
        else:
            block_label = None
        depth, height, width = block_size
        z_start, y_start, x_start = self.z_cuting, self.y_cuting, self.x_cuting

        z_stop = self.z_cuting + depth  if self.z_cuting + depth  < self.depth  else self.depth
        y_stop = self.y_cuting + height if self.y_cuting + height < self.height else self.height
        x_stop = self.x_cuting + width  if self.x_cuting + width  < self.width  else self.width

        if self.x_cuting + width < self.width:
            self.x_cuting = self.x_cuting + width
        else:
            self.x_cuting = 0
            if self.y_cuting + height < self.height:
                self.y_cuting = self.y_cuting + height
            else:
                self.y_cuting = 0
                if self.z_cuting + depth < self.depth:
                    self.z_cuting = self.z_cuting + depth
                else:
                    self.z_cuting = 0
                    self.divide_flag = False

        block_temp = self.image_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        if self.label_existing:
            block_temp_label = self.label_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        else:
            block_temp_label = None
        if block_temp.shape == block_size:
            return block_temp, block_temp_label
        else:
            block[:block_temp.shape[0], :block_temp.shape[1], :block_temp.shape[2]] = block_temp
            if self.label_existing:
                block_label[:block_temp.shape[0], :block_temp.shape[1], :block_temp.shape[2]] = block_temp_label
            return block, block_label

    def divide_irregular(self, block_size = Block_Size, number = 500, number_mode = 'auto', root = './'):
        """
        将图像进行随机切块
        :param block_size: 块大小
        :param number: 块数量
        :param number_mode: 规定块数量 in [None, 'auto']
        :return:
        """
        if number_mode != 'auto':
            block_number = number
        elif number_mode == 'auto':
            block_number = self._get_block_number(block_size)
        else:
            raise ValueError("number_mode is in [None, 'auto']")
        print('===================== I am here, number = {}'.format(block_number))
        #   以下操作使得有一定量的空白块
        for index in range(block_number):
            image_save_root = os.path.join(root, str(self.block_number).zfill(6), 'image')
            label_save_root = os.path.join(root, str(self.block_number).zfill(6), 'label')
            print('{} ---- {}'.format(str(block_number).rjust(6), image_save_root))
            z_start, z_stop, y_start, y_stop, x_start, x_stop = self._get_block_boundary(block_size = block_size)
            block = self.image_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
            if block.shape == block_size:
                #self.block_list.append(block)
                block_temp = block
            else:
                block_temp = np.zeros(shape = block_size)
                block_temp[0:block.shape[0],0:block.shape[1],0:block.shape[2]] = block
                #self.block_list.append(block_temp)
            block_temp = np.array(block_temp, np.uint8)
            save_image_3d(block_temp, image_save_root, suffix = '.jpg')
            if self.label_existing:
                block_label = self.label_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                if block_label.shape == block_size:
                    #self.block_list_label.append(block_label)
                    block_label_temp = block_label
                else:
                    block_label_temp = np.zeros(shape = block_size)
                    block_label_temp[0:block_label.shape[0],0:block_label.shape[1],0:block_label.shape[2]] = block_label
                    #self.block_list_label.append(block_label_temp)
                block_label_temp[block_label_temp!=0] = 1
                block_label_temp = np.array(block_label_temp, np.uint8)
                save_image_3d(block_label_temp, label_save_root, suffix = '.png')
            self.block_number += 1
        #   以下操作确保足够量的含纤维图像块
        index = 0
        ratio_boudary = min(0.003, np.sum(self.label_3d) / (self.depth * self.height * self.width)) #if self.depth > min(self.height, self.width) else 0.003
        if block_number < 10:
            block_number *= 10
        elif block_number < 50:
            block_number *= 8
        elif block_number < 200:
            block_number *= 6
        else:
            block_number *= 4
        while index < block_number:
            image_save_root = os.path.join(root, str(self.block_number).zfill(6), 'image')
            label_save_root = os.path.join(root, str(self.block_number).zfill(6), 'label')
            print('{} ---- {}'.format(str(block_number).rjust(6), image_save_root))
            z_start, z_stop, y_start, y_stop, x_start, x_stop = self._get_block_boundary(block_size = block_size)
            if self.label_existing:
                block_label = self.label_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
                if block_label.shape == block_size:
                    # self.block_list_label.append(block_label)
                    block_label_temp = block_label
                else:
                    block_label_temp = np.zeros(shape = block_size)
                    block_label_temp[0:block_label.shape[0], 0:block_label.shape[1], 0:block_label.shape[2]] = block_label
                    # self.block_list_label.append(block_label_temp)
                block_label_temp[block_label_temp != 0] = 1
                if np.sum(block_label_temp) / Block_Size_ < ratio_boudary:
                    continue
                block_label_temp = np.array(block_label_temp, np.uint8)
                save_image_3d(block_label_temp, label_save_root, suffix = '.png')
            block = self.image_3d[z_start:z_stop, y_start:y_stop, x_start:x_stop]
            if block.shape == block_size:
                # self.block_list.append(block)
                block_temp = block
            else:
                block_temp = np.zeros(shape = block_size)
                block_temp[0:block.shape[0], 0:block.shape[1], 0:block.shape[2]] = block
                # self.block_list.append(block_temp)
            block_temp = np.array(block_temp, np.uint8)
            save_image_3d(block_temp, image_save_root, suffix = '.jpg')
            self.block_number += 1
            index += 1

    def _get_block_number(self, block_size):
        depth, height, width = block_size
        #if depth >= self.depth or height >= self.height or width >= self.width:
        print('{}, {}'.format(block_size, self.image_3d.shape))
        #    return 0
        #else:
        return math.ceil(self.depth / depth) * math.ceil(self.height / height) * math.ceil(self.width / width)

    def _get_block_boundary(self, block_size = Block_Size):
        depth, height, width = block_size
        z_start = randint(0, self.depth - depth) if self.depth > depth else 0
        z_stop = z_start + depth if self.depth > depth else self.depth
        y_start = randint(0, self.height - height) if self.height > height else 0
        y_stop = y_start + height if self.height > height else self.height
        x_start = randint(0, self.width - width) if self.width > width else 0
        x_stop = x_start + width if self.width > width else self.width
        return z_start, z_stop, y_start, y_stop, x_start, x_stop


class Image3D_PATH(Image3D):
    """
    将文件路径中保存的一张张二维图像序列转换生成三维图像
    """
    def __init__(self, image_path = None):
        """
        :param image_path: string，保存图像数据的路径名，这个路径下保存的图像尺寸必须相同，这个路径下可以有其他类型文件的存在
        """
        self.image_path = os.path.join(image_path, 'image')
        assert os.path.isdir(self.image_path), self.image_path
        self.label_path = os.path.join(image_path, 'label')
        image = load_image_3d(image_root = self.image_path)
        if os.path.isdir(self.label_path):
            label = load_image_3d(image_root = self.label_path)
        else:
            label = None
        super(Image3D_PATH, self).__init__(image = image, label = label)


def divide_image(root='/home/liqiufu/PycharmProjects/MyDataBase/DataBase_15'):
    sub_pathes = ['angle_0', 'angle_60', 'angle_120', 'angle_180', 'angle_240', 'angle_300']
    # neuron_name_list = ['N041', 'N042', 'N043', 'N044', 'N056', 'N068', 'N075']
    neuron_name_list = ['000000', '000011', '000025', '000036', '000047', '000059', '000072', '000083', '000094',
                        '000001', '000012', '000026', '000037', '000048', '000060', '000073', '000084', '000095',
                        '000002', '000013', '000027', '000038', '000049', '000061', '000074', '000085',
                        '000003', '000015', '000028', '000039', '000050', '000064', '000075', '000086',
                        '000004', '000016', '000029', '000040', '000051', '000065', '000076', '000087',
                        '000005', '000017', '000030', '000041', '000052', '000066', '000077', '000088',
                        '000006', '000018', '000031', '000042', '000053', '000067', '000078', '000089',
                        '000007', '000019', '000032', '000043', '000054', '000068', '000079', '000090',
                        '000008', '000022', '000033', '000044', '000055', '000069', '000080', '000091',
                        '000009', '000023', '000034', '000045', '000056', '000070', '000081', '000092',
                        '000010', '000024', '000035', '000046', '000057', '000071', '000082', '000093']
    neuron_name_list.sort()
    # neuron_name_list = ['000007',]
    for sub_path in sub_pathes:
        for neuron_name in neuron_name_list:
            current_root = os.path.join(root, sub_path, neuron_name)
            image_path = os.path.join(current_root)
            print('generate_label - processing -- {}'.format(image_path))
            image3d = Image3D_PATH(image_path = image_path)
            image3d.save_block(root = os.path.join(current_root, 'block'), mode = 'regular')

def move(root, save_root):
    angle_list = os.listdir(root)
    for angle in angle_list:
        root_angle = os.path.join(root, angle)
        if not os.path.isdir(root_angle):
            continue
        root_angle_save = os.path.join(save_root, angle)
        if not os.path.isdir(root_angle_save):
            os.mkdir(root_angle_save)
        neuron_list = os.listdir(root_angle)
        for neuron in neuron_list:
            root_neuron = os.path.join(root, angle, neuron)
            if not os.path.isdir(root_neuron):
                continue
            root_neuron_save = os.path.join(save_root, angle, neuron)
            if not os.path.isdir(root_neuron_save):
                os.mkdir(root_neuron_save)
            command = 'mv {} {}'.format(os.path.join(root_neuron, 'block'), root_neuron_save)
            os.system(command)

def divide_image_new():
    root = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_4_new'
    root_save = '/data/liqiufu/DATA/Neuron_DataBase_for_Denoiser/DataBase_5_new'
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for neuron_name in neuron_name_list:
        current_root = os.path.join(root, neuron_name)
        image_path = os.path.join(current_root)
        print('generate_label - processing -- {}'.format(image_path))
        image3d = Image3D_PATH(image_path = image_path)
        image3d.save_block(root = os.path.join(root_save, neuron_name), mode = 'irregular')
        del image3d

def resave():
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_5/000019_0'
    block_name_list = os.listdir(root)
    block_name_list.sort()
    for block_name in block_name_list:
        block_path = os.path.join(root, block_name)
        #image_path = os.path.join(block_path, 'image')
        #label_path = os.path.join(block_path, 'label')
        image = Image3D_PATH(image_path = block_path)
        image.label_3d *= 255
        #label = Image3D_PATH(image_path = label_path)
        #label = Image3D_PATH(image_path = label_path)
        image.save(image_save_root = block_path)
        #label.save(image_save_root = label_path)


if __name__ == '__main__':
    #resave()
    resave()
    #root = '/home/liqiufu/PycharmProjects/MyDataBase/DataBase_15'
    #save_root = '/home/liqiufu/PycharmProjects/MyDataBase/DataBase_block'
    #move(root, save_root)