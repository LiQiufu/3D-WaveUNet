"""
这个脚本主要建立于 2020/09/03
主要功能是根据神经元的追踪重建结果生成对应的标签
之前也构建过能够实现类似功能的脚本，但是结果令人很不满意
这个脚本里形状参数为 depth, height, width; 坐标顺序也相应调整为 z, y, x
坐标顺序 (z, y, x)
"""

import os
import cv2
import numpy as np
from pprint import pprint
from tools.math_tool import get_coordinate_along_vector, get_around_points
from collections import OrderedDict
from tools.image_3D_io import save_image_3d, load_image_3d
import torch.nn.functional as F
import torch
import math
from constant import NEURON_NOT_IN_TRAIN_TEST

class Image3D():
    """
    这个类型描述一个三维图像，这个三维图像数据的第一个维度描述图像的层数（Z轴，通道数），第二个描述高度（Y轴，行数），第三个描述宽度（X轴，列数）
    """

    def __init__(self, depth, heigh, width, image_3d):
        self.depth, self.height, self.width = depth, heigh, width
        self.image_3d = image_3d  # 三维 np.narray 类型
        self.check_size_info()

    def shape(self):
        return (self.depth, self.height, self.width)

    def check_size_info(self):
        """
        检查当前是否有数据，以及数据尺寸是否和相应的属性一致
        :return:
        """
        assert isinstance(self.image_3d, np.ndarray)
        shape = self.image_3d.shape
        assert self.depth  == shape[0]
        assert self.height == shape[1]
        assert self.width  == shape[2]

    def save(self, image_save_root, dim):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int, 切片维度
        :return:
        """
        print('     in saving / {}, {} ...'.format(self.shape(), image_save_root))
        self.image_3d = np.array(self.image_3d, np.uint8)
        save_image_3d(self.image_3d, image_save_root = image_save_root, dim = dim)

    def refresh_shape(self):
        """
        获取当前三维图像数据的尺寸信息
        :return:
        """
        assert isinstance(self.image_3d, np.ndarray)
        self.depth, self.height, self.width = self.image_3d.shape

    def add_slice(self, image_2d):
        """
        将一个二维图像矩阵贴到当前三维图像数据上
        :param image_2d: np.array
        :return:
        """
        if not isinstance(self.image_3d, np.ndarray):
            height, width = image_2d.shape
            self.image_3d = image_2d.reshape((1, height, width))
        else:
            height, width = image_2d.shape
            assert height == self.height and width == self.width
            image_2d = image_2d.reshape((1, height, width))
            self.image_3d = np.concatenate((self.image_3d, image_2d), axis = 0)
        self.refresh_shape()

    def extent_image3d(self, another_image_3d):
        """
        将另一个三维图像数据接到当前图像数据后
        :param another_image_3d: np.array
        :return:
        """
        if not isinstance(self.image_3d, np.ndarray):
            self.image_3d = another_image_3d
        else:
            _, height, width = another_image_3d.shape
            assert height == self.height and width == self.width
            self.image_3d = np.concatenate((self.image_3d, another_image_3d))
        self.refresh_shape()

    def concatenate(self, another_image3d):
        """
        将当前 NeuronImage3D 和另一个该类型的对象连接起来
        :param another_image3d:
        :return:
        """
        assert isinstance(another_image3d, Image3D)
        if not isinstance(self.image_3d, np.ndarray):
            self.image_3d = another_image3d.image_3d
        else:
            assert self.height == another_image3d.height and self.width == another_image3d.width
            self.image_3d = np.concatenate((self.image_3d, another_image3d.image_3d))
        self.refresh_shape()

    def resize(self, shape_new = (64, 256, 256)):
        """
        将图像矩阵的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :return:
        """
        print('     in resizing, from {} to {} ...'.format(self.shape(), shape_new))
        self.image_3d = torch.tensor(self.image_3d, dtype = torch.float)
        self.image_3d = torch.unsqueeze(self.image_3d, dim = 0)
        self.image_3d = torch.unsqueeze(self.image_3d, dim = 0)
        self.image_3d = F.interpolate(self.image_3d, size = shape_new)
        self.image_3d = torch.squeeze(self.image_3d, dim = 0)
        self.image_3d = torch.squeeze(self.image_3d, dim = 0)
        self.image_3d = self.image_3d.numpy()
        self.refresh_shape()

    def cut(self, shape = (32,128,128), coor = (0,0,0)):
        """
        对当前图像进行裁剪，裁剪原点为 coor，裁剪尺寸为 shape
        :param shape: (depth, height, width)
        :param coor: (z0, y0, x0)
        :return:
        """
        self.check_size_info()
        assert shape[0] <= self.depth  - coor[0], '{} > {} - {}'.format(shape[0], self.depth,  coor[0])
        assert shape[1] <= self.height - coor[1], '{} > {} - {}'.format(shape[1], self.height, coor[1])
        assert shape[2] <= self.width  - coor[2], '{} > {} - {}'.format(shape[2], self.width,  coor[2])
        assert coor[0] >= 0 and coor[1] >= 0 and coor[2] >= 0
        assert shape[0] >= 0 and shape[1] >= 0 and shape[2] >= 0
        shape = list(shape)
        shape[0] = self.depth  - coor[0] if coor[0] + shape[0] >= self.depth  else shape[0]
        shape[1] = self.height - coor[1] if coor[1] + shape[1] >= self.height else shape[1]
        shape[2] = self.width  - coor[2] if coor[2] + shape[2] >= self.width  else shape[2]
        assert shape[0] >= 0 and shape[1] >= 0 and shape[2] >= 0
        image_3D_cut  = self.image_3d[coor[0]:(coor[0] + shape[0]),
                                      coor[1]:(coor[1] + shape[1]),
                                      coor[2]:(coor[2] + shape[2])]
        return Image3D(depth = shape[0], heigh = shape[1], width = shape[2], image_3d = image_3D_cut)

    def cut_as_label(self, label_3d):
        """
        按照标签 label_3d 的标记提取当前图像中的相关目标，将 label_3d 中标记为 0 的位置像素置为 0
        :param label_3d:
        :return:
        """
        assert isinstance(label_3d, Image3D)
        assert label_3d.depth == self.depth   and \
               label_3d.height == self.height and \
               label_3d.width == self.width
        self.image_3d[label_3d.image_3d == 0] = 0

    def add_noise(self, mean = 0, var = 0.01):
        """
        向三维图像中添加随机噪声
        :param mean: 噪声均值
        :param var: 噪声方差
        :return:
        """
        self.image_3d = np.array(self.image_3d /  255, dtype = float)
        noise = np.random.normal(mean, var ** 0.5, self.shape())
        self.image_3d += noise
        self.image_3d = np.clip(self.image_3d, 0., 1.0)
        self.image_3d = np.uint8(self.image_3d * 255)


class Image3D_PATH(Image3D):
    """
    将文件路径中保存的一张张二维图像序列转换生成三维图像
    """

    def __init__(self, image_path = None):
        """
        :param image_path: string，保存图像数据的路径名，这个路径下保存的图像尺寸必须相同，这个路径下可以有其他类型文件的存在
        """
        assert os.path.isdir(image_path)
        self.image_path = image_path
        image_3d = load_image_3d(image_root = self.image_path)
        self.depth, self.height, self.width = image_3d.shape
        super(Image3D_PATH, self).__init__(depth = self.depth, heigh = self.height, width = self.width, image_3d = image_3d)
        self.suffix = ['.png', '.jepg', '.jpg', '.tiff', '.bmp', '.tif']

    def save(self, image_save_root = None, dim = 0):
        """
        将当前三维图像数据进行保存
        :param image_save_root: string 保存路径
        :param dim: int 切片维度
        :return:
        """
        if image_save_root == None:
            image_save_root = self.image_path + '_1'
        if not os.path.isdir(image_save_root):
            os.makedirs(image_save_root)
        super(Image3D_PATH, self).save(image_save_root = image_save_root, dim = dim)


class NeuronNode():
    """
    这个类描述神经元节点，主要包括 编号、类型、三维坐标、半径、父节点等属性；注意这里节点的坐标可能不为整数
    """

    def __init__(self, id = 1, type = 18, x = 0., y = 0., z = 0., radius = 1., p_id = -1, processed = False, resolution = (1., 1., 1.)):
        """
        :param id:
        :param type:
        :param x:
        :param y:
        :param z:
        :param radius:
        :param p_id:
        :param processed:
        :param resolution: 图像的成像分辨率, 比如 MOST 图像的分辨率可能是 1 /mu m \times 0.35 /mu m \times 0.35 /mu m；在实际使用时，建议使用其默认值 (1.,1.,1.)
        """
        self.id = id
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.p_id = p_id
        self.child_id = list()  # 初始化为空，在生成神经元节点列表过程中被赋值，即被真正初始化
        self.processed = processed  # 表示当前节点是否被处理，若被处理则这个值被重置为 True
        self.resolution_real = resolution
        self.resolution = self._get_resolution(resolution)
        self._check()

    def _check(self):
        assert isinstance(self.id, int)
        assert isinstance(self.type, int)  #神经元节点类型不一定保存
        assert self.x >= 0, 'id = {}, x = {}'.format(self.id, self.x)
        assert self.y >= 0, 'id = {}, y = {}'.format(self.id, self.y)
        assert self.z >= 0
        assert self.radius >= 0, 'id = {}, radius = {}'.format(self.id, self.radius)
        assert isinstance(self.p_id, int)

    def _get_resolution(self, resolution = (1., 1., 1.)):
        reso_z = resolution[0]
        reso_y = resolution[1]
        reso_x = resolution[2]
        m = min(resolution[0], resolution[1], resolution[2])
        return m / reso_z, m / reso_y, m / reso_x

    def __str__(self):
        return ' '.join([str(self.id),
                         str(self.type),
                         str(self.x),
                         str(self.y),
                         str(self.z),
                         str(self.radius),
                         str(self.p_id),
                         str(self.child_id)])

    def get_around_points(self, r_offset = 0, depth = 64, height = 256, width = 256):
        """
        获取以节点坐标为中心点，self.radius为半径内的左右坐标点
        :param r_offset: 有些情况下要对该节点的半径值增加一个缓冲值
        :return:
        """
        #if height >= 512 and width >= 512 and self.radius <= 3:
        #    r_offset = 1
        return get_around_points(point = (self.z, self.y, self.x), radius = self.radius + r_offset, resolution = self.resolution)

    def get_connect_points(self, another_node, r_offset = 0, depth = 64, height = 256, width = 256):
        """
        获取该节点与其某个子节点连线之间的坐标点
        :param another_node: NeuronNode类型
        :return:
        """
        #if height >= 512 and width >= 512 and self.radius < 3:
        #    r_offset = 1
        assert isinstance(another_node, NeuronNode)
        coordinates = []
        radius = (self.radius + another_node.radius) / 2
        distance = self._distance_with_another(another_node = another_node)
        # print('distance = {}, radius = {}'.format(distance, radius))
        if distance < radius:
            return coordinates
        n = math.ceil(distance / radius) - 1
        radius_step = (self.radius - another_node.radius) / (n + 1)
        vector = (another_node.z - self.z, another_node.y - self.y, another_node.x - self.x)
        length = self.radius
        for i in range(n):
            radius_current = self.radius - (i + 1) * radius_step + r_offset
            assert radius_current > 0, 'i = {}/{}, self.radius = {}, radius_step = {}, radius_current = {}, another_node = {}'.format(i, n, self.radius, radius_step, radius_current, str(another_node))
            #print('index = {}, radius_current = {}'.format(i, radius_current))
            z, y, x = get_coordinate_along_vector(point_start = (self.z, self.y, self.x), vector = vector, distance = length)
            coordinate_current = get_around_points(point = (z, y, x), radius = radius_current, resolution = self.resolution)
            coordinates.extend(coordinate_current)
            length += radius_current
            if length >= distance:
                break
        return coordinates

    def _distance_with_another(self, another_node):
        """
        检查当前节点与另一个节点的距离
        :param another_node:
        :return: 返回与另一个节点之间的距离
        """
        assert isinstance(another_node, NeuronNode)
        x = another_node.x - self.x
        y = another_node.y - self.y
        z = another_node.z - self.z
        return math.sqrt(x ** 2 + y ** 2 + z ** 2)

    def _direction_vector_with_another(self, another_node):
        """
        检查与另一个节点之间的方向向量
        :param another_node:
        :return:
        """
        assert isinstance(another_node, NeuronNode)
        vector = another_node.z - self.z, \
                 another_node.y - self.y, \
                 another_node.x - self.x
        distance = self._distance_with_another(another_node = another_node)
        return vector[0] / distance, vector[1] / distance, vector[2] / distance

    def real_distance_with_another(self, another_node):
        assert isinstance(another_node, NeuronNode)
        x = (another_node.x - self.x) * self.resolution_real[0]
        y = (another_node.y - self.y) * self.resolution_real[1]
        z = (another_node.z - self.z) * self.resolution_real[2]
        return math.sqrt(x ** 2 + y ** 2 + z ** 2)


class NeuronNodeList():
    """
    保存多个神经节点的有序字典，字典的键是相应神经节点的编号
    """

    def __init__(self, depth = 32, height = 128, width = 128, resolution = (1., 1., 1.)):
        """
        通常每个神经元节点列表都对应着一个神经元三维图像，这里 self.width、self.height、self.depth 分别是这个三维图像的宽、高、深
        self.depth, self.height, self.widht 实际规定了当前神经元节点列表所处的空间大小
        整个神经元节点列表事实上是通过子函数 add_neuron_node() 生成的
        """
        self._neuron_node_list = OrderedDict()      # 保存各个神经元节点
        self.width = width      # 神经元所处立方体（对应着相应的三维图像数据）的尺寸
        self.height = height
        self.depth = depth
        assert isinstance(self.depth, int) and isinstance(self.height, int) and isinstance(self.width, int)
        self.keys = list()
        self.child_relation = dict()
        self.resolution = resolution
        self._get_size_info()

    def add_neuron_node(self, neuron_node):
        """
        往有序字典中添加一个神经元节点
        :param neuron_node:
        :return:
        """
        assert isinstance(neuron_node, NeuronNode)
        if neuron_node.id not in self.keys:
            self.keys.append(neuron_node.id)
            self._neuron_node_list[neuron_node.id] = neuron_node        # 将新的神经元节点加入当前列表
        # 处理其子节点
        if neuron_node.id in self.child_relation:                   # 说明列表中之前加入的节点中存在当前节点的子节点
            neuron_node.child_id = self.child_relation[neuron_node.id]
        else:                                                       # 列表中已有的节点均不是当前节点的子节点
            self.child_relation[neuron_node.id] = list()
        # 处理其父节点
        if neuron_node.p_id in self.child_relation:
            self.child_relation[neuron_node.p_id].append(neuron_node.id) # 更新当前列表中某个神经元节点的 child_id 属性
        else:
            self.child_relation[neuron_node.p_id] = [neuron_node.id,]
        if neuron_node.p_id in self.keys:
            self._neuron_node_list[neuron_node.p_id].child_id = self.child_relation[neuron_node.p_id]

    def _get_size_info(self):
        """
        获取当前神经元节点列表在空间中的外接立方体的尺寸，以及该立方体在完整三维神经元图像中的左下角坐标
        :return:
        """
        if len(self._neuron_node_list) == 0:
            return
        z_list = [self._neuron_node_list[key].z for key in self._neuron_node_list.keys()]
        y_list = [self._neuron_node_list[key].y for key in self._neuron_node_list.keys()]
        x_list = [self._neuron_node_list[key].x for key in self._neuron_node_list.keys()]
        z_min = min(z_list[10:20])
        self.init_z0 = round(min(z_list))
        self.init_y0 = round(min(y_list))
        self.init_x0 = round(min(x_list))
        self.real_depth  = round(max(z_list)) - self.init_z0 + 1
        self.real_height = round(max(y_list)) - self.init_y0 + 1
        self.real_width  = round(max(x_list)) - self.init_x0 + 1
        #assert self.real_depth  <= self.depth,  '{} <= {}, which is wrong'.format(self.real_depth,  self.depth)
        #assert self.real_height <= self.height, '{} <= {}, which is wrong'.format(self.real_height, self.height)
        #assert self.real_width  <= self.width,  '{} <= {}, which is wrong'.format(self.real_width,  self.width)

    def generate_label_3d(self, r_offset = 0.25, label_mark = 255):
        """
        按照所有节点信息，生成3D标签矩阵；
        这个函数在运行过程中，会破坏self._neuron_node_list中每个神经节点的属性值child_id
        因此在本函数最后重新刷新了self._neuron_node_list中每个神经节点的属性值child_id
        :param r_offset: int，每个节点半径的增加量
        :return:
        """
        self.label_3d = Image3D(depth = self.depth, heigh = self.height, width = self.width,
                                image_3d = np.zeros((self.depth, self.height, self.width), dtype = np.uint8))
        keys = list(self._neuron_node_list.keys())
        while keys != []:
            key = keys[0]
            while True:
                if key in keys:
                    neuron_node = self._neuron_node_list[key]
                else:
                    break
                #print('id = {} -- ({:.2f}, {:.2f}, {:.2f}), processed > {} - p_id = {} -- child_id ==> {}'.
                #      format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z, neuron_node.processed, neuron_node.p_id, neuron_node.child_id))
                if neuron_node.processed:
                    # 当前节点已经被处理
                    if len(neuron_node.child_id) == 0:
                        # 节点本身先于其所有子节点被处理的节点（即有多个子节点）
                        # 若当前节点已被处理，并相应处理了其每个子节点，则将其删除
                        # print('{} -- ({},{},{}) -- {}/{} - {} remove 0'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                        # neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                        if neuron_node.p_id not in keys:
                            keys.remove(key)
                            break
                        elif self._neuron_node_list[neuron_node.p_id].processed:
                            keys.remove(key)
                            break
                        else:
                            key = neuron_node.p_id
                    else:
                        # 节点本身已被处理，但仍有子节点未被处理的，在此处跳转到其子节点的处理
                        neuron_node_child = self._neuron_node_list[neuron_node.child_id[0]]  # 取出当前节点的第一个子节点
                        coordinates = neuron_node.get_connect_points(neuron_node_child, r_offset = r_offset, depth = self.depth, height = self.height, width = self.width)
                        self._refresh_label_3d(coordinates, label_mark = label_mark)
                        if not neuron_node_child.processed:
                            key = self._neuron_node_list[key].child_id.pop(0)
                            continue
                else:
                    # 节点本身未被处理的，在此处处理
                    coordinates = neuron_node.get_around_points(r_offset = r_offset, depth = self.depth, height = self.height, width = self.width)
                    self._refresh_label_3d(coordinates, label_mark = label_mark)
                    self._neuron_node_list[key].processed = True
                    if len(neuron_node.child_id) > 0:
                        # 节点有子节点的，紧接着跳转过去处理其子节点
                        neuron_node_child = self._neuron_node_list[neuron_node.child_id[0]]     # 取出当前节点的第一个子节点
                        self._neuron_node_list[key].child_id = neuron_node.child_id
                        coordinates = neuron_node.get_connect_points(neuron_node_child, r_offset = r_offset, depth = self.depth, height = self.height, width = self.width)
                        self._refresh_label_3d(coordinates, label_mark = label_mark)
                        key = neuron_node.child_id[0]
                        neuron_node.child_id.pop(0)
                        continue
                    #else:
                        # 节点本身后于其所有子节点被处理的节点（即没有子节点，这也意味着到达了一条神经纤维的末端），在此处被删除
                        # print('{} -- ({},{},{}) -- {}/{} - {} remove 1'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                        #                                      neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                        #keys.remove(key)
                        #break
        self.label_3d.refresh_shape()
        self._refresh_childID()  # 刷新修正节点列表里每个节点的属性值 child_id

    def _refresh_label_3d(self, coordinates, label_mark = 1):
        """
        将指定坐标处的label更新为 label_mark
        :param coordinates:
        :param label_mark:
        :return:
        """
        for coordinate in coordinates:
            z, y, x = coordinate
            if z < 0 or z >= self.depth or y < 0 or y >= self.height or x < 0 or x >= self.width:
                continue
            self.label_3d.image_3d[z,y,x] = label_mark

    def _refresh_childID(self):
        """
        根据 self._neuron_node_list 中已有的神经节点，更新每个节点的属性值 child_id
        :param neuron_node:
        :return:
        """
        keys = self._neuron_node_list.keys()
        for key in keys:
            self._neuron_node_list[key].processed = False
            self._neuron_node_list[key].child_id = list()   # 清空当前神经元节点列表中每个神经元节点的 child_id 属性值
        neuronnodelist = NeuronNodeList(depth = self.depth, height = self.height, width = self.width, resolution = self.resolution)
        for index, key in enumerate(keys):
            neuron_node = self._neuron_node_list[key]
            neuronnodelist.add_neuron_node(neuron_node)     # 通过向 neuronnodelist 中逐个添加神经元节点重置每个节点的 child_id 属性
        self._neuron_node_list = neuronnodelist._neuron_node_list
        #del neuronnodelist

    def the_total_length(self):
        """
        计算整个神经元节点列表的长度
        :return:
        """
        self.length = 0.0
        keys = list(self._neuron_node_list.keys())
        while keys != []:
            key = keys[0]
            while True:
                neuron_node = self._neuron_node_list[key]
                #print('{} -- ({},{},{}) -- {}/{} - {}'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                #neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                if len(neuron_node.child_id) == 0:
                    # 节点本身先于其所有子节点被处理的节点（即有多个子节点）
                    # 若当前节点已被处理，并相应处理了其每个子节点，则将其删除
                    # print('{} -- ({},{},{}) -- {}/{} - {} remove 0'.format(neuron_node.id, neuron_node.x, neuron_node.y, neuron_node.z,
                    # neuron_node.child_id, neuron_node.processed, neuron_node.p_id))
                    neuron_node.processed = True
                    keys.remove(key)
                    break
                else:
                    key = self._neuron_node_list[key].child_id.pop(0)
                    self.length += neuron_node.real_distance_with_another(self._neuron_node_list[key])
                    continue
        self._refresh_childID()  # 刷新修正节点列表里每个节点的属性值 child_id

    def shape(self):
        return self.depth, self.height, self.width

    def __len__(self):
        return self._neuron_node_list.__len__()

    def save(self, saved_file_name = './save.swc'):
        """
        将某个神经元节点列表保存为 swc 文件的格式
        :param saved_file_name:
        :return:
        """
        assert saved_file_name != None
        assert self.height != None
        file = open(saved_file_name, 'w')
        for key in self._neuron_node_list.keys():
            neuron_node = self._neuron_node_list[key]
            line = '{} {} {:.3f} {:.3f} {:.3f} {:.3f} {}\n'.format(neuron_node.id,
                                                                  neuron_node.type,
                                                                  neuron_node.x,
                                                                  self.height - neuron_node.y - 1,
                                                                  neuron_node.z,
                                                                  neuron_node.radius,
                                                                  neuron_node.p_id)
            file.write(line)
        file.close()

    def change_other_id(self, another_node_list):
        """
        根据当前节点列表长度，修改另一个节点列表的编号和父节点编号
        :param another_noiselist:
        :return:
        """
        assert isinstance(another_node_list, NeuronNodeList)
        length = self.__len__() + 1
        keys = list(another_node_list.keys())
        for index, key in enumerate(keys):
            node = another_node_list._neuron_node_list[key]
            id_new = index + length     # 节点新编号
            for iden in node.child_id:  # 处理节点各个字节点的父节点编号
                node_child = another_node_list._neuron_node_list[iden]
                node_child.p_id = id_new
                node_child.processed = 1
            if (node.p_id not in keys) and (node.processed == 0):   # 处理节点的父节点编号
                node.p_id = -1          # 局部根节点
            node.id = id_new            # 更新节点编号
        another_node_list._refresh_childID()
        return another_node_list

    def concatenate(self, another_nodelist):
        """
        将当前 NeuronNodeList 和另一个该类型的对象连接起来，只是将被处理对象的节点列表加入当前节点列表中，不考虑其他参数
        :param another_nodelist:
        :return:
        """
        assert isinstance(another_nodelist, NeuronNodeList)
        another_nodelist = self.change_other_id(another_nodelist)
        for index, key in enumerate(another_nodelist.keys()):
            node = another_nodelist._neuron_node_list[key]
            self.add_neuron_node(neuron_node = node)

    def cut_whole(self, redundancy = (5, 10, 10)):
        """
        将神经元完整裁剪，实际上是做了一次平移，这个操作会改变当前神经元节点列表，并改变节点列表所处的空间大小
        :param redundancy: 在完整裁剪时候，在三个轴向上保留的冗余，以使得神经元不是刚刚好贴在裁剪后的图像边儿上的
                            这里需要注意，轴顺序是 depth(Z), height(Y), width(X), 形状shape的三个参数顺序与此相同
                                        坐标顺序是 Z(depth), Y(height), X(width)
                            因此这里三元组redundancy的顺序为depth(Z), height(Y), width(X)
        :return:
        """
        coor = [c0 - c1 for (c0, c1) in zip((self.init_z0, self.init_y0, self.init_x0), redundancy)]
        coor = [c if c > 0 else 0 for c in coor]    # 确定平移后的左下角坐标
        # 确定平移裁剪后的神经元节点列表所处的空间大小，神经元节点列表的实际空间大小无法改变
        shape = [s + 2 * c for (s, c) in zip((self.real_depth, self.real_height, self.real_width), redundancy)]
        shape[0] = shape[0] if shape[0] <= self.depth  - coor[0] else self.depth  - coor[0]
        shape[1] = shape[1] if shape[1] <= self.height - coor[1] else self.height - coor[1]
        shape[2] = shape[2] if shape[2] <= self.width  - coor[2] else self.width  - coor[2]
        #print('depth = {}, height = {}, width = {}'.format(self.depth, self.height, self.width))
        #print('shape = {}, coor = {}'.format(shape, coor))
        self._change_size_info(coor = coor, shape = shape)

    def _change_size_info(self, coor = (0,0,0), shape = (64, 256, 256)):
        """
        修改当前神经元节点列表在空间中的外接立方体的尺寸，以及该立方体在完整三维神经元图像中的左下角坐标
        :param shape: 将神经元节点列表所处立方体的空间大小变换为 shape(depth, height, width)
        :param coor: 将神经元节点列表所处立方体的原点坐标平移至此， Z(depth), Y(height), X(width)，
                      这里移动的是坐标原点，相当于对原图像进行裁剪
        :return:
        """
        #assert shape[0] <= self.depth  - coor[0], '{} > {} - {}'.format(shape[0], self.depth,  coor[0])
        #assert shape[1] <= self.height - coor[1], '{} > {} - {}'.format(shape[1], self.height, coor[1])
        #assert shape[2] <= self.width  - coor[2], '{} > {} - {}'.format(shape[2], self.width,  coor[2])
        assert coor[0] <= self.init_z0 and \
               coor[1] <= self.init_y0 and \
               coor[2] <= self.init_x0          # 平移的位置不能超过当前 neuronnodelist 的位置
        assert shape[0] >= self.real_depth
        assert shape[1] >= self.real_height
        assert shape[2] >= self.real_width      # 修改后的空间大小要大于 neuronnodelist 的实际大小
        self.depth  = shape[0]
        self.height = shape[1]
        self.width  = shape[2]                  # 修改空间大小
        for key in self._neuron_node_list.keys():   # 平移各个节点
            self._neuron_node_list[key].z -= coor[0]
            self._neuron_node_list[key].y -= coor[1]
            self._neuron_node_list[key].x -= coor[2]
        self._get_size_info()

    def resize(self, shape_new = (64, 256, 256)):
        """
        将神经元节点列表所处的立方体的尺寸缩放为 shape_new
        :param shape_new: depth(Z), height(Y), width(X)
        :return:
        """
        r_z = shape_new[0] / self.depth
        r_y = shape_new[1] / self.height
        r_x = shape_new[2] / self.width
        r_r = min(r_x, r_y, r_z)
        for key in self.keys:
            node = self._neuron_node_list[key]
            node.x = node.x * r_x
            node.y = node.y * r_y
            node.z = node.z * r_z
            node.radius = max(node.radius * r_r, 1)
            self._neuron_node_list[key] = node
        self.depth  = shape_new[0]
        self.height = shape_new[1]
        self.width  = shape_new[2]


class NeuronNodeList_SWC(NeuronNodeList):
    """
    从一个swc文件生成一个神经元节点列表 NeuronNodeList
    这个类描述某个 swc 文件中保存的所有神经元节点，保存为字典形式
    在某些swc文件中，给出了神经节点的编号、坐标，和神经节点之间的父子关系，但没有给出每个神经节点的半径（此时所有的半径均被设置为1）
    """

    def __init__(self, file_swc, depth = 32, height = 128, width = 128, resolution = (1, 1, 1)):
        """
        :param file_swc: 当前神经元节点列表保存文件
        :param depth: 神经元节点列表所处空间大小
        :param height:
        :param width:
        :param resolution: 对应神经元图像的采样物理分辨率
        """
        super(NeuronNodeList_SWC, self).__init__(depth = depth, height = height, width = width, resolution = resolution)
        assert file_swc.endswith('swc')
        self.swc_file_name = file_swc
        self.neuron_node_list()

    def neuron_node_list(self):
        """
        将 file_swc 文件中的每行信息生成 NeuronNode 类型后保存到 self._neuron_node_list 中，以其 ID 为键值
        :return:
        """
        node_line_list = (line.strip() for line in open(self.swc_file_name, 'r') if line[0] != '#')
        for node_line in node_line_list:
            if node_line == '':
                continue
            e = node_line.split()
            id = int(e[0])
            type = int(e[1])
            x = float(e[2]) if float(e[2]) >= 0 else 0
            y = self.height - float(e[3]) - 1 if self.height - float(e[3]) - 1 >= 0 else 0
            z = float(e[4]) if float(e[4]) >= 0 else 0
            radius = float(e[5])
            p_id = int(e[6])
            processed = False
            node = NeuronNode(id = id, type = type, x = x, y = y, z = z,
                              radius = radius, p_id = p_id, resolution = self.resolution, processed = False)
            self.add_neuron_node(node)

        super(NeuronNodeList_SWC, self)._get_size_info()

    def save(self, saved_file_name = './save.swc'):
        """
        将某个神经元节点列表保存为 swc 文件的格式
        :param saved_file_name:
        :return:
        """
        if saved_file_name == None:
            saved_file_name, suffix = os.path.splitext(self.swc_file_name)
            saved_file_name += ('_1' + suffix)
        super(NeuronNodeList_SWC, self).save(saved_file_name = saved_file_name)


class ImageWithLabel():
    """
    联合三维图像和三维标签的类型，对图像数据和标签进行协同操作
    """

    def __init__(self, image_3d, neuron_node_list):
        self.image3d = image_3d
        self.neuronnodelist = neuron_node_list
        self.label3d = None
        self._get_the_shape()

    def _get_the_shape(self):
        self.depth = self.image3d.depth
        self.height = self.image3d.height
        self.width = self.image3d.width
        assert self.depth == self.neuronnodelist.depth
        assert self.height == self.neuronnodelist.height
        assert self.width == self.neuronnodelist.width

    def shape(self):
        return (self.depth, self.height, self.width)

    def size(self):
        """
        返回图像大小
        :return:
        """
        return self.depth * self.height * self.width

    def map_size(self):
        """
        返回二维切片大小
        :return:
        """
        return self.width * self.height

    def save(self, image_save_root = None, saved_file_name = None, label_save_root = None):
        """
        保存图像和标签数据
        :param image_save_root: 神经元图像保存路径
        :param saved_file_name: swc 文件保存路径
        :param label_save_root: 标签图像保存路径
        :return:
        """
        if image_save_root == None:
            pass
        else:
            self.image3d.save(image_save_root = image_save_root, dim = 0)
        if saved_file_name == None:
            pass
        else:
            self.neuronnodelist.save(saved_file_name = saved_file_name)
        if label_save_root == None:
            pass
        elif label_save_root == image_save_root:
            raise ValueError('神经元图像数据保存路径和神经元标签图像保存路径不能相同，请重置')
        else:
            if self.label3d == None:
                self.neuronnodelist.generate_label_3d()
                self.label3d = self.neuronnodelist.label_3d
            self.label3d.save(image_save_root = label_save_root, dim = 0)

    def cut_whole(self, redundancy = (5, 10, 10)):
        """
        将神经元节点列表在其所处的立方体内完整裁剪，并将对应的神经元图像数据进行裁剪
        :param redundancy: 在完整裁剪时候，在三个轴向上保留的冗余，以使得神经元不是刚刚好贴在裁剪后的图像边儿上的
                            这里需要注意，轴顺序是 depth(Z), height(Y), width(X), 形状shape的三个参数顺序与此相同
                                        坐标顺序是 Z(depth), Y(height), X(width)
                            因此这里三元组redundancy的顺序为depth(Z), height(Y), width(X)
        :return:
        """
        coor = [c0 - c1 for (c0, c1) in zip((self.neuronnodelist.init_z0, self.neuronnodelist.init_y0, self.neuronnodelist.init_x0), redundancy)]
        coor = [c if c > 0 else 0 for c in coor]    # 确定平移后的左下角坐标
        # 确定平移裁剪后的神经元节点列表所处的空间大小，神经元节点列表的实际空间大小无法改变
        shape = [s + 2 * c for (s, c) in zip((self.neuronnodelist.real_depth, self.neuronnodelist.real_height, self.neuronnodelist.real_width), redundancy)]
        shape[0] = shape[0] if shape[0] <= self.depth  - coor[0] else self.depth  - coor[0]
        shape[1] = shape[1] if shape[1] <= self.height - coor[1] else self.height - coor[1]
        shape[2] = shape[2] if shape[2] <= self.width  - coor[2] else self.width  - coor[2]
        #print('depth = {}, height = {}, width = {}'.format(self.depth, self.height, self.width))
        #print('shape = {}, coor = {}'.format(shape, coor))
        self.image3d = self.image3d.cut(shape = shape, coor = coor)
        self.neuronnodelist.cut_whole(redundancy = redundancy)

    def cut_as_label(self):
        """
        按照标签 label 提取图像中的目标神经元
        :return:
        """
        if self.label3d == None:
            self.neuronnodelist.generate_label_3d()
            self.label3d = self.neuronnodelist.label_3d
        self.image3d.cut_as_label(self.label3d)

    def resize(self, shape_new = (64, 256, 256)):
        self.image3d.resize(shape_new = shape_new)
        self.neuronnodelist.resize(shape_new = shape_new)


class NeuronImageWithLabel(ImageWithLabel):
    """
    神经元图像数据和标签数据的协同处理类型
    """

    def __init__(self, image_path, file_swc, resolution = (1.,1.,1.), label = False):
        """
        根据给定的图像路径生成 Image3D_PATH 类型，根据给定的 swc 文件生成 NeuronNodeList_SWC 类型
        :param image_path: 保存神经元的二维图像序列的路径
        :param file_swc: swc 文件全名
        """
        assert os.path.isdir(image_path), image_path
        assert os.path.isfile(file_swc), file_swc
        image_3d = Image3D_PATH(image_path = image_path)
        neuron_node_list = NeuronNodeList_SWC(depth = image_3d.depth, height = image_3d.height, width = image_3d.width, file_swc = file_swc, resolution = resolution)
        super(NeuronImageWithLabel, self).__init__(image_3d = image_3d, neuron_node_list = neuron_node_list)
        self.label_3d = None
        if label:
            self.label_3d = self.neuronnodelist.generate_label_3d()

    def cut_whole(self, label=False):
        """
        将神经元节点列表在其所处的立方体内完整裁剪，并将对应的神经元图像数据进行裁剪
        :return:
        """
        super(NeuronImageWithLabel, self).cut_whole()
        if label:
            del self.label_3d
            self.label_3d = self.neuronnodelist.generate_label_3d()


def cut_all():
    """
    将所有图像按神经元实际大小进行切割
    :return:
    """
    root = '/media/liqiufu/Neuron_Data_10T/liqiufu/PycharmProjects/MyDataBase/DataBase_1'
    save_root = '/media/liqiufu/Neuron_Data_10T/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1'
    source = '/media/liqiufu/Neuron_Data_10T/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/info_2'
    neuron_info_list = open(source).readlines()
    neuron_info_list = [e.split() for e in neuron_info_list if not e.startswith('#')]
    for index, neuron_info in enumerate(neuron_info_list):
        if index != 52:
            continue
        neuron_name = neuron_info[0]
        file_swc_name = neuron_info[1]
        print('{} --- processing {}'.format(str(index).ljust(6), neuron_name))
        image_path = os.path.join(root, neuron_name, 'image_tiff')
        file_swc = os.path.join(root, neuron_name, file_swc_name)
        image_save_root = os.path.join(save_root, neuron_name, 'image')
        image_save_root_1 = os.path.join(save_root, neuron_name, 'image_1')
        label_save_root = os.path.join(save_root, neuron_name, 'label')
        saved_file_name = os.path.join(save_root, neuron_name, neuron_name + '.swc')

        neuron_image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        neuron_image_with_label.cut_whole()
        neuron_image_with_label.save(image_save_root = image_save_root, saved_file_name = saved_file_name, label_save_root = label_save_root)
        neuron_image_with_label.cut_as_label()
        neuron_image_with_label.image3d.save(image_save_root = image_save_root_1)

def generate_label():
    """
    将所有图像按神经元实际大小进行切割
    :return:
    """
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized'
    save_root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_2_new'
    #source = '/media/liqiufu/Neuron_Data_10T/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/info_2'
    #neuron_info_list = open(source).readlines()
    #neuron_info_list = [e.split() for e in neuron_info_list if not e.startswith('#')]
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for index, neuron_name in enumerate(neuron_name_list):
        print('{} / {} --- processing {}'.format(str(index).ljust(6), len(neuron_name_list), neuron_name))
        image_path = os.path.join(root, neuron_name, 'image')
        if not os.path.isdir(image_path):
            continue
        file_swc = os.path.join(root, neuron_name, neuron_name + '.swc')
        image_save_root = os.path.join(save_root, neuron_name, 'image')
        image_save_root_1 = os.path.join(save_root, neuron_name, 'image_1')
        label_save_root = os.path.join(save_root, neuron_name, 'label')
        saved_file_name = os.path.join(save_root, neuron_name, neuron_name + '.swc')

        neuron_image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        neuron_image_with_label.save(image_save_root = image_save_root, saved_file_name = saved_file_name, label_save_root = label_save_root)
        neuron_image_with_label.cut_as_label()
        neuron_image_with_label.image3d.save(image_save_root = image_save_root_1)

def add_noise():
    """
        将所有图像按神经元实际大小进行切割
        :return:
    """
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_2_new'
    save_root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_3_new'
    source = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/info_2'
    neuron_info_list = open(source).readlines()
    neuron_info_list = [e.split() for e in neuron_info_list if not e.startswith('#')]
    for index, neuron_info in enumerate(neuron_info_list):
        #if index != 57:
        #    continue
        neuron_name = neuron_info[0]
        print('{} --- processing {}'.format(str(index).ljust(6), neuron_name))
        file_swc_name = neuron_info[1]
        if len(neuron_info) > 2:
            add_noise_mode = neuron_info[2]
        else:
            add_noise_mode = 'continue'

        image_path = os.path.join(root, neuron_name, 'image')
        file_swc = os.path.join(root, neuron_name, neuron_name + '.swc')
        image_save_root = os.path.join(save_root, neuron_name, 'image')
        saved_file_name = os.path.join(save_root, neuron_name, neuron_name + '.swc')

        neuron_image_with_label = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        if add_noise_mode == 'add_noise':
            neuron_image_with_label.image3d.add_noise(var = 0.01)
        neuron_image_with_label.save(image_save_root = image_save_root, saved_file_name = saved_file_name)

        image_root_1 = os.path.join(root, neuron_name, 'image_1')
        image_3d_1 = Image3D_PATH(image_path = image_root_1)
        image_save_root_1 = os.path.join(save_root, neuron_name, 'image_1')
        image_3d_1.save(image_save_root = image_save_root_1)

        label_root = os.path.join(root, neuron_name, 'label')
        label_3d = Image3D_PATH(image_path = label_root)
        label_save_root = os.path.join(save_root, neuron_name, 'label')
        label_3d.save(image_save_root = label_save_root)

def cut_image():
    """
    从图像中裁剪出没有干扰纤维的图像块；干扰纤维未进行标记，无法用于制作训练数据
    :return:
    """
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_3_new'
    save_root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_4_new'
    source = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/resize_info'
    neuron_info_list = open(source).readlines()
    neuron_info_list = [e.strip().split() for e in neuron_info_list if not e.startswith('#')]
    for index, neuron_info in enumerate(neuron_info_list):
        neuron_name = neuron_info[0]
        if neuron_name in NEURON_NOT_IN_TRAIN_TEST:
            continue
        print('{} --- processing {}'.format(str(index).ljust(6), neuron_name))
        depth_o = int(neuron_info[1])
        height_o = int(neuron_info[2])
        width_o = int(neuron_info[3])
        ratio_resize = float(neuron_info[4])
        if len(neuron_info) > 5:
            block_info = neuron_info[5]
        else:
            command = 'cp -r {} {}'.format(os.path.join(root, neuron_name), save_root)
            os.system(command)
            continue
        image_path = os.path.join(root, neuron_name, 'image')
        label_path = os.path.join(root, neuron_name, 'label')
        image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        assert image.shape() == label.shape()
        block_boundary_info_list = get_block_boundary_info(block_info = block_info, shape = image.shape(), shape_o = (depth_o, height_o, width_o), ratio_resize = ratio_resize)
        for index, block_boundary_info in enumerate(block_boundary_info_list):
            image_save_path = os.path.join(save_root, neuron_name + '_' + str(index), 'image')
            label_save_path = os.path.join(save_root, neuron_name + '_' + str(index), 'label')
            z0 = block_boundary_info['z0']
            z1 = block_boundary_info['z1']
            y0 = block_boundary_info['y0']
            y1 = block_boundary_info['y1']
            x0 = block_boundary_info['x0']
            x1 = block_boundary_info['x1']
            image_block = image.image_3d[z0:z1, y0:y1, x0:x1]
            label_block = label.image_3d[z0:z1, y0:y1, x0:x1]
            save_image_3d(image_block, image_save_root = image_save_path)
            save_image_3d(label_block, image_save_root = label_save_path)

def get_block_boundary_info(block_info, shape, shape_o = (64, 256, 256), ratio_resize = 1.0):
    """
    解析图像块边界信息
    :param block_info:
    :param shape:
    :return:
    """
    depth, height, width = shape
    depth_o, height_o, width_o = shape_o
    if ratio_resize == 1.0:
        assert depth == depth_o
        assert height == height_o
        assert width == width_o
    block_info_list = list()
    block_info_list_init = block_info.split(';')
    for block_info_init in block_info_list_init:
        z_info, y_info, x_info = block_info_init.split(',')
        z0, z1 = z_info.split(':')
        y0, y1 = y_info.split(':')
        x0, x1 = x_info.split(':')
        z0 = int(z0)
        z1 = int(z1)
        y0_ = int(y0)
        y1_ = int(y1)
        x0 = int(x0)
        x1 = int(x1)
        y0 = 0 if y1_ == -1 else height_o - y1_
        y1 = height_o if y0_ == 0 else height_o - y0_
        z1 = depth_o if z1 == -1 else z1
        x1 = width_o if x1 == -1 else x1

        if ratio_resize < 1.0:
            x0 = int(x0 * ratio_resize)
            x1 = int(x1 * ratio_resize)
            y0 = int(y0 * ratio_resize)
            y1 = int(y1 * ratio_resize)
        if depth_o >= 32 / ratio_resize:
            print('有点儿复杂 。。。 ')
            z0 = int(z0 * ratio_resize)
            z1 = int(z1 * ratio_resize)

        block_info_list.append({'z0': z0, 'z1': z1, 'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1})
    return block_info_list

def cut_image_1():
    root = '/media/liqiufu/Neuron_Data_10T/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_4'
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for index, neuron_name in enumerate(neuron_name_list):
        print('{} --- {}'.format(str(index).ljust(6), neuron_name))
        neuron_path = os.path.join(root, neuron_name)
        if '_' not in neuron_name:
            continue
        image_1_save_path = os.path.join(neuron_path, 'image_1')
        image_path = os.path.join(neuron_path, 'image')
        label_path = os.path.join(neuron_path, 'label')
        image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        image_1 = image.image_3d
        image_1[label.image_3d == 0] = 0
        save_image_3d(image_1, image_1_save_path)

def resize():
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1'
    root_save = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized'
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    resize_info = open(os.path.join(root_save, 'resize_info'), 'w')
    for index, neuron_name in enumerate(neuron_name_list):
        print('{} / {} -- processing {}'.format(str(index).rjust(3), len(neuron_name_list), neuron_name))
        image_path = os.path.join(root, neuron_name, 'image')
        file_swc = os.path.join(root, neuron_name, neuron_name + '_revised.swc')
        image_path_save = os.path.join(root_save, neuron_name, 'image')
        file_swc_save = os.path.join(root_save, neuron_name, neuron_name + '.swc')

        imagewithlabel = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
        depth, height, width = imagewithlabel.shape()
        length = max(depth, height, width)
        ratio = 1.0
        if length > 1024:
            ratio = 1024 / length
            depth_ = round(depth * ratio) if depth > 32 / ratio else depth
            height_ = round(height * ratio)
            width_ = round(width * ratio)
            shape_new = (depth_, height_, width_)
            imagewithlabel.resize(shape_new = shape_new)
        imagewithlabel.save(image_save_root = image_path_save, saved_file_name = file_swc_save)
        line = '\t'.join([neuron_name, str(depth), str(height), str(width), '{:.3f}'.format(ratio), '\n'])
        resize_info.write(line)

def comput_label_ratio():
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_3'
    neuron_name_list = os.listdir(root)
    neuron_name_list.sort()
    for index, neuron_name in enumerate(neuron_name_list):
        image_path = os.path.join(root, neuron_name, 'image')
        label_path = os.path.join(root, neuron_name, 'label')
        #image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        label.image_3d[label.image_3d != 0] = 1.
        ratio = np.sum(label.image_3d) / (label.depth * label.height * label.width)
        print('{}\t{:.6f}'.format(neuron_name.ljust(16), ratio))

def resave():
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_5/000020_0'
    block_name_list = os.listdir(root)
    block_name_list.sort()
    for block_name in block_name_list:
        block_path = os.path.join(root, block_name)
        image_path = os.path.join(block_path, 'image')
        label_path = os.path.join(block_path, 'label')
        image = Image3D_PATH(image_path = image_path)
        label = Image3D_PATH(image_path = label_path)
        #label = Image3D_PATH(image_path = label_path)
        image.save(image_save_root = image_path)
        label.save(image_save_root = label_path)

if __name__ == '__main__':
    """
    image_path = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1/000020/image'
    file_swc = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1/000020/000020_revised.swc'

    image_path_save = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized/000020/image'
    file_swc_save = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized/000020/000020.swc'

    imagewithlabel = NeuronImageWithLabel(image_path = image_path, file_swc = file_swc)
    depth, height, width = imagewithlabel.shape()
    length = max(depth, height, width)
    if length > 1024:
        ratio = 1024 / length
        depth  = round(depth  * ratio)
        height = round(height * ratio)
        width  = round(width  * ratio)
    shape_new = (depth, height, width)
    imagewithlabel.resize(shape_new = shape_new)
    imagewithlabel.save(image_save_root = image_path_save, saved_file_name = file_swc_save)
    """
    #resize()
    #generate_label()
    #add_noise()
    cut_image()


    #comput_label_ratio()
    #resave()