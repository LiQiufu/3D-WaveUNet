"""
这个脚本对预测结果和真实结果进行对比
"""

import numpy as np
from tools.printer import Printer
from copy import deepcopy, copy
import torch


class Evaluator(object):
    def __init__(self, num_class, printer, mode = 'train'):
        self.num_class = num_class
        self.printer = printer
        self.mode = mode
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        self.IoU_class = MIoU
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength = self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        if gt_image.size()[1] == 1:
            gt_image.squeeze_(dim = 1)
        if pre_image.size()[1] == 1:
            pre_image.squeeze_(dim = 1)
        pre_image = pre_image.cpu().numpy()
        gt_image = gt_image.cpu().numpy()
        assert gt_image.shape == pre_image.shape, 'gt_image_shape: {} != pre_image: {}'.format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        del gt_image, pre_image

    def printing(self):
        _print(classes = self.confusion_matrix, printer = self.printer, model = self.mode, maximum = self.num_class - 1)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Evaluator_Torch(object):
    def __init__(self, num_class, printer, mode = 'train'):
        self.num_class = num_class
        self.printer = printer
        self.mode = mode
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        self.IoU_class = MIoU
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask] + pre_image[mask]
        count = torch.bincount(label, minlength = self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix.cpu().numpy()

    def add_batch(self, gt_image, pre_image):
        if gt_image.size()[1] == 1:
            gt_image.squeeze_(dim = 1)
        if pre_image.size()[1] == 1:
            pre_image.squeeze_(dim = 1)
        #pre_image = pre_image.cpu().numpy()
        #gt_image = gt_image.cpu().numpy()
        assert gt_image.shape == pre_image.shape, 'gt_image_shape: {} != pre_image: {}'.format(gt_image.shape, pre_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        del gt_image, pre_image

    def printing(self):
        _print(classes = self.confusion_matrix, printer = self.printer, model = self.mode, maximum = self.num_class - 1)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Compare_Segmentation_Label():
    """
    描述预测值与真值比较的类型，这里真值和预测值都是矩阵，他们用于描述图像分割
    """
    def __init__(self, ground_truth, predict_label, printer, model = 'train'):
        """
        :param ground_truth: 真值矩阵
        :param predict_label: 预测矩阵
        :param printer: 打印输出
        """
        self.ground_truth = ground_truth
        self.predict_label = predict_label
        self.printer = printer
        self.model = model
        self._check()
        self.size = self.ground_truth.size
        self.classes = self._classes()
        self.compare()
        self.grid_size = 36

    def _check(self):
        """
        将真值矩阵和预测矩阵中维度尺寸等于1的那些维度去除，之后比较二者尺寸是否一致
        :return:
        """
        assert isinstance(self.ground_truth, np.ndarray)
        assert isinstance(self.predict_label, np.ndarray)
        shape_old = self.ground_truth.shape
        if 1 in shape_old:
            shape_new = [x for x in shape_old if x != 1]
            self.ground_truth.resize(shape_new)
        shape_old = self.predict_label.shape
        if 1 in shape_old:
            shape_new = [x for x in shape_old if x != 1]
            self.predict_label.resize(shape_new)
        assert self.ground_truth.shape == self.predict_label.shape
        assert isinstance(self.printer, Printer)

    def _classes(self):
        """
        确定真值和预测值矩阵所包含的种类
        :return:
        """
        self.maximum = int(max(np.max(self.ground_truth), np.max(self.predict_label)))
        return dict(zip(range(self.maximum + 1), [None] * (self.maximum + 1)))

    def compare(self):
        """
        比较每一类的准确类
        :return:
        """
        gt = deepcopy(self.ground_truth)
        pl = deepcopy(self.predict_label)
        diff = ((gt == pl) + 0.)
        self.right_size = np.sum(diff)
        right_rate = self.right_size / self.size
        wrong_rate = 1. - right_rate
        self.classes[0] = {'right_rate': '{:6f} = {:8d} / {:8d}'.format(right_rate, int(self.right_size), int(self.size)),
                           'wrong_rate': '{:6f} = {:8d} / {:8d}'.format(wrong_rate, int(self.size - self.right_size), int(self.size))}
        self.accuracy = right_rate

        for i in range(self.maximum):
            gt = deepcopy(self.ground_truth)
            pl = deepcopy(self.predict_label)
            gt[gt != (i + 1)] = 0
            pl[pl != (i + 1)] = 0
            sumimum = (np.sum(gt) + np.sum(pl)) / (i + 1)

            gt_ = deepcopy(gt)
            gt_[gt_ != (i + 1)] = -1
            diff = ((gt_ == pl) + 0.)
            sumimum = sumimum - np.sum(diff)
            number_0 = np.sum(diff)
            right_rate = number_0 / sumimum

            pl_ = deepcopy(pl)
            pl_[gt == (i + 1)] = 0
            number_1 = np.sum(pl_) / (i + 1)
            accepting_false_rate = number_1 / sumimum

            gt_ = deepcopy(gt)
            gt_[pl == (i + 1)] = 0
            number_2 = np.sum(gt_) / (i + 1)
            forsake_truth_rate = number_2 / sumimum
            self.classes[i+1] = {'right': '{:.6f} = {:8d} / {:8d}'.format(right_rate, int(number_0), int(sumimum)),
                                 'accepting_false': '{:.6f} = {:8d} / {:8d}'.format(accepting_false_rate, int(number_1), int(sumimum)),
                                 'forsake_truth': '{:.6f} = {:8d} / {:8d}'.format(forsake_truth_rate, int(number_2), int(sumimum))}

    def printing(self):
        """
        打印结果
        :return:
        """
        text = ('[{}]'.format(self.model) + 'class_NO').ljust(self.grid_size, ' ') \
               + ''.join([str(x).center(self.grid_size, ' ') for x in list(range(self.maximum + 1))])
        self.printer.pprint(text = text)

        text = ('[{}]'.format(self.model) + 'right_rate').ljust(self.grid_size, ' ') \
               + '{}'.format(self.classes[0]['right_rate']).center(self.grid_size, ' ')
        for i in range(self.maximum):
            text += '{}'.format(self.classes[i + 1]['right']).center(self.grid_size, ' ')
        self.printer.pprint(text = text)

        text = ('[{}]'.format(self.model) + 'accepting_false(0-1)').ljust(self.grid_size, ' ') \
               + '{}'.format(self.classes[0]['wrong_rate']).center(self.grid_size, ' ')
        for i in range(self.maximum):
            text += '{}'.format(self.classes[i + 1]['accepting_false']).center(self.grid_size, ' ')
        self.printer.pprint(text = text)

        text = ('[{}]'.format(self.model) + 'forsake_truth(1-0)').ljust(self.grid_size, ' ') \
               + '--'.center(self.grid_size, ' ')
        for i in range(self.maximum):
            text += '{}'.format(self.classes[i + 1]['forsake_truth']).center(self.grid_size, ' ')
        self.printer.pprint(text = text)


class Compare_Segmentation_Label_New():
    """
    描述预测值与真值比较的类型，这里真值和预测值都是矩阵，他们用于描述图像分割
    """
    def __init__(self, ground_truth, predict_label, printer, model = 'train'):
        """
        :param ground_truth: 真值矩阵
        :param predict_label: 预测矩阵
        :param printer: 打印输出
        :param model: 标识符
        """
        self.ground_truth = ground_truth
        self.predict_label = predict_label
        self.printer = printer
        self.model = model
        self._check()
        self.size = self.ground_truth.size
        self.classes = self._classes()
        self.compare()
        self.grid_size = 36

    def _check(self):
        """
        将真值矩阵和预测矩阵中维度尺寸等于1的那些维度去除，之后比较二者尺寸是否一致
        :return:
        """
        assert isinstance(self.ground_truth, np.ndarray)
        assert isinstance(self.predict_label, np.ndarray)
        shape_old = self.ground_truth.shape
        if 1 in shape_old:
            shape_new = [x for x in shape_old if x != 1]
            self.ground_truth.resize(shape_new)
        shape_old = self.predict_label.shape
        if 1 in shape_old:
            shape_new = [x for x in shape_old if x != 1]
            self.predict_label.resize(shape_new)
        assert self.ground_truth.shape == self.predict_label.shape
        assert isinstance(self.printer, Printer)

    def _classes(self):
        """
        确定真值和预测值矩阵所包含的种类
        :return:
        """
        self.maximum = int(max(np.max(self.ground_truth), np.max(self.predict_label)))
        return np.zeros((self.maximum + 1, self.maximum + 1))

    def compare(self):
        """
        比较每一类的准确类
        :return:
        """
        gt = deepcopy(self.ground_truth)
        pl = deepcopy(self.predict_label)
        same = ((gt == pl) + 0.)
        self.right_size = np.sum(same)
        self.accuracy = self.right_size / self.size

        for i in range(self.maximum + 1):
            gt = deepcopy(self.ground_truth)
            pl = deepcopy(self.predict_label)
            gt[gt != i] = -1
            pl[gt != i] = -1
            for j in range(self.maximum + 1):
                pl_ = deepcopy(pl)
                pl_[pl_ != j] = -1
                pl_ = pl_ + 1
                number = np.sum(pl_) / (j + 1)
                self.classes[i,j] = number

    def printing(self):
        """
        打印结果
        :return:
        """
        self.printer.pprint(text = '-' * (4 * self.grid_size))
        text = ('[{}]'.format(self.model) + 'class_no').ljust(self.grid_size) \
               + 'recall_rate'.center(self.grid_size) \
               + 'precision'.center(self.grid_size) \
               + 'IoU'.center(self.grid_size)
        self.printer.pprint(text = text)
        sum_recall = np.sum(self.classes, axis = 1)
        sum_pre = np.sum(self.classes, axis = 0)
        for i in range(self.maximum + 1):
            text = ('[{}]'.format(self.model) + str(i) + '_rpi').ljust(self.grid_size)
            text += ('{:.6f} = {:8d} / {:8d}'.format(self.classes[i][i]/sum_recall[i],
                                                     int(self.classes[i][i]),
                                                     int(sum_recall[i]))).center(self.grid_size)
            text += ('{:.6f} = {:8d} / {:8d}'.format(self.classes[i][i]/sum_pre[i],
                                                     int(self.classes[i][i]),
                                                     int(sum_pre[i]))).center(self.grid_size)
            text += ('{:.6f} = {:8d} / {:8d}'.format(self.classes[i][i]/(sum_recall[i]+sum_pre[i]-self.classes[i][i]),
                                                     int(self.classes[i][i]),
                                                     int(sum_recall[i]+sum_pre[i]-self.classes[i][i]))).center(self.grid_size)
            self.printer.pprint(text = text)
        text = ('[{}]'.format(self.model) + 'total_precision:').ljust(self.grid_size) + \
               ('{:.6f} = {:8d} / {:8d}'.format(self.right_size/self.size,
                                                int(self.right_size),
                                                int(self.size))).center(self.grid_size) + \
               ('[{}]'.format(self.model) + 'total_false:').center(self.grid_size) + \
               ('{:.6f} = {:8d} / {:8d}'.format(1 - (self.right_size / self.size),
                                                int(self.size - self.right_size),
                                                int(self.size))).center(self.grid_size)
        self.printer.pprint(text = text)
        self.printer.pprint(text = '-' * (4 * self.grid_size))

        text = ('[{}]'.format(self.model) + 'ground_truth').ljust(self.grid_size) + \
               ''.join([str(i).center(self.grid_size) for i in range(self.maximum + 1)])
        self.printer.pprint(text = text)
        for i in range(self.maximum + 1):
            text = ('[{}]'.format(self.model) + str(i) + '_predicted_to').ljust(self.grid_size)
            for j in range(self.maximum + 1):
                text += ('{:.6f} = {:8d} / {:8d}'.format(self.classes[i][j]/sum_recall[i],
                                                         int(self.classes[i][j]),
                                                         int(sum_recall[i]))).center(self.grid_size)
            self.printer.pprint(text = text)
        self.printer.pprint(text = '-' * (4 * self.grid_size))

        text = ('[{}]'.format(self.model) + 'predict_label').ljust(self.grid_size) + \
               ''.join([str(i).center(self.grid_size) for i in range(self.maximum + 1)])
        self.printer.pprint(text = text)
        for i in range(self.maximum + 1):
            text = ('[{}]'.format(self.model) + str(i) + '_source_from').ljust(self.grid_size)
            for j in range(self.maximum + 1):
                text += ('{:.6f} = {:8d} / {:8d}'.format(self.classes[j][i]/sum_pre[i],
                                                         int(self.classes[j][i]),
                                                         int(sum_pre[i]))).center(self.grid_size)
            self.printer.pprint(text = text)
        self.printer.pprint(text = '-' * (4 * self.grid_size))


class Compare_Segmentation_Label_CUDA_New():
    """
    描述预测值与真值比较的类型，这里真值和预测值都是矩阵，他们用于描述图像分割
    """
    classes = None
    def __init__(self, ground_truth, predict_label, printer, model = 'train'):
        """
        :param ground_truth: 真值矩阵
        :param predict_label: 预测矩阵
        :param printer: 打印输出
        :param model: 标识符
        """
        self.ground_truth = ground_truth
        if isinstance(predict_label, torch.cuda.LongTensor):
            self.predict_label = predict_label
        else:
            self.predict_label = torch.Tensor(predict_label).long().cuda()
        self.printer = printer
        self.model = model
        self._check()
        self.size = float(self.ground_truth.numel())
        self.classes = self._classes()
        self.compare()
        self.grid_size = 36

    def _check(self):
        """
        将真值矩阵和预测矩阵中维度尺寸等于1的那些维度去除，之后比较二者尺寸是否一致
        :return:
        """
        self.ground_truth = self.ground_truth.squeeze()
        self.predict_label = self.predict_label.squeeze()
        assert self.ground_truth.shape == self.predict_label.shape
        assert isinstance(self.printer, Printer)

    def _classes(self):
        """
        确定真值和预测值矩阵所包含的种类
        :return:
        """
        self.maximum = int(max(self.ground_truth.max(), self.predict_label.max()))
        return np.zeros((self.maximum + 1, self.maximum + 1))

    def compare(self):
        """
        比较每一类的准确率
        :return:
        """
        same = self.ground_truth.eq(self.predict_label)
        self.right_size = float(same.sum())
        self.accuracy = self.right_size / self.size

        for i in range(self.maximum + 1):
            gt = deepcopy(self.ground_truth)
            pl = deepcopy(self.predict_label)
            gt[gt != i] = -1
            pl[gt != i] = -1
            for j in range(self.maximum + 1):
                pl_ = deepcopy(pl)
                pl_[pl_ != j] = -1
                pl_ = pl_ + 1
                number = pl_.sum() / (j + 1)
                self.classes[i,j] = number
        if self.model == 'train':
            return
        if Compare_Segmentation_Label_CUDA_New.classes is None:
            Compare_Segmentation_Label_CUDA_New.classes = self.classes
        else:
            Compare_Segmentation_Label_CUDA_New.classes += self.classes

    def printing(self):
        _print(self.classes, self.printer, model = self.model, grid_size = self.grid_size, maximum = self.maximum)

    def printing_(self):
        _print(Compare_Segmentation_Label_CUDA_New.classes, self.printer, model = self.model, grid_size = self.grid_size, maximum = self.maximum)
        Compare_Segmentation_Label_CUDA_New.classes = None




def _print(classes, printer, model = 'train', grid_size = 36, maximum = 2):
    """
    打印结果
    :return:
    """
    right_size = sum(classes[i,i] for i in range(maximum + 1))
    size = np.sum(classes)
    printer.pprint(text = '-' * (4 * grid_size))
    text = ('[{}]'.format(model) + 'class_no').ljust(grid_size) \
           + 'recall_rate'.center(grid_size) \
           + 'precision'.center(grid_size) \
           + 'IoU'.center(grid_size)
    printer.pprint(text = text)
    sum_recall = np.sum(classes, axis = 1)
    sum_pre = np.sum(classes, axis = 0)
    for i in range(maximum + 1):
        text = ('[{}]'.format(model) + str(i) + '_rpi').ljust(grid_size)
        text += ('{:.6f} = {:8d} / {:8d}'.format(classes[i][i]/sum_recall[i],
                                                 int(classes[i][i]),
                                                 int(sum_recall[i]))).center(grid_size)
        text += ('{:.6f} = {:8d} / {:8d}'.format(classes[i][i]/sum_pre[i],
                                                 int(classes[i][i]),
                                                 int(sum_pre[i]))).center(grid_size)
        text += ('{:.6f} = {:8d} / {:8d}'.format(classes[i][i]/(sum_recall[i]+sum_pre[i]-classes[i][i]),
                                                 int(classes[i][i]),
                                                 int(sum_recall[i]+sum_pre[i]-classes[i][i]))).center(grid_size)
        printer.pprint(text = text)
    text = ('[{}]'.format(model) + 'total_precision:').ljust(grid_size) + \
           ('{:.6f} = {:8d} / {:8d}'.format(right_size/size,
                                            int(right_size),
                                            int(size))).center(grid_size) + \
           ('[{}]'.format(model) + 'total_false:').center(grid_size) + \
           ('{:.6f} = {:8d} / {:8d}'.format(1 - (right_size / size),
                                            int(size - right_size),
                                            int(size))).center(grid_size)
    printer.pprint(text = text)
    printer.pprint(text = '-' * (4 * grid_size))

    text = ('[{}]'.format(model) + 'ground_truth').ljust(grid_size) + \
           ''.join([str(i).center(grid_size) for i in range(maximum + 1)])
    printer.pprint(text = text)
    for i in range(maximum + 1):
        text = ('[{}]'.format(model) + str(i) + '_predicted_to').ljust(grid_size)
        for j in range(maximum + 1):
            text += ('{:.6f} = {:8d} / {:8d}'.format(classes[i][j]/sum_recall[i],
                                                     int(classes[i][j]),
                                                     int(sum_recall[i]))).center(grid_size)
        printer.pprint(text = text)
    printer.pprint(text = '-' * (4 * grid_size))

    text = ('[{}]'.format(model) + 'predict_label').ljust(grid_size) + \
           ''.join([str(i).center(grid_size) for i in range(maximum + 1)])
    printer.pprint(text = text)
    for i in range(maximum + 1):
        text = ('[{}]'.format(model) + str(i) + '_source_from').ljust(grid_size)
        for j in range(maximum + 1):
            text += ('{:.6f} = {:8d} / {:8d}'.format(classes[j][i]/sum_pre[i],
                                                     int(classes[j][i]),
                                                     int(sum_pre[i]))).center(grid_size)
        printer.pprint(text = text)
    printer.pprint(text = '-' * (4 * grid_size))
