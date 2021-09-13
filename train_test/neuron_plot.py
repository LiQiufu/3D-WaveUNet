"""
这个脚本对日志文件进行解析并绘图
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze(file_name, mode = 'train'):
    lines = (line.strip() for line in open(file_name).readlines())
    lines = (line for line in lines if len(line) > 26)
    lines = (line.split() for line in lines)
    mode = mode.lower()
    if mode == 'train_loss':
        loss = []
        for line in lines:
            if len(line) > 3 and line[3] == 'train_loss':
                loss.append(float(line[5][:-1]))
        plt.figure()
        plt.plot(range(len(loss)), loss, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.show()
    elif mode == 'test_loss':
        loss = []
        for line in lines:
            if len(line) > 3 and line[3] == 'test_loss':
                loss.append(float(line[5][:-1]))
        plt.figure()
        plt.plot(range(len(loss)), loss, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.show()
    else:
        rpi_0 = []
        rpi_1 = []
        rpi_2 = []
        lr = []
        for line in lines:
            if line[1] == '[' + mode + ']0_rpi':
                rpi_0.append(float(line[2]))
            if line[1] == '[' + mode + ']1_rpi':
                rpi_1.append(float(line[2]))
            if line[1] == '[' + mode + ']2_rpi':
                rpi_2.append(float(line[2]))
            if line[1] == 'training' and line[3] == 'Iter':
                lr.append(float(line[-1]))
        plt.figure()
        plt.subplot(221)
        plt.plot(range(len(rpi_0)), rpi_0, marker = '+', color = 'r', label = '3')
        plt.grid()
        plt.subplot(223)
        plt.plot(range(len(rpi_1)), rpi_1, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.subplot(222)
        plt.plot(range(len(rpi_2)), rpi_2, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.subplot(224)
        plt.plot(range(len(lr)), lr, marker = '+', color = 'c', label = '3')
        plt.grid()
        plt.show()

if __name__ == '__main__':
    root = '/home/li-qiufu/PycharmProjects/NeuronImageProcess/neuron_pytorch_dl/info/neuron_segmentation_U_77'
    file_name = os.path.join(root, 'neuron_segmentation_U_77_2019-03-31_11-57-46.info')
    analyze(file_name, mode = 'test')