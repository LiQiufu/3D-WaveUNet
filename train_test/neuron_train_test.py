"""
这个脚本描述使用神经元数据集合训练网络参数的过程，并提供接口
"""

import os, math
from torch import optim
from torch.utils.data.dataloader import DataLoader
from train_test.neuron_label_compare import Evaluator_Torch as Evaluator
from train_test.neuron_data import Neuron_Data_Set
from train_test.neuron_loss import CrossEntropy_Loss
from networks.neuron_net import *
from constant import NeuronNet

class Train_Test_Process():
    def __init__(self, args):
        self.args = args
        self.printer = self.args.printer
        self._data()
        self.net = self._get_net()
        self.net.train()
        self.loss = CrossEntropy_Loss(w_crossentropy = 1, class_weight = self.args.class_weight.cuda())
        self.optimizer = self._optimizer()

        if torch.cuda.is_available():
            self.net = DataParallel(module = self.net.cuda(), device_ids = self.args.gpu, output_device = self.args.out_gpu)
            self.loss = self.loss.cuda()
        if self.args.resume != None:
            self.load_pretrained_model()
        self.best_pred = 0.0

        for key, value in self.args.__dict__.items():
            if not key.startswith('_'):
                self.printer.pprint('{} ==> {}'.format(key.rjust(24), value))

    def _get_net(self):
        """
        调用网络模型
        :return:
        """
        if self.args.net == 'neuron_segnet':
            return Neuron_SegNet(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width)
        elif self.args.net == 'neuron_unet_v1':
            return Neuron_UNet_V1(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width)
        elif self.args.net == 'neuron_unet_v2':
            return Neuron_UNet_V2(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width)
        elif self.args.net == 'neuron_wavesnet_v1':
            return Neuron_WaveSNet_V1(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width, wavename = self.args.wavename)
        elif self.args.net == 'neuron_wavesnet_v2':
            return Neuron_WaveSNet_V2(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width, wavename = self.args.wavename)
        elif self.args.net == 'neuron_wavesnet_v3':
            return Neuron_WaveSNet_V3(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width, wavename = self.args.wavename)
        elif self.args.net == 'neuron_wavesnet_v4':
            return Neuron_WaveSNet_V4(num_class = self.num_class, with_BN = self.args.with_BN, channel_width = self.args.channel_width, wavename = self.args.wavename)
        else:
            raise ValueError('args.net is {}, which should in {}'.format(self.args.net, NeuronNet))

    def _optimizer(self):
        weight_p, bias_p = [], []
        for name, p in self.net.named_parameters():
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        return optim.SGD([{'params': weight_p, 'weight_decay': self.args.weight_decay},
                          {'params': bias_p, 'weight_decay': 0}],
                         lr = self.args.lr,
                         momentum = self.args.momentum)

    def _data(self):
        """
        设置训练数据以及测试数据
        :return:
        """
        self.train_root = self.args.dataroot
        self.test_root = self.args.dataroot
        self.neuron_train_set = Neuron_Data_Set(root = self.train_root, source = self.args.data_train_source,
                                                depth = self.args.depth,
                                                height = self.args.height,
                                                width = self.args.width)
        self.iters_per_epoch = len(self.neuron_train_set)
        self.train_data_loader = DataLoader(dataset = self.neuron_train_set, batch_size = self.args.batch_size,
                                            shuffle = True, num_workers = self.args.workers)
        self.neuron_test_set_0 = Neuron_Data_Set(root = self.test_root, source = self.args.data_test_source,
                                                depth = self.args.depth,
                                                height = self.args.height,
                                                width = self.args.width)
        self.test_data_loader_0 = DataLoader(dataset = self.neuron_test_set_0, batch_size = self.args.test_batch_size,
                                           shuffle = False, num_workers = self.args.workers)
        self.num_class = self.neuron_train_set.num_class
        self.evaluator_train = Evaluator(num_class = self.num_class, printer = self.args.printer, mode = 'train')
        self.evaluator_test = Evaluator(num_class = self.num_class, printer = self.args.printer, mode = 'test')

    def load_pretrained_model(self):
        """
        加载预训练的模型参数，有可能某些模型参数被修改或增加减少某些层，这些情况应被处理
        :return:
        """
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(self.args.resume, map_location = self.args.gpu_map)
        keys_abandon = [k for k in pretrained_dict if k not in model_dict]
        keys_without_load = [k for k in model_dict if k not in pretrained_dict]
        pretrained_dict = [(k, v) for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape]
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)
        for key in pretrained_dict:
            text = 'have loaded "{}" layer'.format(key[0])
            self.printer.pprint(text)
        for key in keys_abandon:
            text = 'A --- "{}" layer in pretrained model did not been loaded'.format(key)
            self.printer.pprint(text)
        for key in keys_without_load:
            text = 'B --- "{}" layer in current model did not been initilizated with pretrain model'.format(key)
            self.printer.pprint(text)

    def adjust_learning_rate(self, epoch, iteration):
        """
        调整学习率
        :param epoch: 当前 epoch 序数
        :param iteration: 当前 epoch 的 iteration 序数
        :return:
        """
        T = epoch * self.iters_per_epoch + iteration
        N = self.args.epochs * self.iters_per_epoch
        warmup_iters = self.args.warmup_epochs * self.iters_per_epoch

        if self.args.lr_scheduler == 'step':
            lr = self.args.lr * (0.1 ** (epoch // self.args.lr_step))
        elif self.args.lr_scheduler == 'poly':
            lr = self.args.lr * pow((1 - 1.0 * T / N), 0.9)
        elif self.args.lr_scheduler == 'cos':
            lr = 0.5 * self.args.lr * (1 + math.cos(1.0 * T / N * math.pi))
        else:
            raise ValueError('lr_scheduler is {}, which should be in ["step", "poly", "cos"]'.format(self.args.lr_scheduler))

        if warmup_iters > 0 and T < warmup_iters:
            lr = lr * 1.0 * T / warmup_iters

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def test(self):
        """
        对测试数据进行测试
        :return:
        """
        with torch.no_grad():
            for test_loader in [self.test_data_loader_0,]:
                self.evaluator_test.reset()
                start = datetime.now()
                loss_total = 0.0
                self.net.eval()
                length = len(test_loader)
                for index, sample in enumerate(test_loader):
                    start_1 = datetime.now()
                    test_image = sample['image']
                    test_label = sample['label']
                    test_image = test_image.cuda()
                    test_label = test_label.cuda()
                    test_image.unsqueeze_(dim = 1)
                    #test_image = (test_image - 127.5) / 255
                    output0 = self.net.forward(test_image)
                    test_loss, str_loss = self.loss.forward(inputs = output0, targets = test_label)
                    loss_total += test_loss.data
                    _, predict_label = output0.topk(1, dim = 1)
                    self.evaluator_test.add_batch(gt_image = test_label, pre_image = predict_label)
                    stop_1 = datetime.now()
                    acc = self.evaluator_test.Pixel_Accuracy()
                    mIoU = self.evaluator_test.Mean_Intersection_over_Union()
                    text = '{}/{}, test_loss = {:.6f} / {:.6f}, acc = {:.6f}, mIoU = {:.6f}, took {} hours'.\
                        format(index, length, test_loss, loss_total / (index+1), acc, mIoU, stop_1 - start_1)
                    self.printer.pprint('testing - ' + text)

                self.printer.pprint('testing totally ---- ')
                self.evaluator_test.printing()
                mIoU = self.evaluator_test.Mean_Intersection_over_Union()
                if mIoU > self.best_pred:
                    self.best_pred = mIoU
                    filename = os.path.join(self.args.weight_root, self.args.checkname + '_best.pth.tar')
                    torch.save({'state_dict': self.net.state_dict(), 'best_pred': self.best_pred, }, filename)

                stop = datetime.now()
                text = 'testing took {} hours'.format(stop - start)
                self.printer.pprint(text = text)
                self.printer.pprint(' ')
        self.net.train()

    def train(self):
        """
        对数据进行训练
        :return:
        """
        start = datetime.now()
        number = 0
        train_length = len(self.train_data_loader)
        for epoch in range(self.args.epochs):
            start_0 = datetime.now()
            self.epoch = epoch
            train_loss_epoch = 0.0
            self.evaluator_train.reset()
            for iteration_in_epoch, sample in enumerate(self.train_data_loader):
                start_1 = datetime.now()
                #text = 'Iter {:6d}, Epoch {:3d}/{:3d}, batch {:4d} / {}, lr = {:.6f}'.format(number, epoch, self.args.epochs,
                #                                                                             iteration_in_epoch, train_length,
                #                                                                             self.optimizer.param_groups[0]['lr'])
                #self.printer.pprint('training - ' + text)
                train_image = sample['image']
                train_label = sample['label']
                train_image.requires_grad_()
                train_image = train_image.cuda()
                train_label = train_label.cuda()
                train_image.unsqueeze_(dim = 1)
                #train_image = (train_image - 127.5) / 255

                output0 = self.net.forward(train_image)
                train_loss, str_loss = self.loss.forward(inputs = output0, targets = train_label)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                _, predict_label = output0.topk(1, dim = 1)
                train_loss_batch = train_loss.data
                train_loss_epoch += train_loss_batch
                #"""
                self.evaluator_train.add_batch(gt_image = train_label, pre_image = predict_label)
                if number % self.args.iteration_printing == 0:
                    self.evaluator_train.printing()
                acc = self.evaluator_train.Pixel_Accuracy()
                mIoU = self.evaluator_train.Mean_Intersection_over_Union()
                #"""
                stop_1 = datetime.now()
                #text = 'Epoch {:3d}/{:3d}, batch {:4d} / {}, train_loss = {:.6f} / {:.6f}, took {} / {}, lr = {:.6f}'. \
                #    format(epoch, self.args.epochs, iteration_in_epoch, train_length, train_loss_batch,
                #           train_loss_epoch / (iteration_in_epoch + 1), stop_1 - start_1, stop_1 - start,
                #           self.optimizer.param_groups[0]['lr'])
                text = 'Epoch {:3d}/{:3d}, batch {:4d} / {}, train_loss = {:.6f} / {:.6f}, acc = {:.6f}, mIoU = {:.6f}, took {} / {}, lr = {:.6f}'.\
                    format(epoch, self.args.epochs, iteration_in_epoch, train_length, train_loss_batch, train_loss_epoch / (iteration_in_epoch+1), acc, mIoU, stop_1 - start_1, stop_1 - start, self.optimizer.param_groups[0]['lr'])
                self.printer.pprint('training - ' + text)
                number += 1
                self.adjust_learning_rate(epoch = epoch, iteration = iteration_in_epoch)

            self.test()
            if (epoch + 1) % self.args.epoch_to_save == 0 or (epoch + 1) == self.args.epochs:
                text = 'saving weights ...'
                self.printer.pprint(text)
                filename = os.path.join(self.args.weight_root, 'epoch_{}'.format(epoch) + '.pth.tar')
                torch.save({'epoch': epoch + 1, 'state_dict': self.net.state_dict(),
                            'optimizer': self.optimizer.state_dict(), 'best_pred': self.best_pred, }, filename)

            stop_0 = datetime.now()
            text = 'Epoch - {:3d}, train_loss = {:.6f}, took {} hours -- {}'.format(epoch, train_loss_epoch / self.iters_per_epoch,
                                                                              stop_0 - start_0, stop_0 - start)
            self.printer.pprint(text)
            self.printer.pprint(' ')
        stop = datetime.now()
        text = 'train_test_process finish, took {} hours !!'.format(stop - start)
        self.printer.pprint(text)


if __name__ == '__main__':
    train_test_process = Train_Test_Process()
    train_test_process.train()

