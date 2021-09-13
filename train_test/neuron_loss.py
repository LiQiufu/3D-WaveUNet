"""
这个脚本定义训练神经元图像分割时候，所用的损失函数
由于神经元图像中神经元纤维占比非常少，即背景像素数量和属于神经纤维的像素数量非常不均衡，
因此，这里定义的损失函数将常用于分类的交叉熵与一般的欧氏距离进行加权求和
"""

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from train_test.neuron_args import ARGS

class CrossEntropy_MSE_Loss(Module):
    def __init__(self, w_crossentropy, w_mse):
        """
        :param w_crossentropy: 交叉熵损失的权重
        :param w_mse: 欧氏距离损失的权重
        """
        super(CrossEntropy_MSE_Loss, self).__init__()
        self.w_crossentropy = w_crossentropy
        self.w_mse = w_mse

    def forward(self, input_0, input_1, truth_label):
        """
        :param truth_label: 真实标签
        :param input: 预测标签
        :return:
        """
        loss_crossentropy = torch.nn.CrossEntropyLoss(size_average = True)
        loss_mse = torch.nn.MSELoss(size_average = True)
        self.loss_crossentropy_value = loss_crossentropy.forward(input = input_0, target = truth_label)
        self.loss_mse_value = loss_mse.forward(input = input_1, target = truth_label.float().view(input_1.shape))
        self.loss = self.w_crossentropy * self.loss_crossentropy_value + self.w_mse * self.loss_mse_value
        # print(self.__str__())
        return self.loss, self.__str__()

    def __str__(self):
        return '{} * {:.6f} + {} * {:.6f} = {:.6f}'.format(self.w_crossentropy,
                                               float(self.loss_crossentropy_value.data.cpu().numpy()),
                                               self.w_mse, float(self.loss_mse_value.data.cpu().numpy()),
                                               float(self.loss.data.cpu().numpy()))


class CrossEntropy_Loss(Module):
    def __init__(self, w_crossentropy, class_weight):
        """
        :param w_crossentropy: 交叉熵损失的权重
        :param w_mse: 欧氏距离损失的权重
        """
        super(CrossEntropy_Loss, self).__init__()
        self.w_crossentropy = w_crossentropy
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        """
        :param truth_label: 真实标签
        :param input: 预测标签
        :return:
        """
        loss_crossentropy = torch.nn.CrossEntropyLoss(size_average = True, weight = self.class_weight)
        self.loss_crossentropy_value = loss_crossentropy.forward(input = inputs, target = targets)
        self.loss = self.w_crossentropy * self.loss_crossentropy_value
        return self.loss, self.__str__()

    def __str__(self):
        return '{} * {:.6f} = {:.6f}'.format(self.w_crossentropy,
                                             float(self.loss_crossentropy_value.data.cpu().numpy()),
                                             float(self.loss.data.cpu().numpy()))



class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        C = inputs.size(1)
        P = F.softmax(inputs).view(targets.numel(), C)

        class_mask = inputs.data.new(targets.numel(), C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = - alpha * (torch.pow((1-probs), self.gamma)) * log_p
        if self.size_average:
            self.loss = batch_loss.mean()
        else:
            self.loss = batch_loss.sum()
        return self.loss, self.__str__()

    def __str__(self):
        return '{}'.format(float(self.loss.data.cpu().numpy()))


def one_hot(index, classes, alpha = None):
    size = index.size()[0:1] + (classes,) + index.size()[1:]
    view = index.size()[0:1] + (1,) + index.size()[1:]

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view).data.cpu()

    if isinstance(index, Variable):
        mask = Variable(mask, volatile = index.volatile)

    y_one_hot = mask.scatter_(1, index, 1)
    return y_one_hot


class FocalLoss_(nn.Module):

    def __init__(self, alpha, gamma = ARGS['FL_gamma'], eps = 1e-7):
        super(FocalLoss_, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha.view(-1,1)

    def forward(self, inputs, targets):
        y = one_hot(targets, inputs.size()[1], self.alpha).cuda()
        logit = F.softmax(inputs, dim = -1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        loss.transpose_(1, -1)
        loss = torch.matmul(loss, self.alpha)
        self.loss = loss.mean()

        return self.loss, self.__str__()

    def __str__(self):
        return '{}'.format(float(self.loss.data.cpu().numpy()))