import argparse, os
import torch
import numpy as np
from datetime import datetime
from tools.printer import Printer
from train_test.neuron_train_test import Train_Test_Process
from constant import WAVENAME_LIST, NeuronNet, DataTrain_Root, TrainSource, TestSource

def my_mkdir(file_name, mode = 'file'):
    """
    创建根路径
    :param mode: 'path', 'file'
    """
    if mode == 'path':
        if not os.path.isdir(file_name):
            os.makedirs(file_name)
            return
    elif mode == 'file':
        root, name = os.path.split(file_name)
        if not os.path.isdir(root):
            os.makedirs(root)
        return
    else:
        assert mode in ['path', 'file']

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net', type = str, default = 'neuron_segnet', choices = NeuronNet)
    parser.add_argument('--p_dropout', type=str, default='0.5, 0.1',
                        help='use which gpu to train, must be a comma-separated list of floats only')
    parser.add_argument('--wavename', type = str, default = 'none', choices = WAVENAME_LIST)
    parser.add_argument('--dataroot', type=str, default=DataTrain_Root, help='the path to the dataset')
    parser.add_argument('--data_train_source', type=str, default=TrainSource, help='the train image name list of the dataset')
    parser.add_argument('--data_test_source', type=str, default=TestSource, help='the test image name list of the dataset')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--data_size', type = str, default = '32,128,128',
                        help = 'the size of the input neuronal cube, must be a comma-separated list of three integers (default = 32,128,128)')
    parser.add_argument('--class_weight', type = str, default = '1.,5.',
                        help = 'the weights for the catergories in the loss function, must be a comma-separated list of three integers (default = 1,10.)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        metavar='N', help='the epochs for warming up the training')
    parser.add_argument('--epoch_to_save', type=int, default=1, metavar='N',
                        help='number of epochs to save the trained parameters')
    parser.add_argument('--iteration_printing', type=int, default=50, metavar='N',
                        help='number of epochs to save the trained parameters')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--channel_width', type = int, default = 4,
                        metavar = 'N', help = 'the channel width of the network model (default: 8)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    parser.add_argument('--with_BN', action='store_true', default=True,
                        help='whether to use BN in the network model (default: True)')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    # optimizer params
    parser.add_argument('--lr', '--learning_ratio', type = float, default = 0.1, metavar = 'LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--lr_step', type=int, default=15,
                        metavar='N', help='epoch step to decay lr')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default = False, help = 'disables CUDA training')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None, help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', '--fine_tuning', action='store_true', default=False, help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')

    args = parser.parse_args()  #在执行这条命令之前，所有命令行参数都给不会生效
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.gpu_map = {}
            for index, gpu_id in enumerate(args.gpu_ids):
                args.gpu_map['cuda:{}'.format(gpu_id)] = 'cuda:{}'.format(index)
            args.gpu = list([i for i in range(len(args.gpu_ids))])
            args.out_gpu = 0
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    args.p_dropout = tuple(float(s) for s in args.p_dropout.split(','))
    args.data_size = tuple(int(s) for s in args.data_size.split(','))
    args.depth = args.data_size[0]
    args.height = args.data_size[1]
    args.width = args.data_size[2]
    args.class_weight = torch.tensor(np.array(tuple(float(s) for s in args.class_weight.split(',')))).float()

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.checkname is None:
        args.checkname = args.net + '_' + args.wavename

    args.info_file = os.path.join('.', 'info', args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.info'))
    args.weight_root = os.path.join('.', 'weight', args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    my_mkdir(args.info_file, mode = 'file')
    my_mkdir(args.weight_root, mode = 'path')
    args.printer = Printer(args.info_file)
    torch.manual_seed(args.seed)
    ttper = Train_Test_Process(args)
    #ttper.args.printer.pprint('Starting Epoch: {}'.format(ttper.args.start_epoch))
    ttper.args.printer.pprint('Total Epoches: {}'.format(ttper.args.epochs))
    args.time_begin = datetime.now()
    ttper.test()
    ttper.train()

    #trainer.writer.close()

if __name__ == "__main__":
   main()