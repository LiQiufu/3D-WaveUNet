"""
对神经元图像进行分割处理
"""

from postprocess.segmentation_neuron_image import denoise_neurons_for_test, denoise_neurons_for_HUST
import argparse
from constant import NeuronNet, WAVENAME_LIST

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net_name', type = str, default = 'neuron_segnet', choices = NeuronNet)
    parser.add_argument('--wavename', type = str, default = 'none')
    parser.add_argument('--model_epoch', type=str, default='29,', metavar='N', help='the epoch number to choice the pretrain model file')
    parser.add_argument('--data_type', type = str, default = 'BigNeuron', help = 'data type, BigNeuron or HUST')
    parser.add_argument('--batchsize', type=int, default=8, metavar='N', help='batchsize')
    parser.add_argument('--gpu', type=int, default=0, metavar='N', help='gpu_id')
    args = parser.parse_args()
    model_epochs = args.model_epoch.split(',')
    model_epochs = [int(e) for e in model_epochs if e]
    wavenames = args.wavename.split(',')
    for model_epoch in model_epochs:
        for wavename in wavenames:
            if args.data_type == 'BigNeuron':
                denoise_neurons_for_test(net_name = args.net_name, wavename = wavename,
                                         model_epoch = model_epoch, batchsize = args.batchsize, gpus = (args.gpu,))
            elif args.data_type == 'HUST':
                denoise_neurons_for_HUST(net_name = args.net_name, wavename = wavename,
                                         model_epoch = model_epoch, batchsize = args.batchsize, gpus = (args.gpu,))

if __name__ == '__main__':
    main()
