"""
这个脚本生成训练好的网络模型
"""

from networks.neuron_net import *
from constant import NUM_CLASSES, NeuronNet


class Network_Pretrained():
    def __init__(self, net_name, model_path, with_BN = True, channel_width = 4, wavename = 'haar', gpus = (0,)):
        self.num_class = NUM_CLASSES
        self.net_name = net_name
        self.net = self._get_net(net_name = net_name, with_BN = with_BN, channel_width = channel_width, wavename = wavename)
        self._load_pretrained_model(model_path = model_path, gpus = gpus)

    def _get_net(self, net_name, with_BN = True, channel_width = 4, wavename = 'haar'):
        if net_name == 'neuron_segnet':
            return Neuron_SegNet(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width)
        elif net_name == 'neuron_unet_v1':
            return Neuron_UNet_V1(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width)
        elif net_name == 'neuron_unet_v2':
            return Neuron_UNet_V2(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width)
        elif net_name == 'neuron_wavesnet_v1':
            return Neuron_WaveSNet_V1(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width, wavename = wavename)
        elif net_name == 'neuron_wavesnet_v2':
            return Neuron_WaveSNet_V2(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width, wavename = wavename)
        elif net_name == 'neuron_wavesnet_v3':
            return Neuron_WaveSNet_V3(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width, wavename = wavename)
        elif net_name == 'neuron_wavesnet_v4':
            return Neuron_WaveSNet_V4(num_class = self.num_class, with_BN = with_BN, channel_width = channel_width, wavename = wavename)
        else:
            raise ValueError('args.net is {}, which should in {}'.format(net_name, NeuronNet))

    def _load_pretrained_model(self, model_path, gpus):
        gpu_map = {}
        for index, gpu_id in enumerate(gpus):
            gpu_map['cuda:{}'.format(gpu_id)] = 'cuda:{}'.format(index)
        gpus = list([i for i in range(len(gpus))])
        out_gpu = 0
        self.net = DataParallel(module = self.net.cuda(), device_ids = gpus, output_device = out_gpu)
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(model_path, map_location = gpu_map)
        pretrained_dict = [(k, v) for k, v in pretrained_dict['state_dict'].items()]
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)
        self.net.eval()

    def __call__(self, input):
        return self.net(input)