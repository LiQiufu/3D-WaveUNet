"""
对分割后的神经元图像进行自动追踪重建、对比、提取结果等操作
"""

from postprocess.batched_process_neurons import BatchedProcessingSegmentedNeuron
from constant import RESULT_SAVE_PATH, WAVENAME_LIST, DataNeuron_Root, FUNC, RESULT_SAVE_PATH_HUST
import os


def process_noisy_neuronal_images(root, tracker = 'APP2'):
    result_name = '_'.join(['result', tracker])

    batched_processing_segmented_neuron = BatchedProcessingSegmentedNeuron(root = root, tracker = tracker,
                                                                           path_leaf = 'image')
    # 将二维图像序列转化为 v3draw 格式
    batched_processing_segmented_neuron.encoding_images_to_v3draw(flip = '10')
    # 使用 vaa3d 软件自动生成神经元图像数字化重建
    batched_processing_segmented_neuron.generating_neuron_reconstruction()
    # 对比自动数字化重建和手动生成的标准重建结果
    batched_processing_segmented_neuron.comparing_two_reconstructions()
    # 解析对比结果
    batched_processing_segmented_neuron.batched_parse(result_name = result_name)
    # 删除生成的 .v3draw 数据，太占地方了
    batched_processing_segmented_neuron.remove(v3draw = False)


def process_segmented_neuronal_images(trackers = ('APP2',), net_name = 'neuron_wavesnet_v4',
                                      model_epochs = (29,),
                                      wavename_list = WAVENAME_LIST):
    for wavename in wavename_list:
        if wavename == 'none':
            continue
        subpath = net_name if wavename == 'none' or wavename == None else net_name + '_' + wavename
        for model_epoch in model_epochs:
            for index, tracker in enumerate(trackers):
                root = os.path.join(RESULT_SAVE_PATH_HUST, subpath, 'epoch_{}'.format(model_epoch))
                result_name = '_'.join([net_name, 'epoch', str(model_epoch), tracker]) if wavename == None or wavename == 'none' \
                    else '_'.join([net_name, wavename, 'epoch', str(model_epoch), tracker])

                batched_processing_segmented_neuron = BatchedProcessingSegmentedNeuron(root = root, tracker = tracker, path_leaf = 'image_pre_init')
                # 将二维图像序列转化为 v3draw 格式
                batched_processing_segmented_neuron.encoding_images_to_v3draw()
                # 使用 vaa3d 软件自动生成神经元图像数字化重建
                batched_processing_segmented_neuron.generating_neuron_reconstruction()
                # 对比自动数字化重建和手动生成的标准重建结果
                batched_processing_segmented_neuron.comparing_two_reconstructions()
                # 解析对比结果
                batched_processing_segmented_neuron.batched_parse(result_name = result_name)
                if index == len(trackers) - 1:
                    # 删除生成的 .v3draw 数据，太占地方了
                    batched_processing_segmented_neuron.remove(v3draw = True)


def process_segmented_neuronal_images_nowave(trackers = FUNC, net_name = 'neuron_segnet',
                                      model_epochs = (29,)):
    subpath = net_name
    for model_epoch in model_epochs:
        for index, tracker in enumerate(trackers):
            #root = os.path.join(RESULT_SAVE_PATH, subpath, 'epoch_{}'.format(model_epoch))
            root = os.path.join(RESULT_SAVE_PATH_HUST, subpath, 'epoch_{}'.format(model_epoch))
            result_name = '_'.join([net_name, 'epoch', str(model_epoch), tracker])

            batched_processing_segmented_neuron = BatchedProcessingSegmentedNeuron(root = root, tracker = tracker, path_leaf = 'image_pre_init')
            # 将二维图像序列转化为 v3draw 格式
            batched_processing_segmented_neuron.encoding_images_to_v3draw()
            # 使用 vaa3d 软件自动生成神经元图像数字化重建
            batched_processing_segmented_neuron.generating_neuron_reconstruction()
            # 对比自动数字化重建和手动生成的标准重建结果
            batched_processing_segmented_neuron.comparing_two_reconstructions()
            # 解析对比结果
            batched_processing_segmented_neuron.batched_parse(result_name = result_name)
            if index == len(trackers) - 1:
                # 删除生成的 .v3draw 数据，太占地方了
                batched_processing_segmented_neuron.remove(v3draw = True)


if __name__ == '__main__':
    root = '/home/liqiufu/PycharmProjects/BLSTM/result'
    process_noisy_neuronal_images(root)