# 有些文件路径需要在使用时做相应修改！
from constant import ROOT_STANDARD, NEURON_NAME_TEST, NEURON_NAME_NOT_COMPARE, FUNC
from tools.v3draw_coder import encode_to_v3draw, encode_to_v3draw_slice
import os
from collections import OrderedDict


def get_reconstruction_command(input_file_name, output_file_name = None, tracker = 'APP2', bkg_thresh = 1, length_thresh = 1):
    """
    除了 APP2 之外，其他追踪算法无法修改其输出文件名，这里只是为了保持一致
    :param input_file_name:
    :param output_file_name:
    :param tracker:
    :return:
    """
    assert tracker in FUNC.keys()
    if tracker == 'APP2':
        #bkg_thresh = 1
        #length_thresh = 1
        para = 'NULL 0 {} 1 1 0 0 {}'.format(bkg_thresh, length_thresh)     #使用 APP2 追踪分割后的神经元图像 bkg_thresh = 1， length_thresh = 1是比较好的参数设置
        command = '{} -i {} -o {} -p {}'.format(FUNC[tracker], input_file_name, output_file_name, para)
        return command
    elif tracker == 'NeuroGPSTree':
        x_resolution = 0.5
        y_resolution = 0.5
        z_resolution = 1
        binaryThreshold = 15
        traceValue = 10
        linearEnhanceValue = 150
        para = '{} {} {} {} {} {}'.format(x_resolution, y_resolution, z_resolution, binaryThreshold, traceValue, linearEnhanceValue)
        command = '{} -i {} -p {}'.format(FUNC[tracker], input_file_name, para)
        return command
    elif tracker in ['MST_Tracing', 'XY_3D_TreMap', 'snake']:
        command = '{} -i {}'.format(FUNC[tracker], input_file_name)
        return command


class SegmentedNeuronPostprocess():
    """
    这个类型处理分割生成的神经元去噪后图像
    """
    def __init__(self, root, neuron_name, tracker = 'APP2', path_leaf = 'image_pre_init'):
        self.root = root
        self.neuron_name = neuron_name
        self.tracker = tracker
        self.path_leaf = path_leaf
        self.image_path = os.path.join(self.root, self.neuron_name, self.path_leaf)

        self.v3draw_file_name = os.path.join(self.root, self.neuron_name, self.neuron_name + '.v3draw')
        self.reconstruction_file_name = os.path.join(self.root, self.neuron_name, self.neuron_name + '.v3draw' + '_{}.swc'.format(self.tracker))
        self.reconstruction_standard_file_name = os.path.join(ROOT_STANDARD, self.neuron_name, self.neuron_name + '.swc')
        self.dist_file_name = os.path.join(self.root, self.neuron_name, self.neuron_name + '_{}.dist'.format(self.tracker))

    def encoding_to_v3draw(self, flip):
        """
        将图像序列编码为 vaa3d 软件能够处理的 v3draw 格式数据
        :return:
        """
        if not os.path.isfile(self.v3draw_file_name):
            encode_to_v3draw_slice(image_path = self.image_path, v3draw_file_name = self.v3draw_file_name, flip = flip)
        else:
            return

    def generating_reconstruction(self, bkg_thresh = 1, length_thresh = 1):
        """
        使用 vaa3d 软件中的 APP2 插件生成当前神经元图像中的神经元数字化重建
        more information about this function could be found in
            https://github.com/Vaa3D/Vaa3D_Wiki/wiki/commandLineAccess.wiki
        :param bkg_thresh:
        :param length_thresh:
        :return:
        """
        command = get_reconstruction_command(input_file_name = self.v3draw_file_name, output_file_name = self.reconstruction_file_name,
                                             tracker = self.tracker, bkg_thresh = bkg_thresh, length_thresh = length_thresh)
        print(command, self.tracker)
        os.system(command)

    def comparing_with_the_stadard(self):
        """
        将生成的神经元重建结果与标准的手工重建结果进行对比
        :return:
        """
        func = '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_utilities/neuron_distance/libneuron_dist.so -f neuron_distance'
        input = '{} {}'.format(self.reconstruction_standard_file_name, self.reconstruction_file_name)
        command = '{} -i {} -o {}'.format(func, input, self.dist_file_name)
        os.system(command)

    def parse_the_dist_file(self):
        """
        解析对比结果文件，提取其中的三个距离信息
        :return:
        """
        try:
            lines = open(self.dist_file_name).readlines()
            ESA_line = lines[4]
            e = ESA_line.split()
            self.entire_structure_averages = float(e[-1])
            DSA_line = lines[5]
            e = DSA_line.split()
            self.differen_structure_average = float(e[-1])
            ADS_line = lines[8]
            e = ADS_line.split()
            self.average_different_structure = float(e[-1])
        except:
            return 0, 0, 0
        return self.entire_structure_averages, self.differen_structure_average, self.average_different_structure

    def remove(self, v3draw = False, reconstruction = False, dist_file = False):
        """
        删除生成的中间文件
        :return:
        """
        if v3draw:
            command = 'rm {}'.format(self.v3draw_file_name)
            os.system(command)
        if reconstruction:
            command = 'rm {}'.format(self.reconstruction_file_name)
            os.system(command)
        if dist_file:
            command = 'rm {}'.format(self.dist_file_name)
            os.system(command)


class BatchedProcessingSegmentedNeuron():
    """
    这个类型批量化处理多个分割后的神经元图像
    """
    def __init__(self, root, tracker = 'APP2', path_leaf = 'image_pre_init'):
        """
        :param root: 神经元图像保存的根目录
        :param tracker: 使用的追踪算法， in FUNC.keys()
        :param path_leaf: in ['image', 'image_pre_init']，正常神经元图像数据保存在 image 中，经过分割得到的神经元图像默认保存在 image_pre_init 中
        """
        self.root = root
        self.tracker = tracker
        self.path_leaf = path_leaf
        self.neuron_name_list = NEURON_NAME_TEST
        self.neuron_name_list.sort()
        self._segmented_neuron_postprocess()

    def _segmented_neuron_postprocess(self):
        """
        按神经元图像名称生成对象列表
        :return:
        """
        self.segmented_neuron_postprocess_list = OrderedDict()
        for index, neuron_name in enumerate(self.neuron_name_list):
            if not os.path.isdir(os.path.join(self.root, neuron_name, self.path_leaf)):
                print(os.path.join(self.root, neuron_name, self.path_leaf))
                continue
            self.segmented_neuron_postprocess_list[neuron_name] = SegmentedNeuronPostprocess(root = self.root,
                                                                                             tracker = self.tracker,
                                                                                             neuron_name = neuron_name,
                                                                                             path_leaf = self.path_leaf)

    def encoding_images_to_v3draw(self, flip = '10'):
        """
        批量化的将神经元图像编码为 vaa3d 软件能够处理的 v3draw 格式数据
        :param flip 翻转
        :return:
        """
        for index, neuron_name in enumerate(self.neuron_name_list):
            self.segmented_neuron_postprocess_list[neuron_name].encoding_to_v3draw(flip = flip)

    def generating_neuron_reconstruction(self, bkg_thresh = 1, length_thresh = 1):
        """
        批量化的生成神经元图像自动追踪重建
        :param root:
        :return:
        """
        for index, neuron_name in enumerate(self.neuron_name_list):
            self.segmented_neuron_postprocess_list[neuron_name].generating_reconstruction(bkg_thresh = bkg_thresh, length_thresh = length_thresh)

    def comparing_two_reconstructions(self):
        """
        批量化对比自动生成的追踪重建和标准手工重建
        :return:
        """
        for index, neuron_name in enumerate(self.neuron_name_list):
            self.segmented_neuron_postprocess_list[neuron_name].comparing_with_the_stadard()

    def batched_parse(self, result_name = None):
        """
        批量化解析提取对比文件中的结果
        :param result_name:
        :return:
        """
        result_name = result_name or 'result_{}'.format(self.tracker)
        file_save_result = open(os.path.join(self.root, result_name + '.info'), 'w')
        info_line = '{} {} {} {}'.format('neuron_name'.rjust(12), 'ESA'.rjust(12), 'DSA'.rjust(12), 'ADS'.rjust(12))
        print(info_line)
        file_save_result.write(info_line + '\n')
        number_neuron = 0
        ESA = 0.0
        DSA = 0.0
        ADS = 0.0
        for index, neuron_name in enumerate(self.neuron_name_list):
            if not os.path.isdir(os.path.join(self.root, neuron_name)):
                print('==========> {} not in {} <========'.format(neuron_name, self.root))
                continue
            if neuron_name in NEURON_NAME_NOT_COMPARE:
                continue
            esa, dsa, ads = self.segmented_neuron_postprocess_list[neuron_name].parse_the_dist_file()
            ESA += esa
            DSA += dsa
            ADS += ads
            esa_ = '{:.6f}'.format(esa)
            dsa_ = '{:.6f}'.format(dsa)
            ads_ = '{:.6f}'.format(ads)
            info_line = '{} {} {} {}'.format(neuron_name.rjust(12), esa_.rjust(12), dsa_.rjust(12), ads_.rjust(12))
            print(info_line)
            file_save_result.write(info_line + '\n')
            number_neuron += 1
        ESA /= number_neuron
        DSA /= number_neuron
        ADS /= number_neuron
        info_line = '{} {} {} {}'.format('mean'.rjust(12), '{:.6f}'.format(ESA).rjust(12), '{:.6f}'.format(DSA).rjust(12), '{:.6f}'.format(ADS).rjust(12))
        file_save_result.write(info_line + '\n')
        print(info_line)
        file_save_result.close()

    def remove(self, v3draw = False, reconstruction = False, dist_file = False):
        """
        批量化删除中间文件
        :param v3draw:
        :param reconstruction_APP2:
        :param dist_file:
        :return:
        """
        for neuron_name in self.neuron_name_list:
            self.segmented_neuron_postprocess_list[neuron_name].remove(v3draw = v3draw, reconstruction = reconstruction, dist_file = dist_file)



def parse_the_dist_file(dist_file):
    """
    解析两个神经元重建文件的对比结果
    :param dist_file:
    :return:
    """
    lines = open(dist_file).readlines()

    ESA_line = lines[4]
    e = ESA_line.split()
    entire_structure_averages = float(e[-1])

    DSA_line = lines[5]
    e = DSA_line.split()
    differen_structure_average = float(e[-1])

    ADS_line = lines[8]
    e = ADS_line.split()
    average_different_structure = float(e[-1])

    return entire_structure_averages, differen_structure_average, average_different_structure

def encoding_images_to_v3draw(root, path_leaf = 'image'):
    NEURON_NAME_TEST.sort()
    for index, neuron_name in enumerate(NEURON_NAME_TEST):
        print('--- encoding {}th / {} neuron, {}'.format(index, len(NEURON_NAME_TEST), neuron_name))
        neuron_image_path = os.path.join(root, neuron_name, path_leaf)
        v3draw_file_name_save = os.path.join(root, neuron_name, neuron_name + '.v3draw')
        encode_to_v3draw_slice(image_path = neuron_image_path, v3draw_file_name = v3draw_file_name_save)

def generating_neuron_reconstruction_APP2(root, bkg_thresh = 1, length_thresh = 1):
    """
    more information about this function could be found in
        https://github.com/Vaa3D/Vaa3D_Wiki/wiki/commandLineAccess.wiki
    :param root:
    :return:
    """
    NEURON_NAME_TEST.sort()
    func = '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_tracing/Vaa3D_Neuron2/libvn2.so -f app2'
    para = 'NULL 0 {} 1 1 0 0 {}'.format(bkg_thresh, length_thresh)
    for index, neuron_name in enumerate(NEURON_NAME_TEST):
        if not os.path.isdir(os.path.join(root, neuron_name)):
            continue
        print('=' * 20 + '  {}  '.format(neuron_name) + '=' * 20)
        input = os.path.join(root, neuron_name, neuron_name + '.v3draw')
        output = os.path.join(root, neuron_name, neuron_name + '_pre.swc')
        command = '{} -i {} -o {} -p {}'.format(func, input, output, para)
        os.system(command)

def comparing_two_reconstructions(root_standard, root_pre):
    NEURON_NAME_TEST.sort()
    func = '/home/liqiufu/Vaa3D_CentOS_64bit_v3.458/vaa3d -x /home/liqiufu/Vaa3D_CentOS_64bit_v3.458/plugins/neuron_utilities/neuron_distance/libneuron_dist.so -f neuron_distance'
    assert os.path.isdir('/home/liqiufu/Vaa3D_CentOS_64bit_v3.458'), 'change the path to vaa3d in "func ({})", and delete this line'.format(func)
    for index, neuron_name in enumerate(NEURON_NAME_TEST):
        if not os.path.isdir(os.path.join(root_pre, neuron_name)):
            continue
        print('\n' + '=' * 20 + '  {}  '.format(neuron_name) + '=' * 20)
        input = '{} {}'.format(os.path.join(root_standard, neuron_name, neuron_name + '.swc'), os.path.join(root_pre, neuron_name, neuron_name + '_pre.swc'))
        output = os.path.join(root_pre, neuron_name, neuron_name + '.dist')
        command = '{} -i {} -o {}'.format(func, input, output)
        os.system(command)

def batched_parse(root, result_name):
    NEURON_NAME_TEST.sort()
    file_save_result = open(os.path.join(root, result_name + '.info'), 'w')
    info_line = '{} {} {} {}'.format('neuron_name'.rjust(12), 'ESA'.rjust(12), 'DSA'.rjust(12), 'ADS'.rjust(12))
    print(info_line)
    file_save_result.write(info_line + '\n')
    number_neuron = 0
    ESA = 0.0
    DSA = 0.0
    ADS = 0.0
    for index, neuron_name in enumerate(NEURON_NAME_TEST):
        if not os.path.isdir(os.path.join(root, neuron_name)):
            continue
        if neuron_name in NEURON_NAME_NOT_COMPARE:
            continue
        dist_file = os.path.join(root, neuron_name, neuron_name + '.dist')
        esa, dsa, ads = parse_the_dist_file(dist_file = dist_file)
        ESA += esa
        DSA += dsa
        ADS += ads
        esa_ = '{:.6f}'.format(esa)
        dsa_ = '{:.6f}'.format(dsa)
        ads_ = '{:.6f}'.format(ads)
        info_line = '{} {} {} {}'.format(neuron_name.rjust(12), esa_.rjust(12), dsa_.rjust(12), ads_.rjust(12))
        print(info_line)
        file_save_result.write(info_line + '\n')
        number_neuron += 1
    ESA /= number_neuron
    DSA /= number_neuron
    ADS /= number_neuron
    info_line = '{} {} {} {}'.format('mean'.rjust(12), '{:.6f}'.format(ESA).rjust(12), '{:.6f}'.format(DSA).rjust(12), '{:.6f}'.format(ADS).rjust(12))
    file_save_result.write(info_line + '\n')
    print(info_line)
    file_save_result.close()

def segmented_neuron_postprocessing(root_pre, path_leaf = 'image_pre_init', result_name = 'result'):
    NEURON_NAME_TEST.sort()
    for index, neuron_name in enumerate(NEURON_NAME_TEST):
        if not os.path.isdir(os.path.join(root_pre, neuron_name)):
            print('==========> {} not in {} <========'.format(neuron_name, root_pre))
    encoding_images_to_v3draw(root_pre, path_leaf = path_leaf)   # 将神经元图像序列生成 vaa3d 能够处理的 v3draw 格式数据
    generating_neuron_reconstruction_APP2(root = root_pre)              # 使用 vaa3d 软件批量自动追踪神经元
    comparing_two_reconstructions(root_standard = ROOT_STANDARD, root_pre = root_pre)   # 将自动追踪结果与手工追踪结果进行对比
    batched_parse(root = root_pre, result_name = result_name)              # 提取并保存对比结果



if __name__ == '__main__':
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1_resized'
    #encoding_images_to_v3draw(root = root, path_leaf = 'image_pre_init')
    #generating_neuron_reconstruction_APP2(root = root)
    #comparing_two_reconstructions(root_standard = root_standard, root_pre = root)
    segmented_neuron_postprocessing(root_pre = root, path_leaf = 'image')
    #batched_parse(root)