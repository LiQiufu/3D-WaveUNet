from postprocess_huge_image.huge_image_partition_assemble import NeuronImage_Huge

if __name__ == '__main__':
    neuron_name = '000020'
    root = '/home/liqiufu/PycharmProjects/MyDataBase/Neuron_DataBase_for_Denoiser/DataBase_1/000020'
    neuron_image_huge = NeuronImage_Huge(neuron_name = neuron_name, root = root)
    neuron_image_huge.partition()
    neuron_image_huge.assemble()