import os
import shutil


def rebuild_dir(data_path):
    """
    重建文件夹
    如果文件夹不存在就创建
    如果文件夹存在就清空
    :param data_path:
    :return: None
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        shutil.rmtree(data_path)
        os.mkdir(data_path)


if __name__ == "__main__":
    # model_name_list = ['Day5ConstantBaseline']
    # model_name_list = ["LSTMDay5", "LSTMDay5Constant","LSTMDay5ConstantFC", "LSTMDayX", "LSTMDayXConstant", "LSTMDayXConstantFC"]
    model_name_list = ['BiLSTMDayX','BiLSTMDayXConstant','BiLSTMDayXConstantFC', 'BiLSTMDay5', 'BiLSTMDay5Constant', 'BiLSTMDay5ConstantFC']
    # model_name_list = ["LSTMDay1", "LSTMDay1Constant","LSTMDay1ConstantFC"]
    dataset_name_list = ["aki_stage", "ards", "dead", "dic", "liver_injury", "shock"]
    for dataset_name in dataset_name_list:
        for model_name in model_name_list:
            rebuild_dir("checkpoint/log/{}/{}".format(dataset_name, model_name))
            rebuild_dir("checkpoint/model/{}/{}".format(dataset_name, model_name))
            rebuild_dir("picture/{}/{}".format(dataset_name, model_name))
        # shutil.rmtree("checkpoint/log/{}/{}".format(dataset_name, model_name_list))
        # shutil.rmtree("checkpoint/model/{}/{}".format(dataset_name, model_name_list))
        # shutil.rmtree("picture/{}/{}".format(dataset_name, model_name_list))
