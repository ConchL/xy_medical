import random

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score


def evaluate(valid_dataloader, windows, model_type, model, device,label_name):
    model.eval()
    all_predictions = []
    all_labels = []
    # print('windows:',windows)
    # print('model type:',model_type)
    for i, data in enumerate(valid_dataloader):
        inputs, labels = data

        if model_type == 'LSTMDayX' or model_type == 'LSTMDayXConstant' or model_type == 'LSTMDayXConstantFC' or model_type == 'BiLSTMDayX' or model_type == 'BiLSTMDayXConstant' or model_type == 'BiLSTMDayXConstantFC':
            if windows == 4 or label_name == 'dead':
                label_list = [label[label_name] for label in labels]
            else:
                label_list = [label["{}{}".format(label_name, windows + 1)] for label in labels]
        else:
            label_list = [label[label_name] for label in labels]
        input_list = []
        common_list = []
        for t in inputs:
            temp = []
            if model_type == 'LSTMDayX' or model_type == 'LSTMDayXConstantFC' or model_type == 'BiLSTMDayX' or model_type == 'BiLSTMDayXConstantFC' or model_type == 'LSTMDay1' or model_type == 'LSTMDay1ConstantFC':
                for i in range(1, windows + 1):
                    temp.append(t["day{}".format(i)])

            elif model_type == 'LSTMDay5' or model_type == 'LSTMDay5ConstantFC' or model_type == 'BiLSTMDay5' or model_type == 'BiLSTMDay5ConstantFC' or model_type == 'Day5Baseline':
                for i in range(5 - windows, 5):
                    temp.append(t["day{}".format(i)])

            elif model_type == 'LSTMDayXConstant' or model_type == 'BiLSTMDayXConstant' or model_type == 'LSTMDay1Constant':
                for i in range(1, windows + 1):
                    temp.append(t["day{}constant".format(i)])
                # if windows == 1:
                #     temp.append(t["day1constant"])
                # elif windows == 2:
                #     temp.append(t["day1constant"])
                #     temp.append(t["day2constant"])
                # elif windows == 3:
                #     temp.append(t["day1constant"])
                #     temp.append(t["day2constant"])
                #     temp.append(t["day3constant"])
                # elif windows == 4:
                #     temp.append(t["day1constant"])
                #     temp.append(t["day2constant"])
                #     temp.append(t["day3constant"])
                #     temp.append(t["day4constant"])
            elif model_type == 'LSTMDay5Constant' or model_type == 'BiLSTMDay5Constant' or model_type == 'Day5ConstantBaseline':
                for i in range(5 - windows, 5):
                    temp.append(t["day{}constant".format(i)])
            # temp.append(t["day1"])
            # temp.append(t["day2"])
            # temp.append(t["day3"])
            # temp.append(t["day4"])
            # 第五天输入不放进去
            # temp.append(t["day5"])
            input_list.append(temp)
            common_list.append(t["common"])
        # move data to device
        labels = torch.from_numpy(np.array(label_list)).to(device)
        common_list = torch.from_numpy(np.array(common_list)).to(device).to(torch.float)

        with torch.no_grad():
            # forward
            # 加常变量需修改此处
            # logits = model(input_list, common_list)
            # logits = model(input_list)
            if model_type[-2:] == 'FC':
                loss, logits = model(input_list, common_list, labels)
            else:
                loss, logits = model(input_list, labels)

        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1),
                                                                   'macro')
    return accuracy, p_macro, r_macro, f1_macro, all_predictions, all_labels


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def set_random_seed(random_seed):
    # This is the random_seed of hope.
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # check whether random.
    # torch.use_deterministic_algorithms(True