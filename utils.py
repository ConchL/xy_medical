import random

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score


def evaluate(valid_dataloader, model, device,label_name):
    model.eval()
    all_predictions = []
    all_labels = []
    for i, data in enumerate(valid_dataloader):
        inputs, labels = data

        label_list = [label[label_name] for label in labels]
        input_list = []
        common_list = []
        for t in inputs:
            temp = []
            temp.append(t["day1"])
            temp.append(t["day2"])
            temp.append(t["day3"])
            temp.append(t["day4"])
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
            logits = model(input_list, common_list)
            # logits = model(input_list)

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