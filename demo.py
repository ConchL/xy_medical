'''
nohup python -u train.py --model_type=TextCNN --gpu_id=1 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_CNN.pth > logs/CAIL2018_TextCNN.log &
nohup python -u train.py --model_type=TextRNN --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_RNN.pth > logs/CAIL2018_TextRNN.log &
nohup python -u train.py --model_type=Transformer --gpu_id=3 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_Transformer.pth > logs/CAIL2018_Transformer.log &

nohup python -u train.py --model_type=LSTM --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_LSTM.pth -log=logs/CAIL2018_LSTM.log &



'''
from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn

import argparse
import os
import logging

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
import numpy as np
import random
import pandas as pd

from sklearn.metrics import accuracy_score, roc_curve, auc


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


def evaluate(valid_dataloader, windows, model_type, model, device, label_name):
    model.eval()
    all_predictions = []
    all_labels = []
    # print('windows:',windows)
    # print('model type:',model_type)
    for i, data in enumerate(valid_dataloader):
        inputs, labels = data

        if model_type == 'LSTMDayX' or model_type == 'LSTMDayXConstant' or model_type == 'LSTMDayXConstantFC':
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
            if model_type == 'LSTMDayX' or model_type == 'LSTMDayXConstantFC' or model_type == 'LSTMDay1' or model_type == 'LSTMDay1ConstantFC':
                for i in range(1, windows + 1):
                    temp.append(t["day{}".format(i)])

            elif model_type == 'LSTMDay5' or model_type == 'LSTMDay5ConstantFC' or model_type == 'Day5Baseline':
                for i in range(5 - windows, 5):
                    temp.append(t["day{}".format(i)])

            elif model_type == 'LSTMDayXConstant' or model_type == 'LSTMDay1Constant':
                for i in range(1, windows + 1):
                    temp.append(t["day{}constant".format(i)])

            elif model_type == 'LSTMDay5Constant' or model_type == 'Day5ConstantBaseline':
                for i in range(5 - windows, 5):
                    temp.append(t["day{}constant".format(i)])

            input_list.append(temp)
            common_list.append(t["common"])
        # move data to device
        labels = torch.from_numpy(np.array(label_list)).to(device)
        common_list = torch.from_numpy(np.array(common_list)).to(device).to(torch.float)

        with torch.no_grad():
            # forward
            # 加常变量需修改此处
            # logits = model(input_list, common_list)
            logits = model(input_list)

        all_predictions.append(logits.softmax(dim=1).detach().cpu())
        all_labels.append(labels.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    accuracy, p_macro, r_macro, f1_macro = get_precision_recall_f1(all_labels, np.argmax(all_predictions, axis=1),
                                                                   'macro')
    return accuracy, p_macro, r_macro, f1_macro, all_predictions, all_labels


class MedicalData(Dataset):
    def __init__(self, mode='train', train_file=None, valid_file=None, test_file=None, label_name=None):
        assert mode in ['train', 'valid', 'test'], f"mode should be set to the one of ['train', 'valid', 'test']"
        self.mode = mode
        self.dataset = []
        if mode == 'train':
            self.dataset = self._load_data(train_file, label_name)
            print(f'Number of training dataset: {len(self.dataset)}')
        elif mode == 'valid':
            self.dataset = self._load_data(valid_file, label_name)
            print(f'Number of validation dataset: {len(self.dataset)}')
        else:
            self.dataset = self._load_data(test_file, label_name)
            print(f'Number of test dataset: {len(self.dataset)}.')

    def __getitem__(self, idx):
        features_content = self.dataset[idx]['data']
        if self.mode in ['train', 'valid', 'test']:
            labels = self.dataset[idx]['label']
            return features_content, labels
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_function(batch):
        features_content, labels = zip(*batch)
        return features_content, labels

    def _format_label(self,label):
        if np.isnan(label):
            label = 0
        else:
            label = int(label)

        if label > 1:
            label = 1
        return label

    def _format_data(self,data):
        if data == "M":
            return 1.0
        if data == "F":
            return 0.0

        if np.isnan(data):
            return 0.0
        else:
            return float(data)





    def _load_data(self, file_name, label_name):

        df = pd.read_csv('dataset/{}/{}'.format(label_name, file_name))

        common_need = ['Coagulation.dysfunction', 'age', 'age_raw', 'aids', 'alcohol_abuse',
                            'antibiotic.used_kinds_n',
                            'antibiotic_group.used_kinds_n', 'antibiotic_group1', 'antibiotic_group2',
                            'antibiotic_group3',
                            'antibiotic_group4', 'antibiotic_group5', 'antibiotic_group6', 'antibiotic_group7',
                            'antibiotic_group8',
                            'antibiotic_group9', 'apsiii', 'apsiii_prob', 'basic_disease_n', 'blood_loss_anemia',
                            'cardiac_arrhythmias',
                            'chronic_pulmonary', 'coagulopathy', 'congestive_heart_failure', 'deficiency_anemias',
                            'depression', 'diabetes_complicated',
                            'diabetes_uncomplicated', 'drug_abuse', 'expire_flag', 'fluid_electrolyte', 'gender',
                            'hypertension', 'hypothyroidism',
                            'key_disease_group.diabete', 'key_disease_group.heat', 'key_disease_group.liver_disease',
                            'key_disease_group.lung',
                            'key_disease_group.other', 'key_disease_group.renal_failure', 'key_disease_group.tumor',
                            'key_disease_n', 'liver_disease', 'lods', 'los_hospital', 'los_icu', 'lymphoma',
                            'metastatic_cancer', 'obesity', 'other_neurological', 'paralysis', 'peptic_ulcer',
                            'peripheral_vascular',
                            'psychoses', 'pulmonary_circulation', 'qsofa', 'renal_failure', 'rheumatoid_arthritis',
                            'sirs', 'sofa', 'solid_tumor', 'valvular_disease', 'weight_loss']

        result = []
        for index, row in df.iterrows():
            data = {}
            data["day1"] = []
            data["day2"] = []
            data["day3"] = []
            data["day4"] = []
            data['common'] = []
            # data["day5"] = []
            data["day1constant"] = []
            data["day2constant"] = []
            data["day3constant"] = []
            data["day4constant"] = []

            label = {}

            for col in df.columns:
                if col == "dead":
                    label["dead"] = self._format_label(row[col])
                elif col == "shock_Day5":
                    label["shock"] = self._format_label(row[col])
                elif col == "ards_Day5":
                    label["ards"] = self._format_label(row[col])
                elif col == "aki_stage_Day5":
                    label["aki_stage"] = self._format_label(row[col])
                elif col == "liver_injury_Day5":
                    label["liver_injury"] = self._format_label(row[col])
                elif col == "dic_Day5":
                    label["dic"] = self._format_label(row[col])
                elif col == "shock_Day2":
                    label["shock2"] = self._format_label(row[col])
                elif col == "shock_Day3":
                    label["shock3"] = self._format_label(row[col])
                elif col == "shock_Day4":
                    label["shock4"] = self._format_label(row[col])
                elif col == "ards_Day2":
                    label["ards2"] = self._format_label(row[col])
                elif col == "ards_Day3":
                    label["ards3"] = self._format_label(row[col])
                elif col == "ards_Day4":
                    label["ards4"] = self._format_label(row[col])
                elif col == "aki_stage_Day2":
                    label["aki_stage2"] = self._format_label(row[col])
                elif col == "aki_stage_Day3":
                    label["aki_stage3"] = self._format_label(row[col])
                elif col == "aki_stage_Day4":
                    label["aki_stage4"] = self._format_label(row[col])
                elif col == "liver_injury_Day2":
                    label["liver_injury2"] = self._format_label(row[col])
                elif col == "liver_injury_Day3":
                    label["liver_injury3"] = self._format_label(row[col])
                elif col == "liver_injury_Day4":
                    label["liver_injury4"] = self._format_label(row[col])
                elif col == "dic_Day2":
                    label["dic2"] = self._format_label(row[col])
                elif col == "dic_Day3":
                    label["dic3"] = self._format_label(row[col])
                elif col == "dic_Day4":
                    label["dic4"] = self._format_label(row[col])
                else:
                    if col != "icustay_id":
                        if "Day1" in col:
                            data["day1"].append(self._format_data(row[col]))
                            data["day1constant"].append(self._format_data(row[col]))
                        if "Day2" in col:
                            data["day2"].append(self._format_data(row[col]))
                            data["day2constant"].append(self._format_data(row[col]))
                        if "Day3" in col:
                            data["day3"].append(self._format_data(row[col]))
                            data["day3constant"].append(self._format_data(row[col]))
                        if "Day4" in col:
                            data["day4"].append(self._format_data(row[col]))
                            data["day4constant"].append(self._format_data(row[col]))
                        # if "Day5" in col:
                        #     data["day5"].append(self._format_data(row[col]))
                if col in common_need:
                    data["common"].append(self._format_data(row[col]))
                    data["day1constant"].append(self._format_data(row[col]))
                    data["day2constant"].append(self._format_data(row[col]))
                    data["day3constant"].append(self._format_data(row[col]))
                    data["day4constant"].append(self._format_data(row[col]))



            temp = {}
            temp["data"] = data
            temp["label"] = label
            result.append(temp)
        return result


class LSTMDay1Constant(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(LSTMDay1Constant, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size  # 调参 一开始为200
        self.layers = layers

        self.lstm = nn.LSTM(153, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                            bidirectional=False)  # num_layers
        self.CE_loss = nn.CrossEntropyLoss()

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

        # self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

        # print('input_size:',inputs.shape)

        outputs, _ = self.lstm(inputs)  # B * S * H
        # B * H
        x = outputs[:, -1, :]  # torh.mean(1)
        # x, _ = torch.max(outputs, 1)
        # x = self.dropout(x)
        logits = self.linear(x)  # dropout

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits

if __name__ == "__main__":

    set_random_seed(2022)

    parser = argparse.ArgumentParser(description="Medical Prediction")

    parser.add_argument('--label_name', type=str, default='liver_injury',
                        help='[aki_stage, ards, dic, dead, liver_injury, shock]')
    parser.add_argument('--model_type', type=str, default='LSTMDay1Constant',
                        help='[LSTM, LSTMFC, LSTMFCBaseline, LSTMBaseline, '
                             'LSTMDayX, LSTMDayXConstant, '
                             'LSTMDayXConstantFC, LSTMDay5, '
                             'LSTMDay5Constant, LSTMDay5ConstantFC, LSTMDay1, '
                             'LSTMDay1Constant, LSTMDay1ConstantFC, Day5Baseline, Day5ConstantBaseline]')
    parser.add_argument('--hidden_size', type=int, default=128, help='[16, 32, 64, 128, 256, 512]')
    parser.add_argument('--layers', '-l', type=int, default=2, help='[1, 2, 3, 4]')
    parser.add_argument('--epochs', type=int, default=100, help='default: 100')
    parser.add_argument('--windows', type=int, default=4, help='[1, 2, 3, 4]')
    args = parser.parse_args()
    print(args)
    label_name = args.label_name
    model_type = args.model_type
    layers = args.layers
    hidden_size = args.hidden_size
    epochs = args.epochs
    windows = args.windows

    log_file_name = 'checkpoint/log/{}/{}/{}_{}_{}_window{}_{}.log'.format(label_name, model_type, label_name, layers,
                                                                           hidden_size, windows, model_type)
    save_path = 'checkpoint/model/{}/{}/{}_{}_{}_window{}_{}.pth'.format(label_name, model_type, label_name, layers,
                                                                         hidden_size, windows, model_type)
    gpu_id = '0'

    if os.path.exists(log_file_name):
        # 如果文件存在，删除文件
        os.remove(log_file_name)

    # parser.add_argument('--log_file_name', '-log', type=str, default='checkpoint/log/{}/{}_{}.log'.format(label_name, label_name, model_type))

    logging.basicConfig(filename=log_file_name,
                        level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    model_name = '{}_layer:{}_hidden_size:{}_window:{}_{}'.format(label_name, layers, hidden_size, windows,
                                                                  model_type)
    logging.info(f"model name is: {model_name}")

    device = 'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    torch.cuda.empty_cache()


    train_path = 'train_add_normal_std_{}.csv'.format(label_name)  # 有5天时间变量（90/day）和常规变量的数据
    valid_path = 'valid_add_normal_std_{}.csv'.format(label_name)
    test_path = 'test_add_normal_std_{}.csv'.format(label_name)

    logging.info(f'Train_path: {train_path}')
    logging.info(f'Valid_path: {valid_path}')
    logging.info(f'Test_path: {test_path}')

    training_data = MedicalData(mode='train', train_file=train_path, label_name=label_name)
    valid_data = MedicalData(mode='valid', valid_file=valid_path, label_name=label_name)
    test_data = MedicalData(mode='test', test_file=test_path, label_name=label_name)

    batch_size = 256
    num_classes = 2
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=training_data.collate_function, drop_last=True)
    valid_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)

    if model_type == 'LSTMDay1Constant':
        model = LSTMDay1Constant(layers, hidden_size, device, num_classes=num_classes)

    else:
        raise NameError

    logging.info(f'Load {model_type} model.')

    optimizer = AdamW(model.parameters(), lr=0.001)


    model.to(device)


    # Load Best Checkpoint
    logging.info('Load best checkpoint for testing model.')
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu, all_predictions, all_labels = evaluate(test_dataloader,
                                                                                                     windows,
                                                                                                     model_type,
                                                                                                     model, device,
                                                                                                     label_name)
    logging.info(
        f'test disease macro accuracy:{accuracy_accu:.4f}, precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')

    # 绘制ROC_AOC曲线
    fpr, tpr, threshold = roc_curve(all_labels, np.argmax(all_predictions, axis=1))  # y_data, y_score
    roc_auc = auc(fpr, tpr)

    plt.figure()

    logging.info(f'test disease ROC_AUC:{roc_auc:.4f}')

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title('{}_ROC_AUC_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    fig_name = '{}_ROC_AUC_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type)
    plt.savefig('picture/{}/{}/{}.png'.format(label_name, model_type, fig_name))
    plt.show()
