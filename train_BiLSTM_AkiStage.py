'''
nohup python -u train.py --model_type=TextCNN --gpu_id=1 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_CNN.pth > logs/CAIL2018_TextCNN.log &
nohup python -u train.py --model_type=TextRNN --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_RNN.pth > logs/CAIL2018_TextRNN.log &
nohup python -u train.py --model_type=Transformer --gpu_id=3 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_Transformer.pth > logs/CAIL2018_Transformer.log &

nohup python -u train.py --model_type=LSTM --gpu_id=0 --dataset_type=CAIL2018 -s=./checkpoints/CAIL2018_LSTM.pth -log=logs/CAIL2018_LSTM.log &



'''
import shutil
import sys

from utils import get_precision_recall_f1, evaluate, set_random_seed
from model import LSTM, LSTMFC, LSTMBaseline, LSTMFCBaseline, FCCommonBaseline, FCBaseline, BiLSTMDay5, BiLSTMDayX, \
    BiLSTMDay5Constant, BiLSTMDayXConstant, BiLSTMDay5ConstantFC, BiLSTMDayXConstantFC, Day5Baseline, Day5ConstantBaseline, \
    LSTMDay1ConstantFC, LSTMDay1Constant, LSTMDay1
from dataset import MedicalData
import argparse
import os
import logging

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# def seed_it(seed):
#     random.seed(seed) #可以注释掉
#     os.environ["PYTHONSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) #这个懂吧
#     torch.backends.cudnn.deterministic = True #确定性固定
#     torch.backends.cudnn.benchmark = True #False会确定性地选择算法，会降低性能
#     torch.backends.cudnn.enabled = True  #增加运行效率，默认就是True
#     torch.manual_seed(seed)
# seed_it(1314)


if __name__ == "__main__":

    set_random_seed(2022)

    parser = argparse.ArgumentParser(description="Medical Prediction")
    # parser.add_argument('--batch_size', '-b', type=int, default=128, help='default: 128')
    # parser.add_argument('--epochs', type=int, default=30, help='default: 30')
    # parser.add_argument('--learning_rate', type=float, default=1e-3, help='default: 1e-3')
    # parser.add_argument('--input_max_length', '-l', type=int, default=500, help='default: 500')
    # parser.add_argument('--random_seed', type=int, default=3407, help='default: 3407')
    # parser.add_argument('--num_classes', type=int, default=119, help='default: 119')
    # parser.add_argument('--model_type', type=str, default='LSTM',
    #                     help='[TextRNN, TextCNN, TextRCNN, TextAttRNN, Transformer, SAttCaps, Capsule]')
    # parser.add_argument('--gpu_id', type=str, default='0', help='default: 0')
    # parser.add_argument('--resume', '-r', action='store_true', help='default: False')
    # parser.add_argument('--word_embed_path', type=str, default='./datasets/word_embed/small_w2v.txt')
    # parser.add_argument('--dataset_type', type=str, default='CAIL2018', help='[CAIL2018]')
    # parser.add_argument('--save_path', '-s', type=str, default='./checkpoints/model_baseline_best.pth')
    # parser.add_argument('--log_file_name', '-log', type=str, default='./checkpoint/log/model_baseline_best.log')
    # parser.add_argument('--resume_checkpoint_path', '-c', type=str, default='./checkpoints/model_baseline_best.pth')

    parser.add_argument('--label_name', type=str, default='dic', help='[aki_stage, ards, dic, dead, liver_injury, shock]')
    parser.add_argument('--model_type', type=str, default='BiLSTMDay5ConstantFC',
                        help='[LSTM, LSTMFC, LSTMFCBaseline, LSTMBaseline, '
                             'BiLSTMDayX, BiLSTMDayXConstant, '
                             'BiLSTMDayXConstantFC, BiLSTMDay5, '
                             'BiLSTMDay5Constant, BiLSTMDay5ConstantFC, LSTMDay1, '
                             'LSTMDay1Constant, LSTMDay1ConstantFC, Day5Baseline, Day5ConstantBaseline]')
    parser.add_argument('--hidden_size', type=int, default=256, help='[16, 32, 64, 128, 256, 512]')
    parser.add_argument('--layers', '-l', type=int, default=1, help='[1, 2, 3, 4]')
    parser.add_argument('--epochs', type=int, default=100, help='default: 100')
    parser.add_argument('--windows', type=int, default=1, help='[1, 2, 3, 4]')
    args = parser.parse_args()
    print(args)
    label_name = args.label_name
    model_type = args.model_type
    layers = args.layers
    hidden_size = args.hidden_size
    epochs = args.epochs
    windows = args.windows

    # log_file_name = args['log_file_name']

    # model_type = 'LSTMBaseline'

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

    # check the device
    # torch.cuda.set_device(1)
    device = 'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))

    torch.cuda.empty_cache()

    # seed random seed
    # set_random_seed(args.random_seed)

    # prepare training data
    # train_path = 'train.csv'
    # valid_path = 'valid.csv'
    # test_path = 'test.csv'

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

    # load the model
    if model_type == 'LSTM':
        model = LSTM(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'LSTMFC':
        model = LSTMFC(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'LSTMBaseline':
        model = LSTMBaseline(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'LSTMFCBaseline':
        model = LSTMFCBaseline(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'FCCommonBaseline':
        model = FCCommonBaseline(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'FCBaseline':
        model = FCBaseline(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDay5':
        model = BiLSTMDay5(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDayX':
        model = BiLSTMDayX(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDay5Constant':
        model = BiLSTMDay5Constant(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDayXConstant':
        model = BiLSTMDayXConstant(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDay5ConstantFC':
        model = BiLSTMDay5ConstantFC(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'BiLSTMDayXConstantFC':
        model = BiLSTMDayXConstantFC(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'Day5Baseline':
        model = Day5Baseline(layers, hidden_size, windows, device, num_classes=num_classes)
    elif model_type == 'Day5ConstantBaseline':
        model = Day5ConstantBaseline(layers, hidden_size, windows, device, num_classes=num_classes)
    elif model_type == 'LSTMDay1':
        model = LSTMDay1(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'LSTMDay1Constant':
        model = LSTMDay1Constant(layers, hidden_size, device, num_classes=num_classes)
    elif model_type == 'LSTMDay1ConstantFC':
        model = LSTMDay1ConstantFC(layers, hidden_size, device, num_classes=num_classes)

    else:
        raise NameError

    logging.info(f'Load {model_type} model.')

    optimizer = AdamW(model.parameters(), lr=0.001)
    # scheduler = ReduceLROnPlateau(
    #    optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # max for acc

    # resume checkpoint
    # if args.resume:
    #     checkpoint = torch.load(args.resume_checkpoint_path, map_location='cpu')
    #     model.load_state_dict(checkpoint['state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     # 'cpu' to 'gpu'
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device)
    #
    #     logging.info(
    #         f"Resume model and optimizer from checkpoint '{args.resume_checkpoint_path}' with epoch {checkpoint['epoch']} and best F1 score of {checkpoint['best_f1_score']}")
    #     logging.info(f"optimizer lr: {optimizer.param_groups[0]['lr']}")
    #     start_epoch = checkpoint['epoch']
    #     best_f1_score = checkpoint['best_f1_score']
    # else:
    #     # start training process
    #     start_epoch = 0
    #     best_f1_score = 0
    start_epoch = 0
    best_f1_score = 0

    model.to(device)

    loss_list = []

    valid_data = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        loss_epoch = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data

            if model_type == 'BiLSTMDayX' or model_type == 'BiLSTMDayXConstant' or model_type == 'BiLSTMDayXConstantFC':
                if windows == 4 or label_name == 'dead':
                    label_list = [label[label_name] for label in labels]
                else:
                    label_list = [label["{}{}".format(label_name, windows+1)] for label in labels]
            else:
                label_list = [label[label_name] for label in labels]
            input_list = []
            common_list = []
            for t in inputs:
                temp = []
                if model_type == 'BiLSTMDayX' or model_type == 'BiLSTMDayXConstantFC' or model_type == 'LSTMDay1' or model_type == 'LSTMDay1ConstantFC':
                    for i in range(1, windows + 1):
                        temp.append(t["day{}".format(i)])

                elif model_type == 'BiLSTMDay5' or model_type == 'BiLSTMDay5ConstantFC' or model_type == 'Day5Baseline':
                    for i in range(5 - windows, 5):
                        temp.append(t["day{}".format(i)])

                elif model_type == 'BiLSTMDayXConstant' or model_type == 'LSTMDay1Constant':
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
                elif model_type == 'BiLSTMDay5Constant' or model_type == 'Day5ConstantBaseline':
                    for i in range(5 - windows, 5):
                        temp.append(t["day{}constant".format(i)])
                    # if windows == 1:
                    #     temp.append(t["day4constant"])
                    # elif windows == 2:
                    #     temp.append(t["day3constant"])
                    #     temp.append(t["day4constant"])
                    # elif windows == 3:
                    #     temp.append(t["day2constant"])
                    #     temp.append(t["day3constant"])
                    #     temp.append(t["day4constant"])
                    # elif windows == 4:
                    #     temp.append(t["day1constant"])
                    #     temp.append(t["day2constant"])
                    #     temp.append(t["day3constant"])
                    #     temp.append(t["day4constant"])
                # temp.append(t["day1"])
                # temp.append(t["day2"])
                # temp.append(t["day3"])
                # temp.append(t["day4"])
                # temp.append(t["day5"])
                input_list.append(temp)
                common_list.append(t["common"])
            # move data to device
            labels = torch.from_numpy(np.array(label_list)).to(device)
            common_list = torch.from_numpy(np.array(common_list)).to(device).to(torch.float)

            # forward and backward propagations

            if model_type[-2:]=='FC':
                loss, logits = model(input_list, common_list, labels)
            else:
                loss, logits = model(input_list, labels)

            # 加常变量需要修改此处
            # loss, logits = model(input_list, common_list, labels)
            # loss, logits = model(input_list, labels)
            loss_epoch += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 反向传播后参数更新

            # 输出训练集acc pre recall f1_score
            if (i + 1) % 10 == 0:
                predictions = logits.softmax(
                    dim=1).detach().cpu().numpy()
                labels = labels.cpu().numpy()

                logging.info(
                    f'epoch{epoch + 1}, step{i + 1:5d}, loss: {loss.item():.4f}')

                pred = np.argmax(predictions, axis=1)
                accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
                    labels, pred, 'macro')

                logging.info(
                    f'train disease macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')

        loss_list.append(loss_epoch / len(train_dataloader))
        if (epoch + 1) % 1 == 0:
            logging.info('Evaluating the model on validation set...')
            accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu, _, __ = evaluate(valid_dataloader, windows,
                                                                                       model_type, model,
                                                                                       device, label_name)

            valid_data[epoch] = {"acc": accuracy_accu, "f1": f1_macro_accu}
            logging.info(
                f'valid disease macro accuracy:{accuracy_accu:.4f}, precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
            # scheduler.step(f1_macro_accu)

            if f1_macro_accu > best_f1_score:
                best_f1_score = f1_macro_accu
                logging.info(
                    f"the valid best average F1 score is {best_f1_score}.")
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1_score': best_f1_score,
                }
                torch.save(state, save_path)
                logging.info(f'Save model in path: {save_path}')

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

    print('loss_list = ', loss_list)

    logging.info('valid_data:')

    for i in valid_data.keys():
        logging.info("Epoch:{} \t acc:{} f1:{}".format(i + 1, valid_data[i]["acc"], valid_data[i]["f1"]))

    # 绘制loss曲线
    plt.plot([i for i in range(1, epochs + 1)], loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('{}_Loss_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type))
    fig_name = '{}_Loss_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type)
    plt.savefig('picture/{}/{}/{}.png'.format(label_name, model_type, fig_name))
    plt.show()

    # 绘制ROC_AOC曲线
    fpr, tpr, threshold = roc_curve(all_labels, np.argmax(all_predictions, axis=1))  # y_data, y_score
    roc_auc = auc(fpr, tpr)

    plt.figure()

    logging.info(f'test disease ROC_AUC:{roc_auc:.4f}')

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(title + ' RNN-LSTM Model ')
    # plt.legend(loc="lower right")
    #
    # plt.plot(fpr, tpr, label='ROC曲线')
    plt.title('{}_ROC_AUC_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    fig_name = '{}_ROC_AUC_{}_{}_window{}_{}'.format(label_name, layers, hidden_size, windows, model_type)
    plt.savefig('picture/{}/{}/{}.png'.format(label_name, model_type, fig_name))
    plt.show()
