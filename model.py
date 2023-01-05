import json
import logging
import os

import pickle as pkl
import time

import numpy as np
from numpy import average
from pandas import NA
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence


# 仅时序变量，按4天时序预测
class LSTM(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(LSTM, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size  # 调参 一开始为200
        self.layers = layers

        self.lstm = nn.LSTM(90, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                            bidirectional=False)  # num_layers
        self.CE_loss = nn.CrossEntropyLoss()

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

        # self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

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


# 仅时序变量，把4天看作1天来预测
class LSTMBaseline(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(LSTMBaseline, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(360, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                            bidirectional=False)
        self.CE_loss = nn.CrossEntropyLoss()

        self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

        inputs = inputs.view(inputs.size(0), 1, -1)

        outputs, _ = self.lstm(inputs)  # B * S * H
        # B * H
        x = outputs[:, -1, :]
        # x, _ = torch.max(outputs, 1)
        # x = self.dropout(x)
        logits = self.linear(x)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits


# 把常变量加入，按4天时序预测
class LSTMFC(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(LSTMFC, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(90, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                            bidirectional=False)
        self.CE_loss = nn.CrossEntropyLoss()

        self.common_inputs = 63  # 常变量数目
        self.common_hidden_size = int(self.hidden_size/4)

        self.common_linear = nn.Linear(in_features=self.common_inputs, out_features=self.common_hidden_size)

        self.linear = nn.Linear(in_features=self.hidden_size+self.common_hidden_size, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, common_inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

        outputs, _ = self.lstm(inputs)  # B * S * H

        # B * H
        x = outputs[:, -1, :]
        # x, _ = torch.max(outputs, 1)

        # x = self.dropout(x)
        common_outputs = self.common_linear(common_inputs)

        add_common_outputs = torch.cat((x, common_outputs), dim=1)  # 将时序变量和常变量拼接起来

        logits = self.linear(add_common_outputs)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits


# 把常变量加入，把4天看作1天来预测
class LSTMFCBaseline(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(LSTMFCBaseline, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.layers = layers

        self.lstm = nn.LSTM(360, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
                            bidirectional=False)
        self.CE_loss = nn.CrossEntropyLoss()

        self.common_inputs = 63  # 常变量数目

        self.common_hidden_size = int(self.hidden_size / 4)  # 常变量隐藏层节点个数

        self.common_linear = nn.Linear(in_features=self.common_inputs, out_features=self.common_hidden_size)

        self.linear = nn.Linear(in_features=self.hidden_size + self.common_hidden_size, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, common_inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

        inputs = inputs.view(inputs.size(0), 1, -1)
        print('inputs:', inputs.shape)
        outputs, _ = self.lstm(inputs)  # B * S * H
        print('outputs:', outputs.shape)
        # B * H
        x = outputs[:, -1, :]
        # x, _ = torch.max(outputs, 1)
        # x = self.dropout(x)
        print('x:', x.shape)
        print('common_inputs:', common_inputs.shape)
        common_outputs = self.common_linear(common_inputs)
        print('common_outputs:', common_outputs.shape)
        add_common_outputs = torch.cat((x, common_outputs), dim=1)  # 将时序变量和常变量拼接起来
        print('add_common_outputs:', add_common_outputs.shape)
        logits = self.linear(add_common_outputs)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits


class FCCommonBaseline(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(FCCommonBaseline, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.layers = layers

        # self.lstm = nn.LSTM(360, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
        #                     bidirectional=False)
        self.CE_loss = nn.CrossEntropyLoss()

        self.common_inputs = 63  # 常变量数目

        # self.common_hidden_size = int(self.hidden_size / 4)  # 常变量隐藏层节点个数

        # self.common_linear = nn.Linear(in_features=self.common_inputs, out_features=self.common_hidden_size)

        # self.linear = nn.Linear(in_features=self.hidden_size + self.common_hidden_size, out_features=num_classes)

        self.linear1 = nn.Linear(in_features=360 + self.common_inputs, out_features=512)

        self.linear2 = nn.Linear(in_features=512, out_features=1024)

        self.linear3 = nn.Linear(in_features=1024, out_features=512)

        self.linear4 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, common_inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)
        # inputs: [B, Day, 时序变量数]
        # inputs: [256, 1, 360]
        inputs = inputs.view(inputs.size(0), 1, -1)
        # outputs: [256, 1, 128]
        # outputs, _ = self.lstm(inputs)  # B * S * H
        # LSTM输出：output, (h_n, c_n)（每一层，最后一个time-step的输出h和c）

        # B * H
        # x: [256, 128]
        x = inputs[:, -1, :]
        # x, _ = torch.max(outputs, 1)
        # x = self.dropout(x)

        # common_inputs: [256, 63]
        # common_outputs: [256, 32]
        # common_outputs = self.common_linear(common_inputs)

        # add_common_outputs:[256, 160]
        add_common_inputs = torch.cat((x, common_inputs), dim=1)  # 将时序变量和常变量拼接起来

        add_common_outputs1 = self.linear1(add_common_inputs)

        add_common_outputs2 = self.linear2(add_common_outputs1)

        add_common_outputs3 = self.linear3(add_common_outputs2)

        logits = self.linear4(add_common_outputs3)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits


class FCBaseline(nn.Module):
    def __init__(self, layers, hidden_size, device, num_classes=None):
        super(FCBaseline, self).__init__()

        self.device = device
        self.num_classes = num_classes

        self.hidden_size = hidden_size
        self.layers = layers

        # self.lstm = nn.LSTM(360, hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True,
        #                     bidirectional=False)
        self.CE_loss = nn.CrossEntropyLoss()

        # self.linear = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

        self.linear1 = nn.Linear(in_features=360, out_features=512)

        self.linear2 = nn.Linear(in_features=512, out_features=1024)

        self.linear3 = nn.Linear(in_features=1024, out_features=512)

        self.linear4 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)  # dropout训练

        # B * 500 -》 B * H -》在H上拼接，做成 B * 2H -》 B* 2

    def forward(self, inputs=None, labels=None):

        # if <PAD> is fed into lstm encoder, it may be cause the error.

        inputs = torch.tensor(inputs).to(self.device)

        inputs = inputs.view(inputs.size(0), 1, -1)

        # outputs, _ = self.lstm(inputs)  # B * S * H
        # B * H
        x = inputs[:, -1, :]
        # x, _ = torch.max(outputs, 1)
        # x = self.dropout(x)

        outputs1 = self.linear1(x)

        outputs2 = self.linear2(outputs1)

        outputs3 = self.linear3(outputs2)

        logits = self.linear4(outputs3)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits