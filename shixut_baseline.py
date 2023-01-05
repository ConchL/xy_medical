import torch
import torch.nn as nn

# 仅时序变量，把4天看作1天来预测
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

        self.linear3 = nn.Linear(in_features=1024, out_features=num_classes)

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

        logits = self.linear3(outputs2)

        if labels is not None:
            loss = self.CE_loss(logits, labels)
            return loss, logits
        else:
            return logits