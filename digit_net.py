"""
@Project : AcouWrite
@File : digit_net.py
@Date : 2023/4/5 22:39
@Author : Qiuyang Zeng
@Software : PyCharm

"""
import datetime

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from constants.constants import DatasetLoadType

from net import Net, Conv2dWithBN, evaluate
from utils.dataset_utils import get_data_loader


class NewNet(nn.Module):
    def __init__(self, layers, in_channels, gru_input_size, gru_hidden_size, num_classes):
        super().__init__()
        self.layers = layers
        self.in_channels = in_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_input_size = gru_input_size
        self.num_classes = num_classes
        self.conv = nn.Sequential(
            nn.Sequential(
                Conv2dWithBN(1, in_channels, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                # nn.Dropout(0.1)
            ),
            self.make_conv_layers(layers),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2)
        )
        self.num_layers = 1
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, num_layers=self.num_layers)
        self.cls_new = nn.Sequential(
            nn.Linear(self.gru_hidden_size * self.num_layers, self.num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, seq_len):  # shape of x: (batch_size, sequence_length, features)
        x = x.transpose(0, 1)  # (sequence_length, batch_size, 1, H, W)
        conv_items = []
        for x_item in x:
            conv_item = self.conv(x_item).unsqueeze(0)
            conv_items.append(conv_item)  # shape of conv_item: (1, batch_size, features)
        x = torch.cat(conv_items, 0)  # shape of x: (sequence_length, batch_size, features)
        x = pack_padded_sequence(x, seq_len)
        _, h_n = self.gru(x)  # shape of x: (sequence_length, batch_size, gru_hidden_size)
        # shape of h_n: (1, batch_size, gru_hidden_size)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.reshape(h_n.shape[0], -1)
        output = self.cls_new(h_n.squeeze(0))  # shape of x: (batch_size, num_classes)
        return output

    def make_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for arg in arch:
            if type(arg) == int:
                layers += [
                    Conv2dWithBN(in_channels=in_channels, out_channels=arg,
                                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    # nn.Dropout(0.1),
                ]
                in_channels = arg
            elif arg == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)]

        return nn.Sequential(*layers)


def train():
    args = ["M", 32, "M", 64, "M", 64]
    model = Net(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
                num_classes=26).cuda()
    # Load the saved parameters
    model.load_state_dict(torch.load('model/letters_50epochs.pth'))
    new_model = NewNet(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
                       num_classes=10).cuda()
    # new_model = NewNet(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
    #                    num_classes=13).cuda()
    old_dict = model.state_dict()
    new_dict = new_model.state_dict()
    for k in old_dict.keys():
        if k in new_dict.keys():
            new_dict[k] = old_dict[k]
    new_model.load_state_dict(new_dict)
    loss_func = nn.CrossEntropyLoss()
    EPOCH = 20
    BATCH_SIZE = 16

    for param in new_model.conv.parameters():
        param.requires_grad = False

    data_path=[
        r"data/digits_1_0.pkl",
        r"data/digits_1_1.pkl",
        r"data/digits_1_2.pkl",
        r"data/digits_1_3.pkl",
        r"data/digits_1_4.pkl",
        r"data/digits_1_5.pkl",
        r"data/digits_1_6.pkl",
        r"data/digits_1_7.pkl",
        # r"data/strokes_1_0.pkl",
        # r"data/strokes_1_1.pkl",
        # r"data/strokes_1_2.pkl",
        # r"data/strokes_1_3.pkl",
        # r"data/strokes_1_4.pkl",
        # r"data/strokes_1_5.pkl",
        # r"data/strokes_1_6.pkl",
        # r"data/strokes_1_7.pkl",
        # r"data/strokes_1_8.pkl",
        # r"data/strokes_1_9.pkl",

    ]
    optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.001, weight_decay=0.01)
    train_loader, valid_loader, test_loader = get_data_loader(loader_type=DatasetLoadType.TrainValidAndTest,
                                                              batch_size=BATCH_SIZE,
                                                              data_path=data_path)
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    test_size = len(test_loader.dataset)
    batch_num = train_size // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_num * 5, gamma=0.75)
    for epoch in range(EPOCH):
        correct = 0
        epoch_loss = 0
        new_model.train()
        start_time = datetime.datetime.now()
        for step, (d_cir_x_batch, seq_len, y_batch) in enumerate(train_loader):
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda().long()
            output = new_model(d_cir_x_batch, seq_len)
            loss = loss_func(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 这里不再用optimizer.step()
            epoch_loss += loss.item()
            predicted = torch.argmax(output, 1)
            correct += sum(y_batch == predicted).item()
        end_time = datetime.datetime.now()
        print("[epoch {}] {}s {} acc {} loss {}".format
              (epoch + 1, (end_time - start_time).seconds, correct,
               round(correct * 1.0 / train_size, 4), round(epoch_loss, 2)))
        evaluate(valid_loader, new_model, "valid", valid_size)
        evaluate(test_loader, new_model, "test", test_size)
    # torch.save(new_model.state_dict(), 'model/digit.pth'.format(EPOCH))


def get_digit_net(path=r'model/digits_20epochs.pth'):
    args = ["M", 32, "M", 64, "M", 64]
    net = NewNet(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
                 num_classes=10).cuda()
    state_dict = torch.load(path)  # 2028 569
    net.load_state_dict(state_dict)
    net.eval()  # 禁用 dropout, 避免 BatchNormalization 重新计算均值和方差
    return net


def get_stroke_net(path=r'model/strokes_40epochs.pth'):
    args = ["M", 32, "M", 64, "M", 64]
    net = NewNet(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
                 num_classes=13).cuda()
    state_dict = torch.load(path)  # 2028 569
    net.load_state_dict(state_dict)
    net.eval()  # 禁用 dropout, 避免 BatchNormalization 重新计算均值和方差
    return net


if __name__ == '__main__':
    pass
    train()
    # C = acoustic_input_evaluation(get_digit_net(), data_path=[
    #     r"data/digits_1_0.pkl",
    #     r"data/digits_1_1.pkl",
    #     r"data/digits_1_2.pkl",
    #     r"data/digits_1_3.pkl",
    #     r"data/digits_1_4.pkl",
    #     r"data/digits_1_5.pkl",
    #     r"data/digits_1_6.pkl",
    #     r"data/digits_1_7.pkl",
    # ], class_num=10)
    # DIGIT = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # show_confusion_matrix_cn(C, x_tick=DIGIT, y_tick=DIGIT)
    # show_indicators_cn(C)
