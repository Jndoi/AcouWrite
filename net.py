"""
@Project :acou-input-torch
@File ：single_net.py
@Date ： 2022/7/15 20:18
@Author ： Qiuyang Zeng
@Software ：PyCharm
https://www.jianshu.com/p/0a889cc03156 迁移学习 冻结参数
"""
import torch
import datetime
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.wav2pickle_utils import DataItem
from utils.dataset_utils import get_data_loader
from constants.constants import DatasetLoadType

BATCH_SIZE = 16
EPOCH = 50
LR = 1e-3


class Conv2dWithBN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dWithBN, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.nonlinear(x)


class Net(nn.Module):
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
            nn.Flatten(),
            nn.Dropout(0.2)
        )
        self.num_layers = 1
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, num_layers=self.num_layers)
        self.cls = nn.Sequential(
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
        output = self.cls(h_n.squeeze(0))  # shape of x: (batch_size, num_classes)
        return output

    def make_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels
        for arg in arch:
            if type(arg) == int:
                layers += [Conv2dWithBN(in_channels=in_channels, out_channels=arg,
                                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
                in_channels = arg
            elif arg == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)]
            elif arg == "GAP":
                layers += [nn.AdaptiveAvgPool2d(1)]
        return nn.Sequential(*layers)


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of params: %.2fM' % (total / 1e6))


def evaluate(data_loader, net, evaluation_type, total):
    correct = 0
    net.eval()
    with torch.no_grad():
        for step, (d_cir_x_batch, seq_len, y_batch) in enumerate(data_loader):
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda()
            output = net(d_cir_x_batch, seq_len)
            predicted = torch.argmax(output, 1)
            correct += sum(y_batch == predicted).item()
        print("{}: {}/{}, acc {}".format(evaluation_type, correct, total, round(correct * 1.0 / total, 6)))


def train():
    args = ["M", 32, "M", 64, "M", 64, "GAP"]
    net = Net(layers=args, in_channels=32, gru_input_size=64, gru_hidden_size=64,
              num_classes=26).cuda()
    print_model_parm_nums(net)
    data_path = [
        r"data/dataset.pkl",
        r"data/dataset_5cm.pkl",
        r"data/dataset_10cm.pkl",
        r"data/dataset_five_fourth.pkl",
        r"data/dataset_four_fifth.pkl",
        r"data/dataset_single.pkl",
        r"data/dataset_5cm_single.pkl",
        r"data/dataset_10cm_single.pkl",
        r"data/dataset_five_fourth_single.pkl",
        r"data/dataset_four_fifth_single.pkl",
    ]
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    # state_dict = torch.load('model/XXX.pth')
    # net.load_state_dict(state_dict)
    train_loader, valid_loader, test_loader = get_data_loader(
        loader_type=DatasetLoadType.UniformTrainValidAndTest,
        batch_size=BATCH_SIZE,
        data_path=data_path)
    train_size = len(train_loader.dataset)
    valid_size = len(valid_loader.dataset)
    test_size = len(test_loader.dataset)
    print("Train on {} samples, validate on {} samples, test on {} samples".
          format(train_size, valid_size, test_size))
    batch_num = train_size // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=batch_num * 5, gamma=0.75)

    for epoch in range(EPOCH):
        net.train()
        correct = 0
        epoch_loss = 0
        start_time = datetime.datetime.now()
        for step, (d_cir_x_batch, seq_len, y_batch) in enumerate(train_loader):
            # d_cir_x_batch = d_cir_x_batch.float() / 255
            d_cir_x_batch = d_cir_x_batch.cuda()
            y_batch = y_batch.cuda().long()
            output = net(d_cir_x_batch, seq_len)
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
        if (epoch + 1) % 5 == 0:
            print("train loss: {} train acc: {}".format
                  (round(epoch_loss, 2), round(correct * 1.0 / train_size, 4)))
            evaluate(valid_loader, net, "valid", valid_size)
            evaluate(test_loader, net, "test", test_size)
            torch.save(net.state_dict(), 'model/{} letters_{}epochs.pth'.format(datetime.date.today(), epoch + 1))


if __name__ == '__main__':
    pass
    train()
