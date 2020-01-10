import os
import pickle

import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.data.read_data import root_dir


class EmbeddingNet(nn.Module):
    def __init__(self, input_embedding_size):
        # TODO: residual network or text cnn or Attention
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
                                     nn.Conv1d(input_embedding_size, 64, 4),
                                     nn.BatchNorm1d(64),
                                     nn.PReLU(),
                                     nn.AvgPool1d(2, stride=2),
                                     nn.Conv1d(64, 32, 4),
                                     nn.BatchNorm1d(32),
                                     nn.PReLU(),
                                     nn.AvgPool1d(2, stride=2),
                                     nn.Conv1d(32, 16, 4),
                                     nn.BatchNorm1d(16),
                                     nn.PReLU(),
                                     # nn.AdaptiveMaxPool1d(1),
                                     # nn.Conv1d(16, 8, 5), nn.PReLU(),
                                     # nn.MaxPool1d(2, stride=2),
                                     )

        self.average_layer = nn.AdaptiveAvgPool1d(1)
        self.device = torch.device('cuda')

    def forward(self, x):
        x = x.to(self.device)
        output = self.convnet(x)
        # average = self.average_layer(x)
        #
        # output = torch.add(output, average)
        output = output.view(output.size()[0], -1)

        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        x = x.to(self.device)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output


class TextCNN(nn.Module):
    def __init__(self, input_embedding_size):
        # TODO: residual network or text cnn or Attention
        super(TextCNN, self).__init__()
        self.convnet = torch.nn.ModuleList()
        kernel_size = [2, 3, 4, 5]

        for _kernel_size in kernel_size:
            for dilated_rate in [1, 2, 3, 4]:
                self.convnet.append(nn.Conv1d(input_embedding_size, 16, _kernel_size, dilation=dilated_rate))

        # self.myparameters = nn.ParameterList(self.convnet)
        self.average_layer = nn.AdaptiveAvgPool1d(1)
        self.device = torch.device('cuda')

    def forward(self, x):
        x = x.to(self.device)
        cnns = [net(x).clamp(min=0) for net in self.convnet]
        # relus = [relu(value) for value in cnns]
        average = [self.average_layer(value) for value in cnns]
        output = torch.cat(average, dim=1)
        # average = self.average_layer(x)
        #
        # output = torch.add(output, average)
        output = output.view(output.size()[0], -1)

        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        x = x.to(self.device)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        return output


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.transform(x1)
        output2 = self.transform(x2)
        output3 = self.transform(x3)
        return output1, output2, output3

    def transform(self, x):
        # x = self.string_to_vec(x)
        output = self.embedding_net(x)
        # output.cuda()
        return output

    def get_embedding(self, x):
        return self.embedding_net.get_embedding(x)


def save_model(model, name):
    with open(os.path.join(root_dir(), "models", "model" + name + ".pk"), "wb") as f:
        pickle.dump(model, f)
