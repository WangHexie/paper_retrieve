import os
import pickle

import torch
import torch.nn as nn

from src.data.read_data import root_dir


class EmbeddingNet(nn.Module):
    def __init__(self, input_embedding_size):
        # TODO: residual network or text cnn or Attention
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(input_embedding_size, 64, 5), nn.PReLU(),
                                     nn.AvgPool1d(2, stride=2),
                                     nn.Conv1d(64, 32, 5), nn.PReLU(),
                                     nn.AvgPool1d(2, stride=2),
                                     nn.Conv1d(32, 16, 5), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     # nn.Conv1d(16, 8, 5), nn.PReLU(),
                                     # nn.MaxPool1d(2, stride=2),
                                     )

        #
        # self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.PReLU(),
        #                         nn.Linear(256, 2)
        #                         )
        self.device = torch.device('cuda')

    def forward(self, x):
        x = x.to(self.device)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)

        # output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


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
        output.cuda()
        return output

    def get_embedding(self, x):
        return self.embedding_net(x)


def save_model(model, name):
    with open(os.path.join(root_dir(), "models", "model" + name + ".pk"), "wb") as f:
        pickle.dump(model, f)
