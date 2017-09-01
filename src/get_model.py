import torch
import torch.nn as nn
import torch.nn.init as init


# a building block of a fully-connected feedforward neural network
class block(nn.Module):
    def __init__(self, in_features, out_features, drop_rate):
        super(block, self).__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Network(nn.Module):

    def __init__(self, input_dim, num_classes, architecture, dropout):

        super(Network, self).__init__()

        architecture.insert(0, input_dim)
        self.features = nn.Sequential(*[
            block(in_features, out_features, drop_rate)
            for in_features, out_features, drop_rate in
            zip(architecture, architecture[1:], dropout)
        ])

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(architecture[-1], num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant(m.weight, 1.0)
                init.constant(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
