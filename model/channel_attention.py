import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # self.fc = nn.Sequential(nn.Linear(in_planes, in_planes // ratio),
        #                         nn.ReLU(),
        #                         nn.Linear(in_planes // ratio, in_planes))
        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False))
        # self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)