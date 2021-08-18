import torch
import torch.nn as nn
from torch.autograd import Function, Variable

CUDA = True if torch.cuda.is_available() else False


'''
MODELS
'''


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=1000):
        super(DeepCORAL, self).__init__()
        self.sharedNet = AlexNet()
        self.fc = nn.Linear(4096, num_classes)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.sharedNet(source)
        source = self.fc(source)

        target = self.sharedNet(target)
        target = self.fc(target)
        return source, target
