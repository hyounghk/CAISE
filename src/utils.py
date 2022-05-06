import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms

class LinearAct(nn.Module):
    def __init__(self, fan_in, fan_out, act='linear', bias=True):
        super().__init__()
        self.fc = nn.Linear(fan_in, fan_out, bias)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'linear':
            self.act = None
        else:
            assert False

    def forward(self, x):
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        return x



def cut_inst_with_leng(inst, leng):

    return inst[:, :max(leng)]



