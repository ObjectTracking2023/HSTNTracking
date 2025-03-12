import torch
from torch import nn
import torch.nn.functional as F

class updated_condition(nn.Module):
    def __init__(self):
        super(updated_condition, self).__init__()

    def forward(self,  score):


        def condition_two(a):
            if a > 0.75:
                return True
            else:
                return False

        if (condition_two(score)):
            return True
        else:
            return False
