"""Matched low-capacity content rule for the existing operator prior.

Same six source features, mean/variance output units and downstream GP.
Mean pooling plus a single affine layer replaces learned nonlinear pooling.
J is constant in this matched J=2 experiment; omit its redundant intercept.
This is a mature linear-content control, not a new architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LinearContentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Linear(6,2)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([0.,-4.]))

    def forward(self,tokens):
        output=self.head(tokens.mean(0))
        return 20*output[:,0],10*F.softplus(output[:,1])
