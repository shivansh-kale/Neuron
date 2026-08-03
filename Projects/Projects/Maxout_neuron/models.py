import torch
import torch.nn as nn
class Maxout(nn.Module):
    def __init__(self,i,o,pieces=4):
        super().__init__(); self.o=o; self.p=pieces; self.fc=nn.Linear(i,o*pieces)
    def forward(self,x):
        x=self.fc(x).view(-1,self.o,self.p)
        return torch.max(x,2)[0]
class ReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)
class GELUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,10))
    def forward(self,x): return self.net(x)
class MaxoutNet(nn.Module):
    def __init__(self):
        super().__init__(); self.f=nn.Flatten(); self.m1=Maxout(784,256); self.m2=Maxout(256,128); self.out=nn.Linear(128,10)
    def forward(self,x): x=self.f(x); x=self.m1(x); x=self.m2(x); return self.out(x)
