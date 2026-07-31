
import time, torch
import torch.nn as nn, torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from models import ReLUNet,GELUNet,MaxoutNet
from utils import count_parameters

device='cuda' if torch.cuda.is_available() else 'cpu'
train=DataLoader(MNIST('data',train=True,download=True,transform=transforms.ToTensor()),batch_size=128,shuffle=True)
test=DataLoader(MNIST('data',train=False,download=True,transform=transforms.ToTensor()),batch_size=256)

models={'ReLU':ReLUNet(),'GELU':GELUNet(),'Maxout':MaxoutNet()}

for name,m in models.items():
    m.to(device)
    opt=optim.Adam(m.parameters(),lr=1e-3)
    lossfn=nn.CrossEntropyLoss()
    start=time.time()
    for e in range(10):
        m.train()
        for x,y in train:
            x,y=x.to(device),y.to(device)
            opt.zero_grad()
            loss=lossfn(m(x),y)
            loss.backward()
            opt.step()
        m.eval()
        c=t=0
        with torch.no_grad():
            for x,y in test:
                x,y=x.to(device),y.to(device)
                p=m(x).argmax(1)
                c+=(p==y).sum().item()
                t+=y.size(0)
        print(name,e+1,c/t*100)
    print('Params',count_parameters(m))
    print('Time',time.time()-start)
