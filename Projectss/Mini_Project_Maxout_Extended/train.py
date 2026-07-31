import os,time,torch
import torch.nn as nn, torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from models import ReLUNet,GELUNet,MaxoutNet
from utils import count_parameters

device='cuda' if torch.cuda.is_available() else 'cpu'
train=DataLoader(MNIST('data',train=True,download=True,transform=ToTensor()),128,shuffle=True)
test=DataLoader(MNIST('data',train=False,download=True,transform=ToTensor()),256)

models={'relu':ReLUNet(),'gelu':GELUNet(),'maxout':MaxoutNet()}
history={}
for name,m in models.items():
 m.to(device); opt=optim.Adam(m.parameters(),1e-3); lossfn=nn.CrossEntropyLoss()
 tl=[]; ta=[]; st=time.time()
 for e in range(5):
  m.train(); run=0
  for x,y in train:
   x,y=x.to(device),y.to(device); opt.zero_grad(); out=m(x); loss=lossfn(out,y); loss.backward(); opt.step(); run+=loss.item()
  tl.append(run/len(train)); m.eval(); c=t=0
  with torch.no_grad():
   for x,y in test:
    x,y=x.to(device),y.to(device); p=m(x).argmax(1); c+=(p==y).sum().item(); t+=y.size(0)
  ta.append(c/t)
 torch.save(m.state_dict(),f'saved_models/{name}.pt')
 history[name]={'loss':tl,'acc':ta,'params':count_parameters(m),'time':time.time()-st}
print(history)
