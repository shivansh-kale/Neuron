import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Maxout(nn.Module):
    def __init__(self,in_f,out_f,pieces=4):
        super().__init__()
        self.out_f=out_f; self.pieces=pieces
        self.fc=nn.Linear(in_f,out_f*pieces)
    def forward(self,x):
        x=self.fc(x).view(-1,self.out_f,self.pieces)
        return x.max(dim=2).values

class Multiplicative(nn.Module):
    def __init__(self,in_f,out_f):
        super().__init__()
        self.a=nn.Linear(in_f,out_f)
        self.b=nn.Linear(in_f,out_f)
    def forward(self,x):
        return self.a(x)*self.b(x)

class MLP(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind=="relu":
            self.net=nn.Sequential(nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10))
        elif kind=="gelu":
            self.net=nn.Sequential(nn.Linear(784,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,10))
        elif kind=="maxout":
            self.net=nn.Sequential(Maxout(784,256),Maxout(256,128),nn.Linear(128,10))
        else:
            self.net=nn.Sequential(Multiplicative(784,256),Multiplicative(256,128),nn.Linear(128,10))
    def forward(self,x): return self.net(x.view(x.size(0),-1))

def count_params(m): return sum(p.numel() for p in m.parameters())

device="cuda" if torch.cuda.is_available() else "cpu"
tr=DataLoader(datasets.MNIST(".",train=True,download=True,transform=transforms.ToTensor()),64,shuffle=True)
te=DataLoader(datasets.MNIST(".",train=False,transform=transforms.ToTensor()),256)

for kind in ["relu","gelu","maxout","multiplicative"]:
    model=MLP(kind).to(device)
    opt=optim.Adam(model.parameters(),1e-3)
    lossfn=nn.CrossEntropyLoss()
    t=time.time()
    for epoch in range(5):
        model.train()
        for x,y in tr:
            x,y=x.to(device),y.to(device)
            opt.zero_grad();loss=lossfn(model(x),y);loss.backward();opt.step()
    elapsed=time.time()-t
    correct=0;total=0
    model.eval()
    with torch.no_grad():
        for x,y in te:
            p=model(x.to(device)).argmax(1).cpu()
            correct+=(p==y).sum().item(); total+=len(y)
    print(kind,correct/total,count_params(model),elapsed)
