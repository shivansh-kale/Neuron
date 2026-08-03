import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Projectss.Mini_Project_Maxout_Extended.Maxout_neuron.models import MaxoutNet
device='cuda' if torch.cuda.is_available() else 'cpu'
model=MaxoutNet().to(device)
model.load_state_dict(torch.load('saved_models/maxout.pt',map_location=device))
model.eval()
loader=DataLoader(MNIST('data',train=False,download=True,transform=ToTensor()),256)
y_true=[]; y_pred=[]
with torch.no_grad():
  for x,y in loader:
    p=model(x.to(device)).argmax(1).cpu()
    y_true.extend(y.tolist()); y_pred.extend(p.tolist())
cm=confusion_matrix(y_true,y_pred)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig('plots/confusion_matrix.png')
