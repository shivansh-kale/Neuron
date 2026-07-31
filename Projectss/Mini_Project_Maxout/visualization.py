
import matplotlib.pyplot as plt
def plot(losses,accs,names):
    for l,n in zip(losses,names): plt.plot(l,label=n)
    plt.legend(); plt.title('Loss'); plt.show()
    plt.figure()
    for a,n in zip(accs,names): plt.plot(a,label=n)
    plt.legend(); plt.title('Accuracy'); plt.show()
