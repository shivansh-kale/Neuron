import matplotlib.pyplot as plt
def plot_history(history):
 plt.figure()
 for k,v in history.items(): plt.plot(v['loss'],label=k)
 plt.legend(); plt.savefig('plots/loss.png')
 plt.figure()
 for k,v in history.items(): plt.plot(v['acc'],label=k)
 plt.legend(); plt.savefig('plots/accuracy.png')
 plt.figure()
 plt.bar(history.keys(),[v['params'] for v in history.values()])
 plt.savefig('plots/parameter_count.png')
