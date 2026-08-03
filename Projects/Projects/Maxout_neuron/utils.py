import time
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Timer:
    def __enter__(self):
        self.s=time.time(); return self
    def __exit__(self,*a):
        self.elapsed=time.time()-self.s
