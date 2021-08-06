import torch 

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean = 0, std = 1):
        super(AddGaussianNoise, self).__init__()
        self.mean = 0
        self.std = 1 
    

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


    def __repl__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
