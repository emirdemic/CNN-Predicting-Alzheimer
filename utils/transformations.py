import torch 

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean = 0, std = 1, device = 'cuda'):
        super(AddGaussianNoise, self).__init__()
        self.mean = 0
        self.std = 1 
        self.device = device

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device = self.device) * self.std + self.mean


    def __repl__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
