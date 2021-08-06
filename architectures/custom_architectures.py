import torch 
from collections import OrderedDict
from utils.transformations import AddGaussianNoise


class SiameseCNN(torch.nn.Module):
    def __init__(self, input_dims, output_dims, device):
        super(SiameseCNN, self).__init__()

        self.input_dims = input_dims 
        self.output_dims = output_dims
        self.device = device

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = input_dims, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 64)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 128),
            AddGaussianNoise(device = self.device)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 256),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            AddGaussianNoise(device = self.device)
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layer10 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layer11 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layer12 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU(),
            AddGaussianNoise(device = self.device)
        )
        self.layer13 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layer14 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )


    def concatenate(self, path1, path2):
        return path1 + path2


    def flatten(self, x):
        return torch.flatten(x, start_dim = 1)


    def propagate_path(self, x): # TODO apply this as a recursive function
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)

        return x


    def fully_connected_layer(self, input_size, output_size, x):
        relu = torch.nn.ReLU()
        combination = torch.nn.Linear(in_features = input_size, out_features = output_size)
        return relu(combination(x))


    def forward(self, x):
        path1 = self.propagate_path(x)
        path2 = self.propagate_path(x)
        path1 = self.flatten(path1)
        path2 = self.flatten(path2)
        concatenated = self.concatenate(path1, path2)
        concatenated = self.fully_connected_layer(concatenated.size(1), 4096, concatenated)
        concatenated = self.fully_connected_layer(4096, 4096, concatenated)
        return self.fully_connected_layer(4096, self.output_dims, concatenated)



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.layer = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        linear_combination = self.layer(x)
        return linear_combination