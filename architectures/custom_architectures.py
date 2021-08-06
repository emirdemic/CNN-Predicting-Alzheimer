import torch 
from collections import OrderedDict
from utils.transformations import AddGaussianNoise


class SiameseCNN(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(SiameseCNN, self).__init__()

        self.input_dims = input_dims 
        self.output_dims = output_dims

        self.layers = OrderedDict()
        self.layers['layer1'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = input_dims, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 64)
        )
        self.layers['layer2'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layers['layer3'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 128),
            AddGaussianNoise()
        )
        self.layers['layer4'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layers['layer5'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features = 256),
        )
        self.layers['layer6'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layers['layer7'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            AddGaussianNoise()
        )
        self.layers['layer8'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layers['layer9'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layers['layer10'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layers['layer11'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )
        self.layers['layer12'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            AddGaussianNoise()
        )
        self.layers['layer13'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3),
            torch.nn.ReLU()
        )
        self.layers['layer14'] = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2)
        )


    def concatenate(self, path1, path2):
        return torch.cat((path1, path2))


    def flatten(self, x):
        return torch.flatten(x, start_dim = 0, end_dim = -1)


    def propagate_path(self, x):
        x = x
        for layer in self.layers:
            x = self.layers[layer](x)

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
        concatenated = self.fully_connected_layer(concatenated.size(0), 4096)
        concatenated = self.fully_connected_layer(4096, 4096)
        return self.fully_connected_layer(4096, self.output_dims)



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.layer = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        linear_combination = self.layer(x)
        return linear_combination