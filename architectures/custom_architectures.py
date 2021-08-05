import torch 


class MyModel(torch.nn.Module):
    def __init__(self):
        pass 


    def forward(self, x):
        pass 


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.layer = torch.nn.Linear(input_dims, output_dims)

    def forward(self, x):
        linear_combination = self.layer(x)
        return linear_combination