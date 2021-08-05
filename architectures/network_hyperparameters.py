import torch 

resnet = {
    'OPTIMIZER' : torch.optim.Adam, 
    'LEARNING_RATE' : 0.0001,
    'CRITERION' : torch.nn.CrossEntropyLoss, 
    'EPOCHS' : 1000, 
    'SCHEDULER' : False,
    'DEVICE' : 'cuda' if torch.cuda.is_available() else 'cpu'
    }

vgg19 = {
    'OPTIMIZER' : torch.optim.Adam,
    'LEARNING_RATE' : torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CRITERION' : torch.nn.CrossEntropyLoss,
    'EPOCHS' : 1000
}