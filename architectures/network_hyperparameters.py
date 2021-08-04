import torch 

resnet50 = {
    'OPTIMIZER' : torch.optim.Adam, 
    'LEARNING_RATE' : torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CRITERION' : torch.nn.CrossEntropyLoss, 
    'EPOCHS' : 1000, 
}

vgg19 = {
    'OPTIMIZER' : torch.optim.Adam,
    'LEARNING_RATE' : torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CRITERION' : torch.nn.CrossEntropyLoss,
    'EPOCHS' : 1000
}