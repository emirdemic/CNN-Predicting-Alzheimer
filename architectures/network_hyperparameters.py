import torch 

resnet = {
    'OPTIMIZER' : torch.optim.Adam, 
    'LEARNING_RATE' : 0.00002/3,
    'CRITERION' : torch.nn.CrossEntropyLoss, 
    'EPOCHS' : 60, 
    'BATCH_SIZE' : 64,
    'WEIGHT_DECAY' :0.009*4,
    'DROPOUT_PROB' : 0.5,
    'DROPOUT_LOC' : None,
    'SCHEDULER' : False,
    'DEVICE' : 'cuda' if torch.cuda.is_available() else 'cpu'
    }

vgg19 = {
    'OPTIMIZER' : torch.optim.Adam,
    'LEARNING_RATE' : torch.optim.lr_scheduler.ReduceLROnPlateau,
    'CRITERION' : torch.nn.CrossEntropyLoss,
    'EPOCHS' : 1000
}