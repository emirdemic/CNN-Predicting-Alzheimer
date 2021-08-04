'''
This module should be the main module for training and validating data.
This module should also send data to TensorBoard.
'''
import torch 
from torch.utils.tensorboard import SummaryWriter
from utils import visualizer
from architectures import pretrained_architectures


def loss_observer(loss, step, epoch):
    observed = f'Step: {step} ||| Epoch: {epoch} ||| LOSS: {loss}'

    return observed


def loss_warning(loss_history, threshold):
    if sorted(loss_history[-threshold: ]) == loss_history[-threshold: ]:
        warning = '\n||| LOSS HAS BEEN CONSTANTLY INCREASING FOR THE LAST {threshold} STEPS |||'
        return warning 
    else:
        return None


def check_validation_set(model):
    pass 


def train_nn(model, optimizer, criterion, learning_rate, dataloaders, epochs, device, scheduler = False):
    '''
    Function trains the model.
    
    Input:
        model
        optimizer
        criterion
        learning_rate
        dataloaders
        epochs
        device
        scheduler 
    '''
    
    for _ in range(epochs):

        for phase in ['train', 'evaluate']:
            
            if phase == 'train':
                model.train()
            elif phase == 'evaluate':
                model.evalu()
            
            for features, labels in dataloaders[phase]:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                output = model(features)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                if scheduler == True:
                    learning_rate.step()

    pass 


if __name__ == '__main__':
    print(None)