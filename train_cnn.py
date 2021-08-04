'''
This module should be the main module for training and validating data.
This module should also send data to TensorBoard.
'''
import torch 
from torch.utils.tensorboard import SummaryWriter
from utils import visualizer
from architectures import pretrained_architectures
from colorama import Fore


def loss_observer(loss : float, step : int, epoch : int) -> str:
    observed = f'Step: {step} ||| Epoch: {epoch} ||| LOSS: {loss}'

    return observed


def loss_warning(loss_history : list, threshold : int) -> str:
    '''
    Method checks whether the last n losses are constantly increasing.
    If so, returns a string with the specified warning. Otherwise, method returns None.
    '''
    if sorted(loss_history[-threshold: ]) == loss_history[-threshold: ]:
        warning = Fore.RED + '\n||| LOSS HAS BEEN CONSTANTLY INCREASING FOR THE LAST {threshold} STEPS |||\n'
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
    summary_writer = SummaryWriter()
    step = 0 
    train_epoch_loss = 0
    train_step_loss = 0
    val_epoch_loss = 0
    val_step_loss = 0

    for _ in range(epochs):

        for phase in ['train', 'evaluate']:
            
            if phase == 'train':
                model.train()
            elif phase == 'evaluate':
                model.eval()
            
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

                step += 1
                #step_loss += loss 
                #summary_writer.add_scalars('Train Loss', step_loss, step)

        train_epoch_loss  = train_step_loss - train_epoch_loss
        val_epoch_loss = val_step_loss - val_epoch_loss

    return None


if __name__ == '__main__':
    pass 