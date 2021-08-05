'''
This module should be the main module for training and validating the model.
This module should also send data to TensorBoard.
'''
import torch 
from torch.utils.tensorboard import SummaryWriter
from utils import visualizer
from architectures import pretrained_architectures, network_hyperparameters
from colorama import Fore
import DataLoader
import copy


def loss_observer(loss : float, step : int, epoch : int) -> str:
    observed = f'Step: {step} ||| Epoch: {epoch} ||| LOSS: {loss}'

    return observed


def loss_warning(loss_history : list, threshold : int) -> str:
    '''
    Method checks whether the last n losses are constantly increasing.
    If so, returns a string with the specified warning. Otherwise, method returns None.
    '''
    if len(loss_history) > threshold:
        if sorted(loss_history[-threshold: ]) == loss_history[-threshold: ]:
            warning = Fore.RED + f'\n||| LOSS HAS BEEN INCREASING FOR THE LAST {threshold} STEPS |||\n'
            return warning 
        else:
            return None
    else:
        return None


def train_nn(model, dataloaders, optimizer, criterion, learning_rate, epochs, device, scheduler = False):
    '''
    Method trains the model.
    '''
    best_model = copy.deepcopy(model.state_dict())
    summary_writer = SummaryWriter()
    step = 0 
    train_epoch_loss = []
    val_epoch_loss = []

    for epoch in range(epochs):

        for phase in ['train', 'evaluate']:
            
            if phase == 'train':
                model.train()
            elif phase == 'evaluate':
                model.eval()
            
            batch_loss = 0

            for features, labels in dataloaders[phase]:
                features = features.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        output = model(features)
                        loss = criterion(output, labels)
                        loss.backward()
                        optimizer.step()
                        # Getting the total loss for the batch
                        batch_loss += loss.item() * features.size(0)

                        if scheduler == True:
                            learning_rate.step()
                        
                        summary_writer.add_scalar('Train Batch Loss', loss.item(0), step)
                        print(loss_observer(loss = loss.item(), step = step, epoch = epoch))

                    elif phase == 'evaluate':
                        output = model(features)
                        loss = criterion(output, labels)
                        # Make predictions
                        predictions = torch.nn.functional.softmax(output)
                        predicted_class = torch.argmax(predictions)
                        batch_loss += loss.item() * features.size(0)
                        summary_writer.add_scalar('Validation Batch Loss', loss.item(0), step)
                        
                step += 1
            
            if phase == 'train':
                train_epoch_loss.append(batch_loss)
            elif phase == 'evaluate':
                val_epoch_loss.append(batch_loss)
        
        warning = loss_warning(loss_history = train_epoch_loss, threshold = 7)
        if warning:
            print(warning)
            
    summary_writer.close()
    
    return best_model, model


if __name__ == '__main__':

    dataloaders = DataLoader.get_Dataset(folder_name = '../new_data', bs = 64)
    model = pretrained_architectures.get_resnet(
        output = 4, 
        resnet_version = 34, 
        pretrained = True, 
        freeze = True
        ).to(network_hyperparameters['DEVICE'])

    OPTIMIZER = network_hyperparameters['OPTIMIZER'](filter(lambda p: p.requires_grad, model.parameters()))
    LEARNING_RATE = network_hyperparameters['LEARNING_RATE']
    CRITERION = network_hyperparameters['CRITERION']()
    EPOCHS = network_hyperparameters['EPOCHS']
    DEVICE = network_hyperparameters['DEVICE']
    SCHEDULER = network_hyperparameters['SCHEDULER']

    best_model, model = train_nn(
        model = model, 
        dataloaders = dataloaders,
        optimizer = OPTIMIZER, 
        criterion = CRITERION,
        learning_rate = LEARNING_RATE,
        epochs = EPOCHS,
        device = DEVICE,
        scheduler = SCHEDULER
    )

    MODEL_NAME = 'RESNET34_TRIAL_1'
    torch.save(obj = best_model.state_dict(), f = f'../models_state/{MODEL_NAME}')