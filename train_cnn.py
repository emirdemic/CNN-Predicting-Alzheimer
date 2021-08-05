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
    best_model = None 
    summary_writer = SummaryWriter()
    train_step = 0
    val_step = 0 
    train_epoch_loss = []
    val_epoch_loss = []

    best_accuracy = 0

    for epoch in range(epochs):    
        current_accuracy = 0

        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            elif phase == 'val':
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
                        
                        summary_writer.add_scalar('Train Batch Loss', loss, train_step)
                        print(loss_observer(loss = loss.item(), step = train_step, epoch = epoch))
                        train_step += 1

                    elif phase == 'val':
                        output = model(features)
                        loss = criterion(output, labels)
                        # Make predictions
                        predictions = torch.nn.functional.softmax(output, dim = -1)
                        predicted_class = torch.argmax(predictions)
                        current_accuracy += torch.sum(labels == predicted_class)
                        batch_loss += loss.item() * features.size(0)
                        summary_writer.add_scalar('Validation Batch Loss', loss, val_step)
                        val_step += 1
            
            if phase == 'train':
                train_epoch_loss.append(batch_loss)
            elif phase == 'val':
                val_epoch_loss.append(batch_loss)
            

        current_accuracy = current_accuracy / len(dataloaders['val'].dataset)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = copy.deepcopy(model.state_dict())


        warning = loss_warning(loss_history = train_epoch_loss, threshold = 7)
        if warning:
            print(warning)
            
    summary_writer.close()
    
    return best_model, model


if __name__ == '__main__':

    dataloaders = DataLoader.get_Dataset(folder_name = 'new_data', bs = 16)
    model = pretrained_architectures.get_resnet(
        output = 4, 
        resnet_version = 34, 
        pretrained = True, 
        freeze = True
        ).to(network_hyperparameters.resnet['DEVICE'])

    OPTIMIZER = network_hyperparameters.resnet['OPTIMIZER'](filter(lambda p: p.requires_grad, model.parameters()))
    LEARNING_RATE = network_hyperparameters.resnet['LEARNING_RATE']
    CRITERION = network_hyperparameters.resnet['CRITERION']()
    EPOCHS = network_hyperparameters.resnet['EPOCHS']
    DEVICE = network_hyperparameters.resnet['DEVICE']
    SCHEDULER = network_hyperparameters.resnet['SCHEDULER']

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
    torch.save(obj = best_model.state_dict(), f = f'models_state/{MODEL_NAME}')