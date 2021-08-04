import torchvision 
import torch


def get_resnet50(output, pretrained = True, freeze = True):
    model = torchvision.models.resnet50(pretrained = pretrained)
    if freeze == True:
        for params in model.parameters():
            params.require_grads = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, output)

    return model


def get_vgg19(output, pretrained = True, freeze = True):
    model = torchvision.models.vgg19(pretrained = pretrained)
    if freeze == True:
        for param in model.parameters():
            param.require_grads = False 
    
    # TODO set last layer of vgg19 to output 
    
    return model 