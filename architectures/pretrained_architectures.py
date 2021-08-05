import torchvision 
import torch


def get_resnet(output, resnet_version, pretrained = True, freeze = True):
    if resnet_version == 18:
        model = torchvision.models.resnet18(pretrained = pretrained)
    elif resnet_version == 34: 
        model = torchvision.models.resnet34(pretrained = pretrained)
    elif resnet_version == 50:
        model = torchvision.models.resnet50(pretrained = pretrained)
    elif resnet_version == 101:
        model = torchvision.models.resnet101(pretrained = pretrained)
    elif resnet_version == 152:
        model = torchvision.models.resnet152(pretrained = pretrained)
    else:
        raise KeyError('Model not available')

    if freeze == True:
        for param in model.parameters():
            param.require_grads = False

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, output)

    return model


def get_vgg19(output, pretrained = True, freeze = True):
    model = torchvision.models.vgg19(pretrained = pretrained)
    if freeze == True:
        for param in model.parameters():
            param.require_grads = False 
    
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, output)
    
    return model 


def get_inception3(output, pretrained = True, freeze = True):
    model = torchvision.models.inception_v3(pretrained = True)
    if freeze == True:
        for param in model.parameters():
            param.require_grads = False 
    
    model.fc = torch.nn.Linear(model.fc.in_features, output)

    return model