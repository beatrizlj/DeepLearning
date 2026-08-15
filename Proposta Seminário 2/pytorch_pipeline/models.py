import torch
import torch.nn as nn
from torchvision import models

def get_model(network_name, num_classes=8):
    """
    Factory function para instanciar o modelo escolhido.
    A última camada é adaptada para o `num_classes` (padrão: 8).
    """
    network_name = network_name.lower()
    
    if network_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Adaptar a fully connected
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif network_name == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif network_name == 'densenet':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        
    elif network_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif network_name == 'mobilenet':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    else:
        valid_networks = ['resnet', 'vgg', 'densenet', 'efficientnet', 'mobilenet']
        raise ValueError(f"Rede '{network_name}' inválida! Opções válidas: {valid_networks}")
        
    return model
