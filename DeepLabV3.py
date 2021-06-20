import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models._utils import IntermediateLayerGetter
from timm import create_model


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class HRNetBackbone(nn.Module):
    def __init__(self, model, scale=2):
        super(HRNetBackbone, self).__init__(); self.model = model; self.scale = scale
    def forward(self, x):
        return {'out':self.model(x)[self.scale]}

def get_backbone(backbone_type, extra_input_channel=0):
    if 'resnet' in backbone_type:
        backbone = create_model(backbone_type, pretrained=True)
        backbone = IntermediateLayerGetter(backbone, return_layers={'layer4':'out'})
        if extra_input_channel > 0:
            w = backbone.conv1.weight
            backbone.conv1 = nn.Conv2d(3+extra_input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            init.xavier_uniform_(backbone.conv1.weight)
            z = backbone.conv1.weight[:,-1:,:,:]
            w = nn.Parameter(torch.cat([w,z],dim=1)).to(w.device)
            backbone.conv1.weight = w
        head_dim = 2048
    elif 'resnest' in backbone_type:
        backbone = create_model(backbone_type, pretrained=True)
        backbone = IntermediateLayerGetter(backbone, return_layers={'layer4':'out'})
        if extra_input_channel > 0:
            w = backbone.conv1[0].weight
            backbone.conv1[0] = nn.Conv2d(3+extra_input_channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            init.xavier_uniform_(backbone.conv1[0].weight)
            z = backbone.conv1[0].weight[:,-1:,:,:]
            w = nn.Parameter(torch.cat([w,z],dim=1)).to(w.device)
            backbone.conv1[0].weight = w
        head_dim = 2048
    elif 'dense' in backbone_type:
        backbone = create_model(backbone_type, pretrained=True)
        backbone = IntermediateLayerGetter(backbone.features, return_layers={'denseblock4':'out'})
        if extra_input_channel > 0:
            w = backbone.conv0.weight
            backbone.conv0 = nn.Conv2d(3+extra_input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            init.xavier_uniform_(backbone.conv0.weight)
            z = backbone.conv0.weight[:,-1:,:,:]
            w = nn.Parameter(torch.cat([w,z],dim=1)).to(w.device)
            backbone.conv0.weight = w
        head_dim = 1024
    elif 'hrnet' in backbone_type:
        backbone = create_model(backbone_type, pretrained=True, features_only=True)
        if extra_input_channel > 0:
            w = backbone.conv1.weight
            backbone.conv1 = nn.Conv2d(3+extra_input_channel, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            init.xavier_uniform_(backbone.conv1.weight)
            z = backbone.conv1.weight[:,-1:,:,:]
            w = nn.Parameter(torch.cat([w,z],dim=1)).to(w.device)
            backbone.conv1.weight = w
        backbone = HRNetBackbone(backbone, scale=2)
        head_dim = 256
    else:
        raise NotImplementedError
    return backbone, head_dim

def deeplabv3_backbone(backbone_type, num_classes = 20, extra_input_channel=1):
    backbone, head_dim = get_backbone(backbone_type, extra_input_channel)
    classifier = DeepLabHead(head_dim, num_classes)
    net = DeepLabV3(backbone, classifier, None)
    return net

if __name__=="__main__":
    net = deeplabv3_backbone(
        backbone_type = "resnet50",
        extra_input_channel = 0,
    )