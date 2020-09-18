from .backbone import resnet50,mobilenet_v2,IntermediateLayerGetter,DeepLabHeadV3Plus
import torch.nn as nn
import torch.nn.functional as F
import torch

class DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

def combine_resnet50_and_DeepLabV3(num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone =resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    inplanes = 2048
    low_level_planes = 256
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def combine_mobilenet_and_DeepLabV3(num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def deeplabv3plus_resnet50(num_classes=8, output_stride=8, pretrained_backbone=True):
    model = combine_resnet50_and_DeepLabV3(num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model

def deeplabv3plus_mobilenetv2(num_classes=8, output_stride=8, pretrained_backbone=True):
    model = combine_mobilenet_and_DeepLabV3(num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model


if __name__=="__main__":
    x=torch.rand((2,3,256,256)).cuda()
    model=deeplabv3plus_resnet50().cuda()
    y=model(x)
    print(y.shape)