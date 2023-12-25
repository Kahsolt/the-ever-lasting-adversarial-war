#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import warnings ; warnings.simplefilter('ignore')

import torch
from torch import Tensor
from torch.nn import Module
import torchvision.models as M
from torchvision.models.resnet import ResNet
from torchvision.models.shufflenetv2 import ShuffleNetV2
from torchvision.models.vision_transformer import VisionTransformer

MODELS = [
  'resnet18',
  'resnet34',
  'resnet50',
  'resnet101',
  'resnet152',
  'resnext50_32x4d',
  'resnext101_32x8d',
  'resnext101_64x4d',
  'wide_resnet50_2',
  'wide_resnet101_2',

  'convnext_tiny',
  'convnext_small',
  'convnext_base',
  'convnext_large',

  'densenet121',
  'densenet161',
  'densenet169',
  'densenet201',

  'vit_b_16',
  'vit_b_32',
  'vit_l_16',
  'vit_l_32',
  'vit_h_14',

  'swin_t',
  'swin_s',
  'swin_b',
  'swin_v2_t',
  'swin_v2_s',
  'swin_v2_b',

  'maxvit_t',

  'inception_v3',

  'squeezenet1_0',
  'squeezenet1_1',

  'mobilenet_v2',
  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'shufflenet_v2_x0_5',
  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',
]


def hijack_resnet_forward(self:ResNet, x:Tensor) -> Tensor:
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)
  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)    # stop here!
  return x


def hijack_shufflenet_v2_forward(self:ShuffleNetV2, x:Tensor) -> Tensor:
  x = self.conv1(x)
  x = self.maxpool(x)
  x = self.stage2(x)
  x = self.stage3(x)
  x = self.stage4(x)
  x = self.conv5(x)   # stop here!
  return x


def hijack_vit_forward(self:VisionTransformer, x:Tensor) -> Tensor:
  x = self._process_input(x)
  n = x.shape[0]
  batch_class_token = self.class_token.expand(n, -1, -1)
  x = torch.cat([batch_class_token, x], dim=1)
  x = self.encoder(x)   # stop here!
  return x


def get_model(name:str, hijack:bool=True) -> Module:
  model: Module = getattr(M, name)(pretrained=True)
  if not hijack: return model

  if name.startswith('resnet'):
    model.forward = lambda x: hijack_resnet_forward(model, x)
  elif name.startswith('shufflenet_v2'):
    model.forward = lambda x: hijack_shufflenet_v2_forward(model, x)
  elif name.startswith('vit'):
    model.forward = lambda x: hijack_vit_forward(model, x)
  else:
    raise NotImplementedError('>> not supported, you can add hijack_forward by yourself')

  return model
