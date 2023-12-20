#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/20 

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module

from tqdm import tqdm


@torch.enable_grad()
def pgd(model:Module, images:Tensor, labels:Tensor, eps=8/255, alpha=1/255, steps=20, **kwargs):
  ''' here `labels` are actually intermediate feature-maps peeked from classic pretrained CNNs like resnet50 '''

  normalizer = kwargs.get('normalizer', lambda _: _)

  images = images.clone().detach()
  labels = labels.clone().detach()

  adv_images = images.clone().detach()
  adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
  adv_images = torch.clamp(adv_images, min=0, max=1).detach()

  pbar = tqdm(range(steps))
  for _ in pbar:
    adv_images.requires_grad = True
    outputs = model(normalizer(adv_images))

    loss = F.l1_loss(outputs, labels, reduction='none')
    grad = torch.autograd.grad(loss, adv_images, grad_outputs=loss)[0]

    v_loss = loss.mean().item()
    pbar.set_description_str(f'loss: {v_loss:.5f}')
    if v_loss == 0.0: break   # return early

    with torch.no_grad():
      adv_images = adv_images.detach() + alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-eps, max=eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

  return adv_images
