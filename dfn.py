#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/20 

from random import random, randint
from cv2 import bilateralFilter
from cv2.ximgproc import guidedFilter

from utils import *


# ref: https://github.com/lllyasviel/AdverseCleaner/blob/main/clean.py
def adv_cleaner(img:npimg, b_iter:int=64, g_iter:int=4) -> npimg:
  y = img.copy()

  for _ in range(b_iter):
    y = bilateralFilter(y, 5, 8, 8)

  for _ in range(g_iter):
    y = guidedFilter(img, y, 4, 16)

  return y.clip(0, 255).astype(np.uint8)


# ref: https://huggingface.co/spaces/mf666/mist-fucker/blob/main/app.py
def dynamic_clean_adverse(
  img:npimg,
  diameter_min: int = 4,
  diameter_max: int = 6,
  sigma_color_min: float = 6.0,
  sigma_color_max: float = 10.0,
  sigma_space_min: float = 6.0,
  sigma_space_max: float = 10.0,
  radius_min: int = 3,
  radius_max: int = 6,
  eps_min: float = 16.0,
  eps_max: float = 24.0,
  b_iters: int = 64,
  g_iters: int = 8,
):
  y = img.copy()

  for _ in range(b_iters):
    diameter = randint(diameter_min, diameter_max)
    sigma_color = random() * (sigma_color_max - sigma_color_min) + sigma_color_min
    sigma_space = random() * (sigma_space_max - sigma_space_min) + sigma_space_min
    y = bilateralFilter(y, diameter, sigma_color, sigma_space)

  for _ in range(g_iters):
    radius = randint(radius_min, radius_max)
    eps = random() * (eps_max - eps_min) + eps_min
    y = guidedFilter(img, y, radius, eps)

  return y.clip(0, 255).astype(np.uint8)
