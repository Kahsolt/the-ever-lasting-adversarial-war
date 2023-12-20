#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/20 

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
