#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import json
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Any

import torch
from torch import Tensor
import numpy as np
from numpy import ndarray

BASE_PATH = Path(__file__).parent
REPO_PATH = BASE_PATH / 'repo'
IMG_PATH = BASE_PATH / 'img'
INPUT_FILE  = IMG_PATH / 'input.png'
OUTPUT_FILE = IMG_PATH / 'output.png'
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

npimg = ndarray


def load_im(fp:Path) -> npimg:
  img = Image.open(fp).convert('RGB')
  return pil_to_npimg(img)

def save_im(im:npimg, fp:Path):
  npimg_to_pil(im).save(fp)

def pil_to_npimg(img:PILImage) -> npimg:
  return np.asarray(img, dtype=np.uint8)

def npimg_to_pil(im:npimg) -> PILImage:
  if im.dtype == np.float32:
    assert 0 <= im.min() and im.max() <= 1.0
    im = im_f32_to_u8(im)
  else: assert im.dtype == np.uint8
  return Image.fromarray(im)

def im_u8_to_f32(im:npimg) -> npimg:
  return np.asarray(im / 255, dtype=np.float32)

def im_f32_to_u8(im:npimg) -> npimg:
  return np.asarray(im * 255, dtype=np.uint8)

def im_valid(im:npimg) -> npimg:
  assert im.dtype == np.float32
  return im_u8_to_f32(im_f32_to_u8(im.clip(0.0, 1.0)))

def to_gray(im:npimg) -> npimg:
  return pil_to_npimg(npimg_to_pil(im).convert('L'))

def rgb2bgr(im:npimg) -> npimg:
  return im[:, :, ::-1]

bgr2rgb = rgb2bgr

def color_fix(im:npimg, im_ref:npimg) -> npimg:
  assert im.dtype == im_ref.dtype == np.float32
  tgt_avg = np.mean(im_ref, axis=-1, keepdims=True)
  tgt_std = np.std(im_ref, axis=-1, keepdims=True)
  src_avg = np.mean(im, axis=-1, keepdims=True)
  src_std = np.std(im, axis=-1, keepdims=True)
  im_norm = (im - src_avg) / src_std
  im_shift = im_norm * tgt_std + tgt_avg
  return im_valid(im_shift)


def load_json(fp:Path, defval=dict) -> Any:
  if not fp.exists(): return defval()
  with open(fp, 'r', encoding='utf-8') as fh:
    return json.load(fh)

def save_json(fp:Path, data:Any):
  with open(fp, 'w', encoding='utf-8') as fh:
    return json.dump(data, fh, indent=2, ensure_ascii=False)


def seq_fn(step:int) -> str:
  return f'{step:08d}.png'
