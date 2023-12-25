#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/25 

from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tqdm import tqdm

from models import get_model, MODELS
from run import ndarray_to_tensor
from utils import *


def vis(args):
  log_dp = Path(args.log_path)
  assert log_dp.is_dir()

  load_img_tensor = lambda fp: ndarray_to_tensor(load_im(fp)).to(device)

  model_pred = get_model(args.model, hijack=False).eval().to(device)
  model_fmap = get_model(args.model, hijack=True) .eval().to(device)

  @torch.inference_mode()
  def get_pred(X:Tensor) -> int:
    out = model_pred(X)
    return out.argmax(-1).cpu().numpy().item()

  @torch.inference_mode()
  def get_fmap(X:Tensor) -> Tensor:
    out = model_fmap(X)
    return out[0]

  preds = []
  fmap_L1 = []
  fmap_Linf = []

  fps = sorted([fp for fp in log_dp.iterdir() if fp.suffix == '.png'])
  X0 = load_img_tensor(fps[0])
  pred_ref = get_pred(X0)
  fmap_ref = get_fmap(X0)
  preds.append(pred_ref)
  fmap_L1.append(0.0)
  fmap_Linf.append(0.0)
  for fp in tqdm(fps[1:]):
    X = load_img_tensor(fp)
    pred = get_pred(X)
    fmap = get_fmap(X)

    preds.append(pred)
    DX = torch.abs(fmap - fmap_ref)
    fmap_L1.append(DX.mean().item())
    fmap_Linf.append(DX.max().item())
  
  plt.clf()
  plt.subplot(311) ; plt.title('preds')     ; plt.plot(preds) ; plt.ylim((0, 999))
  plt.subplot(312) ; plt.title('fmap_L1')   ; plt.plot(fmap_L1)
  plt.subplot(313) ; plt.title('fmap_Linf') ; plt.plot(fmap_Linf)
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--log_path', default='log/default', help='generated image folder')
  parser.add_argument('-M', '--model', default='resnet50', choices=MODELS, help='surrogate model for attack')
  args = parser.parse_args()

  vis(args)
