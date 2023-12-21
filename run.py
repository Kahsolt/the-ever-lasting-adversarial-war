#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/12/20 

from time import time
from shutil import copy2
from argparse import ArgumentParser, Namespace
from typing import Dict
from traceback import print_exc

from skimage.metrics import *
from niqe.niqe import get_niqe

from atk import *
from dfn import *
from models import *
from utils import *

ndarray_to_tensor = lambda x: torch.from_numpy(x).float().permute([2, 0, 1]).unsqueeze_(0).to(device)
tensor_to_ndarray = lambda x: x.detach().squeeze(0).permute([1, 2, 0]).cpu().numpy().astype(np.float32)


def get_quality_metrics(im_raw:npimg, im_adv:npimg, model:Module=None) -> Dict[str, float]:
  for x in [im_raw, im_adv]:
    assert isinstance(x, ndarray)
    assert x.dtype == np.uint8
    assert len(x.shape) == 3 and x.shape[-1] == 3

  niqe_raw = get_niqe(im_raw)
  niqe_adv = get_niqe(im_adv)
  mse  = mean_squared_error(im_raw, im_adv)
  rmse = normalized_root_mse(im_raw, im_adv)
  psnr = peak_signal_noise_ratio(im_raw, im_adv)
  ssim = structural_similarity(to_gray(im_raw), to_gray(im_adv))

  metrics = {
    'niqe_raw': niqe_raw,
    'niqe_adv': niqe_adv,
    'mse': mse,
    'rmse': rmse,
    'psnr': psnr,
    'ssim': ssim,
  }

  if model is not None:
    with torch.no_grad():
      X  = ndarray_to_tensor(im_raw.copy())
      AX = ndarray_to_tensor(im_adv.copy())
      pred_raw = model(X)
      pred_adv = model(AX)
      diff = torch.abs(pred_adv - pred_raw)
      metrics['fm_L1']   = diff.mean().item()
      metrics['fm_Linf'] = diff.max ().item()

  return metrics


def apply_pgd(args:NameError, model:Module, im:npimg) -> npimg:
  im = im_u8_to_f32(im)
  X = ndarray_to_tensor(im)
  with torch.no_grad():
    Y = model(X).detach().clone()
  AX = pgd(model, X, Y, args.eps, args.alpha, args.steps)[0]
  im = tensor_to_ndarray(AX)
  im = im_valid(im)
  return im_f32_to_u8(im)


def apply_adv_clean(im:npimg) -> npimg:
  im = rgb2bgr(im)
  im = adv_cleaner(im)
  im = bgr2rgb(im)
  return im


def run(args:Namespace):
  log_dp = LOG_PATH / str(args.name)
  log_fp = LOG_PATH / f'{args.name}.json'
  log_dp.mkdir(exist_ok=True)

  ''' init image '''
  init_fp = log_dp / seq_fn(0)
  if not init_fp.exists():
    copy2(args.file, init_fp)
  im_ref = load_im(init_fp)
  im_ref_f32 = im_u8_to_f32(im_ref)

  ''' record '''
  db: Dict[int, Any] = load_json(log_fp)

  ''' resume '''
  if args.resume:
    fp = log_dp / f'{args.resume}.png'
    assert fp.exists(), f'>> resume from non-existing file {fp}'
    step = args.resume
  else:
    step = max([int(e) for e in db.keys()]) if db.keys() else 0
  im_cur = load_im(log_dp / seq_fn(step))

  ''' model (for atk) '''
  model = get_model(args.model)
  model = model.eval().to(device)

  try:
    for step in range(step+1, 1000):
      print(f'>> [step {step}] ', end='')

      ts = time()
      if step % 2 == 1:   # odd step, apply attack
        print('attack!!')
        im_cur = apply_pgd(args, model, im_cur)
      else:               # even step, apply defense
        print('defend!!')
        im_cur = apply_adv_clean(im_cur)
      ts = time() - ts

      im_cur = im_f32_to_u8(color_fix(im_u8_to_f32(im_cur), im_ref_f32))

      metrics = get_quality_metrics(im_ref, im_cur, model)
      metrics['ts'] = ts
      db[step] = metrics
      print(metrics)
      print('=' * 42)

      save_im(im_cur, log_dp / seq_fn(step))
      if step % 10 == 0: save_json(log_fp, db)
  except KeyboardInterrupt:
    print('>> Exit by Ctrl+C')
  except:
    print_exc()
  finally:
    save_json(log_fp, db)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-F', '--file', default='img/raw.png', help='image file to attack/defend')
  parser.add_argument('-M', '--model', default='resnet50', choices=MODELS, help='surrogate model for attack')
  parser.add_argument('--eps',   default=16/255, type=float, help='PGD attack threshold')
  parser.add_argument('--alpha', default=1/255,  type=float, help='PGD attack step size')
  parser.add_argument('--steps', default=40,     type=int,   help='PGD attack step count')
  parser.add_argument('-R', '--resume', type=int, help='resume from given round of beat')
  parser.add_argument('-N', '--name', default='default', help='experiment name')
  args = parser.parse_args()

  run(args)
