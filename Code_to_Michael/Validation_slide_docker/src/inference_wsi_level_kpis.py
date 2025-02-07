import os
from pathlib import Path

from mmseg.apis import init_model, inference_model

from tqdm import tqdm

import cv2
import torch
import numpy as np
import tifffile
import scipy.ndimage

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="path to WSI file")
parser.add_argument("--config", type=str, help="config path")
parser.add_argument("--ckpt", type=str, help="checkpoint path")
parser.add_argument("--output", type=str, help="output path")
parser.add_argument("--patch_size", type=int, default=2048)
parser.add_argument("--stride", type=int, default=1024)

if __name__=="__main__":
    args = parser.parse_args()
    print(args)

    # define test_pipeline
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]

    # load model
    model = init_model(args.config, args.ckpt)
    # assign test_pipeline
    model.cfg.test_pipeline = test_pipeline
    print(model.cfg.model.backbone.type)

    wsi_path = args.input
    wsi_name = Path(wsi_path).stem

    # data is already in RGB. No need to remove non-tissue area since WSIs are already processed
    wsi_data = tifffile.imread(wsi_path, key=0)
    H, W, _ = wsi_data.shape

    # make sure to inference on 40X digital magnification
    if '/NEP25/' not in wsi_path:
        lv = 2
        wsi_data = scipy.ndimage.zoom(wsi_data, (1/lv, 1/lv, 1), order=1)
        H, W, _ = wsi_data.shape

    num_classes = 2
    pred_wsi_data = torch.full((num_classes, H, W), 0, dtype=torch.float)

    x_slide = int((W - args.patch_size) / args.stride) + 1
    y_slide = int((H - args.patch_size) / args.stride) + 1

    pbar = tqdm(range(x_slide*y_slide), leave=True)
    pbar.set_description(f'{wsi_name}')

    for xi in range(x_slide):
        for yi in range(y_slide):
            # update progress bar
            pbar.update(1)

            if xi == x_slide - 1:
                x_min = W - args.patch_size
            else:
                x_min = xi * args.stride

            if yi == y_slide - 1:
                y_min = H - args.patch_size
            else:
                y_min = yi * args.stride

            sub_wsi = wsi_data[y_min:y_min + args.patch_size, x_min:x_min + args.patch_size, :]
            # convert RGB to BGR
            sub_wsi = cv2.cvtColor(sub_wsi, cv2.COLOR_RGB2BGR)

            assert sub_wsi.shape == (2048, 2048, 3), f'Wrong shape {sub_wsi.shape}'

            # skip if image is non-tissue (i.e., all black)
            if len(np.unique(sub_wsi))==1:
                continue
            
            # predict
            pred_result = inference_model(model, sub_wsi)
            raw_logits = pred_result.seg_logits.data
            # softmax
            raw_logits = torch.softmax(raw_logits, dim=0).cpu()

            # store raw predictions
            pred_wsi_data[:, y_min:y_min + args.patch_size, x_min:x_min + args.patch_size] += raw_logits

    # normalize with softmax
    pred_wsi_data = torch.softmax(pred_wsi_data, dim=0)
    
    # get the predicted mask
    _, pred_seg = pred_wsi_data.max(axis=0, keepdims=True)
    pred_seg = pred_seg.cpu().numpy()[0].astype(np.uint8)

    # Optionally save the predicted segmentation
    if args.output:
        # If you only have a binary segmentation (0 or 1), you can multiply by 255
        save_path = os.path.join(args.output, f"{wsi_name}_pred.png")
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(save_path, (pred_seg * 255))
        print(f"Saved predicted mask to {save_path}")
    else:
        print("No output path provided; skipping save.")