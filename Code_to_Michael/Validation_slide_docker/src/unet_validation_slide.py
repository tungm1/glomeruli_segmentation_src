# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import cv2
import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image

from collections import OrderedDict
from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, Resize
from matplotlib import cm
import matplotlib.pyplot as plt
import tifffile
import scipy.ndimage as ndi
import imageio
import numpy as np
import pandas as pd
import argparse


def save_validate(val_images, val_outputs, output_dir, images, cnt):
    for i in range(val_images.shape[0]):
        folder_list = os.path.dirname(images[cnt+i]).split('/')
        save_folder = os.path.join(output_dir, folder_list[-3], folder_list[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now_image = val_images[i].permute([2,1,0]).detach().cpu().numpy()
        now_pred = val_outputs[i][0].permute([1,0]).detach().cpu().numpy()
        name = os.path.basename(images[cnt+i])
        plt.imsave(os.path.join(save_folder, 'val_%s_img.png' % (name)), now_image)
        plt.imsave(os.path.join(save_folder, 'val_%s_pred.png' % (name)), now_pred, cmap = cm.gray)

    cnt += val_images.shape[0]
    return cnt

def calculate_f1(precision, recall):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Convert NaNs to zero if precision and recall are both zero
    return f1_scores

def sodelete(wsi, min_size):
    """
    Remove objects smaller than min_size from binary segmentation image.

    Args:
    img (numpy.ndarray): Binary image where objects are 255 and background is 0.
    min_size (int): Minimum size of the object to keep.

    Returns:
    numpy.ndarray: Image with small objects removed.
    """
    # Find all connected components (using 8-connectivity, as default)
    _, binary = cv2.threshold(wsi* 255, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8, cv2.CV_32S)

    # Create an output image that will store the filtered objects
    # output = np.zeros_like(wsi, dtype=np.uint8)
    output = np.zeros_like(wsi)

    # Loop through all found components
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]

        # If the size of the component is larger than the threshold, copy it to output
        if size >= min_size:
            output[labels == i] = 1.

    return output


def main(data_dir, model_dir, output_dir, X20_dir, X20_patch_dir, get_X20_wsi, get_X20_patch_preds, get_X20_wsi_preds):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image_paths = glob(os.path.join(data_dir, "*.tiff"))

    # Ensure exactly one image is found
    if len(image_paths) < 1:
        raise FileNotFoundError(f"No .tiff image found in directory: {data_dir}")
    elif len(image_paths) > 1:
        raise RuntimeError(f"Multiple .tiff images found in directory: {data_dir}.\n"
                        f"This script expects only one image.")
    
    image_path = image_paths[0]
    print("Found single input image:", image_path)

    if get_X20_wsi:

        'read wsi and get 20X and patches'
        lv = 2

        now_tiff = tifffile.imread(image_path,key=0)
        tiff_X20 = ndi.zoom(now_tiff, (1/lv, 1/lv, 1), order=1)

        if not os.path.exists(os.path.dirname(image_path.replace(data_dir, X20_dir))):
            os.makedirs(os.path.dirname(image_path.replace(data_dir, X20_dir)))

        imageio.imwrite(image_path.replace(data_dir, X20_dir).replace('.tiff', '.png'), tiff_X20)

        wsi_shape = tiff_X20.shape
        patch_size = 2048
        stride = 1024
        x_slide = int((wsi_shape[0] - patch_size) / stride) + 1
        y_slide = int((wsi_shape[1] - patch_size) / stride) + 1


        cnt = 1
        for xi in range(x_slide):
            for yi in range(y_slide):
                if xi == x_slide - 1:
                    now_x = wsi_shape[0] - patch_size
                else:
                    now_x = xi * stride
                if yi == y_slide - 1:
                    now_y = wsi_shape[1] - patch_size
                else:
                    now_y = yi * stride

                now_patch = tiff_X20[now_x:now_x + patch_size, now_y:now_y + patch_size, :]
                assert now_patch.shape == (patch_size, patch_size, 3)

                root_list = os.path.dirname(image_path).split('/')
                now_folder = os.path.join(X20_patch_dir)
                if not os.path.exists(now_folder):
                    os.makedirs(now_folder)
                imageio.imwrite(os.path.join(now_folder, '%s_%s_%d_%d_%d_img.png' % (root_list[-2], root_list[-1], cnt, now_x, now_y)), now_patch)
                cnt+=1
        print("Downscaling and patching complete.")
    if get_X20_patch_preds:
        patch_dir = os.path.join(X20_patch_dir)
    
        # Grab all patch images from that single folder
        images = sorted(glob(os.path.join(patch_dir, '*_img.png')))
        print('Total patch images:', len(images))

        # define transforms for image and segmentation
        imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize(spatial_size=(512, 512), mode='nearest'), ScaleIntensity()])
        outputrans = Compose([Resize(spatial_size=(2048, 2048), mode='nearest')])
        val_ds = ArrayDataset(images, imtrans)
        # sliding window inference for one image at every iteration
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        model_file = glob('/home/VANDERBILT/tungm1/Desktop/model/mask2former_swin_b_kpis_768_best_mDice.pth')[0]
        print("Model loaded from: ", model_file, "\n")
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            print(k, "\n")
            #name = k[7:] # remove module.
            #print(name, "\n\n")
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        with torch.no_grad():
            cnt = 0
            for val_data in val_loader:
                val_images = val_data.to(device)
                # define sliding window size and batch size for windows inference
                roi_size = (512, 512)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                val_images = outputrans(val_images[0]).unsqueeze(0)
                val_outputs = outputrans(val_outputs)
                cnt = save_validate(val_images, val_outputs, output_dir, images, cnt)
                print(cnt, " at val_data in val_loader")


    if get_X20_wsi_preds:
        wsi_X20_path = image_path.replace(data_dir, X20_dir).replace('.tiff', '.png')
        if not os.path.exists(wsi_X20_path):
            raise FileNotFoundError(f"20X WSI not found: {wsi_X20_path}")

        now_img_shape = plt.imread(wsi_X20_path).shape
        wsi_prediction = np.zeros((now_img_shape[0], now_img_shape[1]), dtype=np.float32)

        # find the folder with patch-level predictions (saved in the previous step)
        patch_folder = os.path.join(
            output_dir,
            os.path.basename(os.path.dirname(wsi_X20_path)),  # or define a simpler approach
            os.path.basename(wsi_X20_path).replace('_wsi.png', '')
        )

        # gather patch predictions
        patches = glob(os.path.join(patch_folder, '*_pred.png'))
        patch_size = 2048

        for patch in patches:
            now_name = os.path.basename(patch).split('_')
            now_x = int(now_name[-4])
            now_y = int(now_name[-3])
            now_patch = plt.imread(patch)
            wsi_prediction[now_x:now_x+patch_size, now_y:now_y+patch_size] = wsi_prediction[now_x:now_x+patch_size, now_y:now_y+patch_size] + now_patch[:,:,0]

        wsi_prediction[wsi_prediction <= 1] = 0
        wsi_prediction[wsi_prediction != 0] = 1

        sm = 20000
        wsi_prediction_sm = sodelete(wsi_prediction, sm)

        preds_folder = os.path.dirname(wsi_X20_path).replace('img', 'pred')
        os.makedirs(preds_folder, exist_ok=True)
        preds_root = os.path.join(
            preds_folder,
            os.path.basename(wsi_X20_path).replace('_wsi', '_pred_final')
        )
        plt.imsave(preds_root, wsi_prediction_sm, cmap='gray')



if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as data_dir:
    # data_dir = '/input_slide/'
    # output_dir = '/output_slide/'
    # patch_data_dir =  '/input_patch/'
    # model_dir = '/model/'
    # patch_output_dir = '/output_patch/'
    # wd = '/myhome/wd'

    # data_dir = '/desktop/input_slide'
    # model_dir = '/desktop/model'
    # output_dir = '/desktop/output_slide'
    # patch_data_dir = '/Data/KPIs/testing_data_wsi_patch_20X'
    # patch_output_dir = '/Data/KPIs/validation_slide_20X_patchoutput'

    # get_X20_wsi = 1
    # get_X20_patch_preds = 1
    # get_X20_wsi_preds = 1

    parser = argparse.ArgumentParser(description="Run WSI Inference Pipeline")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the input WSI."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the trained model (.pth file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for final WSI predictions."
    )
    parser.add_argument(
        "--X20_dir",
        type=str,
        required=True,
        help="Path to the directory where 20X downsampled WSIs will be saved."
    )
    parser.add_argument(
        "--X20_patch_dir",
        type=str,
        required=True,
        help="Path to the directory for 20X WSI patches."
    )
    parser.add_argument(
        "--get_X20_wsi",
        type=int,
        default=1,
        help="1 or 0, whether to downsample original WSI and create 20X patches."
    )
    parser.add_argument(
        "--get_X20_patch_preds",
        type=int,
        default=1,
        help="1 or 0, whether to run inference on the 20X patches."
    )
    parser.add_argument(
        "--get_X20_wsi_preds",
        type=int,
        default=1,
        help="1 or 0, whether to stitch patch predictions and produce final WSI mask."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    #main(data_dir, model_dir, patch_output_dir, output_dir, patch_data_dir, get_X20_wsi, get_X20_patch_preds, get_X20_wsi_preds, df)
    main(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        X20_dir=args.X20_dir,
        X20_patch_dir=args.X20_patch_dir,
        get_X20_wsi=args.get_X20_wsi,
        get_X20_patch_preds=args.get_X20_patch_preds,
        get_X20_wsi_preds=args.get_X20_wsi_preds
    )
