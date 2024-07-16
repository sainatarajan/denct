import os, sys
import glob
import numpy as np
import pandas as pd
import torchio as tio
import nibabel as nib
import random
import cv2
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torchvision

from tqdm import tqdm
from time import time
from datetime import datetime as dt


# class Femur3D(torch.utils.data.Dataset):
#
#     def __init__(self, volumes, exp_path, csv_file='PhantomCalibrationLaw_ConvToK2PO4.csv', mode='train',
#                  refresh=False):
#         self.volumes = volumes
#         self.mode = mode
#         self.refresh = refresh
#         self.EXP_PATH = exp_path
#         self.calibration_df = pd.read_csv(os.path.join('/media/snatarajan/data/denct/femurv2/T_data/', csv_file))
#         self.volume_names = open(os.path.join(self.EXP_PATH, 'train_files.txt'), 'r').readlines()
#         self.volume_names = [f.strip() for f in self.volume_names]
#
#         self.task_df = self.calibration_df[self.calibration_df['ListID'].isin(self.volume_names)]
#
#         self.task_slopes = self.task_df['Slope'].tolist()
#         self.task_intercepts = self.task_df['Intercept'].tolist()
#
#         self.SLOPE_MAX = 1.47310870103702
#         self.SLOPE_MIN = 1.18625047981472
#         self.INTERCEPT_MAX = 13.0775121085252
#         self.INTERCEPT_MIN = -28.3327398931856
#
#         print(self.SLOPE_MIN, self.SLOPE_MAX, self.INTERCEPT_MIN, self.INTERCEPT_MAX)
#         # sys.exit()
#         self.prepared_volumes = []
#         self.load_data()
#
#     def print_data(self):
#
#         for idx, v in enumerate(self.volumes):
#             print(os.path.basename(v), self.slopes[idx], self.intercepts[idx])
#
#     def load_data(self, ):
#
#         for idx, data in enumerate(self.volumes):
#             self.prepared_volumes.append(tio.ScalarImage(data))
#
#         print('Volumes prepared for ', str(self.mode), ' : ', len(self.prepared_volumes))
#
#     def prepare_data(self, input_volume):
#
#         vol = input_volume
#
#         transform = tio.Compose([
#             tio.transforms.RandomFlip(axes=('LR')),
#             tio.transforms.ZNormalization(),
#             tio.RescaleIntensity(out_min_max=(-1, 1)),
#         ])
#         volo = transform(vol)
#         vol = volo.tensor.float()
#         return vol
#
#     def __getitem__(self, index):
#         volume = self.prepare_data(self.prepared_volumes[index])
#         v_name = os.path.basename(self.volumes[index])
#
#         slope = torch.tensor(float(self.calibration_df.loc[self.calibration_df['ListID'] == v_name]['Slope'])).float()
#         intercept = torch.tensor(
#             float(self.calibration_df.loc[self.calibration_df['ListID'] == v_name]['Intercept'])).float()
#
#         slope = 2 * (slope - self.SLOPE_MIN) / (self.SLOPE_MAX - self.SLOPE_MIN) - 1
#         intercept = 2 * (intercept - self.INTERCEPT_MIN) / (self.INTERCEPT_MAX - self.INTERCEPT_MIN) - 1
#
#         gt = torch.tensor([slope, intercept]).float()
#
#         data_dict = {
#             'volume': volume,
#             'gt': gt
#         }
#
#         return data_dict
#
#     def __len__(self, ):
#         return len(self.volumes)


class DenCT3D(torch.utils.data.Dataset):

    def __init__(self, data_dir, type='train', transforms=None):
        self.data_dir = data_dir
        self.type = type
        self.transforms = tio.Compose([
            tio.transforms.RandomFlip(axes=('LR')),
            # tio.transforms.Resize((8, 8, 8)),
            tio.transforms.ZNormalization(),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])

        self.calibration_df = pd.read_csv(os.path.join(self.data_dir, 'calibration.csv'))
        if type == 'train':
            self.volumes = glob.glob(os.path.join(self.data_dir, self.type) + '/*.nii.gz')
        else:
            self.volumes = glob.glob(os.path.join(self.data_dir, self.type) + '/*.nii.gz')

        # Write to a file
        text_file = os.path.join(self.data_dir, self.type + '_files.txt')
        with open(text_file, 'w') as f:
            self.filenames = [os.path.basename(v) for v in self.volumes]
            for fname in self.filenames:
                f.write(f'{fname}\n')

            f.close()

        self.task_df = self.calibration_df[self.calibration_df['ListID'].isin(self.filenames)]

        self.task_slopes = self.task_df['Slope'].tolist()
        self.task_intercepts = self.task_df['Intercept'].tolist()

        self.SLOPE_MAX = 1.47310870103702
        self.SLOPE_MIN = 1.18625047981472
        self.INTERCEPT_MAX = 13.0775121085252
        self.INTERCEPT_MIN = -28.3327398931856

        # print(self.SLOPE_MIN, self.SLOPE_MAX, self.INTERCEPT_MIN, self.INTERCEPT_MAX)

        self.prepared_volumes = []
        self.load_data()

    def load_data(self, ):
        for idx, data in tqdm(enumerate(self.volumes)):
            volume = tio.ScalarImage(data)
            volume = self.transforms(volume)
            volume = volume.tensor.float()
            self.prepared_volumes.append(volume)

        print('Volumes prepared for ', str(self.type), ' : ', len(self.prepared_volumes))

    def __getitem__(self, index):
        # volume = self.prepare_data(self.prepared_volumes[index])
        volume = self.prepared_volumes[index]
        v_name = os.path.basename(self.volumes[index])

        slope = torch.tensor(float(self.calibration_df.loc[self.calibration_df['ListID'] == v_name]['Slope'])).float()
        intercept = torch.tensor(
            float(self.calibration_df.loc[self.calibration_df['ListID'] == v_name]['Intercept'])).float()

        slope = 2 * (slope - self.SLOPE_MIN) / (self.SLOPE_MAX - self.SLOPE_MIN) - 1
        intercept = 2 * (intercept - self.INTERCEPT_MIN) / (self.INTERCEPT_MAX - self.INTERCEPT_MIN) - 1

        gt = torch.tensor([slope, intercept]).float()

        return volume, gt

    def __len__(self, ):
        return len(self.volumes)
