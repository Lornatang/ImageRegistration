# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import multiprocessing
import os
import shutil

import cv2
import numpy as np
from natsort import natsorted
from sewar.full_ref import ssim
from tqdm import tqdm


def main(args) -> None:
    # 重置实验结果文件夹
    if os.path.exists(os.path.join("data", "LR")):
        shutil.rmtree(os.path.join("data", "LR"))
    if os.path.exists(os.path.join("data", "HR")):
        shutil.rmtree(os.path.join("data", "HR"))
    os.makedirs(os.path.join("data", "LR"))
    os.makedirs(os.path.join("data", "HR"))

    image_file_names = natsorted(os.listdir(os.path.join("data", "original", "LR")))

    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Registration rotate image")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    lr = cv2.imread(os.path.join("data", "original", "LR", image_file_name))
    hr = cv2.imread(os.path.join("data", "original", "HR", image_file_name))

    # 截取低分辨图像中心区域，防止旋转后出现黑边
    registration_lr = center_crop(lr, [args.registration_image_height // args.scale_factor, args.registration_image_width // args.scale_factor])

    # 记录相似度最高情况时的角度和SSIM指标
    best_ssim_score = 0.0
    best_angle = 0
    for angle in np.arange(-3, 3, 0.5):
        rotate_hr = rotate(hr, angle)
        crop_hr = center_crop(rotate_hr, [args.registration_image_height, args.registration_image_width])
        registration_hr = cv2.resize(crop_hr,
                                     [args.registration_image_width // args.scale_factor, args.registration_image_height // args.scale_factor],
                                     interpolation=cv2.INTER_CUBIC)

        # 计算SSIM指标
        ssim_score = ssim(registration_lr, registration_hr)[0]

        # 如果当前旋转角度下SSIM指标大于之前角度SSIM指标，则记录当前角度
        if ssim_score > best_ssim_score:
            best_ssim_score = max(ssim_score, best_ssim_score)
            best_angle = angle

    # 配准图像
    rotate_hr = rotate(hr, best_angle)
    registration_hr = center_crop(rotate_hr, [args.registration_image_height, args.registration_image_width])

    cv2.imwrite(os.path.join("data", "LR", image_file_name), registration_lr)
    cv2.imwrite(os.path.join("data", "HR", image_file_name), registration_hr)


def rotate(image: np.ndarray, angle: float, center: int = None, scale_factor: float = 1.0) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random specific angle
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotate_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotate_image


def center_crop(image: np.ndarray, crop_image_size: tuple) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    crop_image_height, crop_image_width = crop_image_size

    # Just need to find the top and left coordinates of the image
    top = (image_height - crop_image_height) // 2
    left = (image_width - crop_image_width) // 2

    # Crop image patch
    crop_image = image[top:top + crop_image_height, left:left + crop_image_width, ...]

    return crop_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--registration_image_height", type=int, default=1904)
    parser.add_argument("--registration_image_width", type=int, default=2560)
    parser.add_argument("--scale_factor", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    main(args)
