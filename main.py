import os
import shutil

import cv2
import numpy as np
import torch
from natsort import natsorted
from sewar.full_ref import ssim

import imgproc
from model import HDRNet


def delete_background(image_dir: str) -> None:
    print("Delete background.....")
    # 粗筛法
    image_file_names = natsorted(os.listdir(image_dir))
    for file_name in image_file_names:
        image = cv2.imread(os.path.join(image_dir, file_name))
        # 利用图像方差来判断图像中是否包含细胞和背景
        if np.average(cv2.meanStdDev(image)[1]) > 20 or np.average(cv2.meanStdDev(image)[1]) < 5:
            os.remove(os.path.join(image_dir, file_name))

    # 细筛法
    image_file_names = natsorted(os.listdir(image_dir))
    for file_name in image_file_names:
        # 读取图像
        image = cv2.imread(os.path.join(image_dir, file_name))
        # 缩小图像，主要减少计算量，加快计算速度
        image = cv2.resize(image, [256, 256], interpolation=cv2.INTER_AREA)
        # 对彩色图像进行灰度化
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 对图像进行自动判别法的二值化
        _, bin_image = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
        # 计算二值图像中黑色区域的面积
        image_height, image_width = bin_image.shape

        tissue_area = 0
        for i in range(image_height):
            for j in range(image_width):
                # 0.0表示全黑
                if bin_image[i, j] == 0.0:
                    tissue_area += 1

        # 计算背景区域占比
        tissue_area /= (image_height * image_width)

        if tissue_area < 0.1:
            os.remove(os.path.join(image_dir, file_name))

    print("Finished delete background.")


def color_reproduction(model_path, inputs_dir, output_dir, device):
    # 创建Dakewe图像色彩还原后的文件夹
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = HDRNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    file_names = natsorted(os.listdir(inputs_dir))

    with torch.no_grad():
        for file_name in file_names:
            print(f"[color_reproduction]: {file_name}...")

            # 读取图像，BGR格式
            input_image = cv2.imread(os.path.join(inputs_dir, file_name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

            # 转为PyTorch所支持的Tensor流格式
            input_tensor = imgproc.image2tensor(input_image, range_norm=False, half=False).to(device).unsqueeze_(0)

            # 推理
            output_tensor = model(input_tensor).clamp_(0.0, 1.0)

            # 反射图 = 原图 / 光照估计
            output_tensor = torch.div(input_tensor, output_tensor + 1e-4)

            # 保存图像
            output_image = imgproc.tensor2image(output_tensor, range_norm=False, half=False)
            cv2.imwrite(os.path.join(output_dir, file_name), output_image)


def register(dakewe_cr_dir, dakewe_dir, nikon_dir, lr_dir, hr_dir):
    nikon_file_names = natsorted(os.listdir(nikon_dir))
    dakewe_file_names = natsorted(os.listdir(dakewe_dir))

    for file_name in nikon_file_names:
        print(f"[register]: {file_name}...")
        # 读取目标图像，BGR格式
        nikon_image = cv2.imread(os.path.join(nikon_dir, file_name), cv2.IMREAD_UNCHANGED)

        # 保存裁剪后的目标图像，此为HR图像
        hr_image = nikon_image[32:-32, 168:-72, ...]
        cv2.imwrite(os.path.join(hr_dir, file_name), hr_image)

        # 缩小HR图像，用作template图像
        template_image = cv2.resize(hr_image, [330, 248], interpolation=cv2.INTER_AREA)

        # 每次寻找完后都重置ssim得分
        max_ssim = 0.
        for dakewe_file_name in dakewe_file_names:
            # 读取原始Dakewe图像和色彩还原后图像
            dakewe_image = cv2.imread(os.path.join(dakewe_dir, dakewe_file_name), cv2.IMREAD_UNCHANGED)
            dakewe_cr_image = cv2.imread(os.path.join(dakewe_cr_dir, dakewe_file_name), cv2.IMREAD_UNCHANGED)

            # 从Dakewe图像中找到与Nikon图像对应的区域并保存
            template_height, template_width = template_image.shape[:2]
            template_result = cv2.matchTemplate(dakewe_cr_image, template_image, cv2.TM_SQDIFF)
            cv2.normalize(template_result, template_result, 0, 1, cv2.NORM_MINMAX, -1)
            _, _, min_loc, _ = cv2.minMaxLoc(template_result)
            lr_image = dakewe_image[min_loc[1]: min_loc[1] + template_height, min_loc[0]: min_loc[0] + template_width, ...]

            if lr_image.shape == template_image.shape:
                # 计算匹配图与模板图的SSIM得分
                ssim_score = ssim(lr_image, template_image)[0]

                # 通常来说，SSIM得分越高表示匹配图与模板图结构越相似
                is_best = ssim_score > max_ssim
                max_ssim = max(max_ssim, ssim_score)

                # 小于0.15表示与图像完全不匹配，出现该情况可能是因为当前配准算法无法准确从原始图像找到与之相对应的配准图像
                # 大于0.5表示背景图像，也就是纯白背景下，SSIM得分相对较高，这是一个误判
                if is_best and 0.15 < ssim_score < 0.5:
                    # 保存与HR对应的LR图像
                    cv2.imwrite(os.path.join(lr_dir, file_name), lr_image)

                    # 保存匹准时找到的Dakewe原始图像
                    cv2.imwrite(os.path.join(dakewe_dir, file_name), dakewe_image)


def autoclean() -> None:
    print("Cleaning.....")
    # 删除色彩还原后的Dakewe图像，仅保留原始图像
    if os.path.exists("samples/color_reproduction"):
        shutil.rmtree("samples/color_reproduction")

    # 删除目录下所有未准确配准的图像
    current_file_name = os.listdir("data/LR")

    for file_name in os.listdir("data/original/NIK"):
        if file_name not in current_file_name:
            if os.path.exists(os.path.join("data", "original", "NIK", file_name)):
                os.remove(os.path.join("data", "original", "NIK", file_name))
            if os.path.exists(os.path.join("data", "original", "DKW", file_name)):
                os.remove(os.path.join("data", "original", "DKW", file_name))
            if os.path.exists(os.path.join("data", "HR", file_name)):
                os.remove(os.path.join("data", "HR", file_name))

    # 将所有处理过后的图像数据转移至数据盘
    current_file_name = os.listdir("data/LR")
    for file_name in current_file_name:
        shutil.move(os.path.join("data", "original", "DKW", file_name), os.path.join("E:/SuperResolution/Dakewe/original/DKW", file_name))
        shutil.move(os.path.join("data", "original", "NIK", file_name), os.path.join("E:/SuperResolution/Dakewe/original/NIK", file_name))
        shutil.move(os.path.join("data", "LR", file_name), os.path.join("E:/SuperResolution/Dakewe/LR", file_name))
        shutil.move(os.path.join("data", "HR", file_name), os.path.join("E:/SuperResolution/Dakewe/HR", file_name))

    # 删除原始Dakewe图像
    shutil.rmtree("data/original/DKW")
    os.makedirs("data/original/DKW")

    print("Finished clean.")


if __name__ == "__main__":
    device = torch.device("cuda", 2)

    # delete_background(image_dir="data/original/DKW")

    color_reproduction(model_path="results/pretrained_models/deepUPE.pth",
                       inputs_dir="data/original/DKW",
                       output_dir="samples/color_reproduction",
                       device=device)

    register(dakewe_cr_dir="samples/color_reproduction",
             dakewe_dir="data/original/DKW",
             nikon_dir="data/original/NIK",
             lr_dir="data/LR",
             hr_dir="data/HR")

    # autoclean()
