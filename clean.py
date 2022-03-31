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
import os
import shutil


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
    autoclean()
