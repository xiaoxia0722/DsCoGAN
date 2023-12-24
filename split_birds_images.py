# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com

import os
import shutil
import sys
import random

# 图片所在路径
from tqdm import tqdm

img_path = sys.argv[1]

# 图片保存路径
output_path = sys.argv[2]

# 测试数据占比
val_rate = float(sys.argv[3])

train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')

print("获取目录下的所有图片")
dir_list = []
img_list = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            img_list.append(os.path.join(os.path.basename(root), file))
            dir_list.append(os.path.basename(root))

random.shuffle(img_list)

lens = len(img_list)

val_size = int(lens * val_rate)

train_size = lens - val_size

print("总图片数为:{}, 训练图片数为:{}, 验证图片数为:{}".format(lens, train_size, val_size))

train_list = img_list[:train_size]

val_list = img_list[train_size:]

print("创建文件夹")
for dir_path in dir_list:
    train_dir = os.path.join(train_path, dir_path)
    val_dir = os.path.join(val_path, dir_path)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

print("对训练集图片进行复制")
for img_name in tqdm(train_list):
    shutil.copyfile(os.path.join(img_path, img_name), os.path.join(train_path, img_name))

print("对测试集图片进行复制")
for img_name in tqdm(val_list):
    shutil.copyfile(os.path.join(img_path, img_name), os.path.join(val_path, img_name))

