# encoding: utf8
# @Author: qiaoyongchao
# @Email: qiaoyc2023@163.com
import time

from src.models.modules import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize)) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs, channel_nums

