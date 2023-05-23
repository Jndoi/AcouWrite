"""
@Project : AcouWrite
@File : data_augmentation_utils.py
@Date : 2022/7/6 16:21
@Author : Qiuyang Zeng
@Software : PyCharm

"""
from scipy.interpolate import interp1d
import numpy as np


def augmentation_speed(y, speed_radio=1.25):
    """

    :param y: the abs of differential cir
    :param speed_radio: the speed radio
    :return:
    """
    if len(y.shape) == 1:  # DataType.RealPhase
        x = np.arange(y.shape[0])
    else:  # DataType.AbsDCir
        x = np.arange(y.shape[1])
    x_new = np.linspace(min(x), max(x), int(len(x) * speed_radio))
    y_new = []
    if len(y.shape) == 1:
        f = interp1d(x, y, kind="cubic")
        y_new = f(x_new)
    else:
        for y_item in y:
            f = interp1d(x, y_item, kind="cubic")  # 三次样条插值
            y_item_new = f(x_new)
            y_new.append(y_item_new)
        y_new = np.array(y_new)
    return y_new
