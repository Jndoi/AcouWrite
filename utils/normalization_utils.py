"""
@Project : AcouWrite
@File : normalization_utils.py
@Date : 2022/4/7 13:26
@Author : Qiuyang Zeng
@Software : PyCharm

"""
import numpy as np


def normalization(data):
    """
    mapping data into [-1, 1]
    :param data:
    :return:
    """
    return data/max(abs(data))


def normalization_signed_1(data):
    """
    mapping data into [0, 1]
    :param data:
    :return:
    """
    _range = np.max(data) - np.min(data)
    return ((data - np.min(data)) / _range - 0.5) * 2


def normalization_unsigned_1(data):
    """
    mapping data into [-1, 1]
    :param data:
    :return:
    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    """
    standardize data
    :param data:
    :return:
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
