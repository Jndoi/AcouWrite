"""
@Project :acouinput_python
@File ：plot_utils.py
@Date ： 2022/4/7 13:28
@Author ： Qiuyang Zeng
@Software ：PyCharm
ref: https://blog.csdn.net/qq_41645987/article/details/109148274
"""
import numpy as np
import seaborn as sns
from scipy.fftpack import fft
import matplotlib.pyplot as plt


def show_fft(signals, **kwargs):
    """
    show the fft result of signals
    :param signals:
    :return:
    """
    plt.plot(abs(fft(signals, **kwargs)), linewidth=0.5)
    plt.show()


def show_signals(signals, is_frames=False):
    """
    plot origin signals
    :param is_frames:
    :param signals:
    :return:
    """
    # plt.figure(figsize=(10, 6), dpi=600)
    plt.figure(figsize=(10, 6))
    if is_frames:
        frame = signals[0]
        for index in np.arange(1, signals.shape[0]):
            frame = np.r_[frame, signals[index]]
    else:
        frame = signals
    # frame = frame.squeeze()
    plt.plot(frame, linewidth=0.5)
    plt.margins(0, 0.1)
    # plt.plot(np.ones(frame.shape)*3, linewidth=0.5)
    plt.show()


def show_finger_movement_distance(data):
    plt.figure(figsize=(10, 5), dpi=600)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.linewidth'] = 1
    plt.grid(axis="both", color='lightgray', linestyle='--', zorder=1)
    plt.plot(np.arange(0, data.shape[0])/100.0, data)
    plt.xlabel("Time (sec)", fontsize=32)
    plt.ylabel("Distance (cm)", fontsize=32)
    plt.margins(0, 0.1)
    plt.savefig(r"C:\Users\zengq\Desktop\finger_movement_distance.pdf", dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def show_finger_movement_d_cir(data):
    plt.figure(figsize=(10, 5))
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 24
    plt.rcParams['axes.linewidth'] = 1
    plt.xlabel("Time (sec)", fontsize=32)
    plt.ylabel("Tap Index", fontsize=32)
    plt.pcolormesh(np.arange(0, data.shape[1]) / 100.0, np.arange(1, data.shape[0] + 1), data, cmap='jet',
                   shading='auto')
    plt.savefig(r"C:\Users\zengq\Desktop\finger_movement_d_cir.pdf", dpi=600, bbox_inches='tight', pad_inches=0.0)



def show_phase(signals, is_frames=False):
    """
    plot origin signals
    :param is_frames:
    :param signals:
    :return:
    """
    plt.figure(figsize=(10, 6), dpi=200)
    if is_frames:
        frame = signals[0]
        for index in np.arange(1, signals.shape[0]):
            frame = np.r_[frame, signals[index]]
    else:
        frame = signals
    # frame = frame.squeeze()
    plt.plot(np.arange(0, frame.shape[0])/100.0, frame, linewidth=0.5)
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    plt.xlabel("Time (sec)")
    plt.ylabel("Distance (cm)")
    plt.show()


def show_d_cir(d_cir, is_frames=False):
    """
    plot dCIR image
    :param is_frames:
    :param d_cir: difference CIR
    :return:
    """
    # the shape of d_cir is (N, 60, 30), we should change it to (60, N*30)
    plt.figure(figsize=(10, 6))
    if is_frames:
        d_cir = d_cir.squeeze()
        d_cir = np.transpose(d_cir, [1, 0, 2])
        d_cir = np.reshape(d_cir, (d_cir.shape[0], -1), order='C')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    plt.xlabel("Time (sec)")
    plt.ylabel("Tap Index")
    plt.pcolormesh(np.arange(0, d_cir.shape[1])/100.0, np.arange(1, d_cir.shape[0]+1), d_cir, cmap='jet', shading='auto')
    plt.show()


if __name__ == '__main__':
    pass
