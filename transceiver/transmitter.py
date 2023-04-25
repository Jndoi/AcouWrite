"""
@Project : AcouWrite
@File : transmitter.py
@Date : 2022/4/7 13:27
@Author : Qiuyang Zeng
@Software :PyCharm

"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft, ifft
from utils.audio_utils import AudioUtils
from constants.constants import TrainingSequence, SinOrCosType, SignalType


class Transmitter(object):

    @classmethod
    def get_passband_sequence(cls, signal_type=SignalType.Barker):
        root_path = "../"
        if os.getcwd().endswith("AcouInput"):
            root_path = ""
        if signal_type == SignalType.GSM:
            return pickle.load(open(root_path+'data/gsm_passband.pkl', 'rb'))
        elif signal_type == SignalType.Barker:
            return pickle.load(open(root_path+'data/barker_passband.pkl', 'rb'))
        else:
            raise Exception("NoSuchTypeError")

    @classmethod
    def get_baseband_sequence(cls, signal_type=SignalType.Barker):
        root_path = "../"
        if os.getcwd().endswith("AcouInput"):
            root_path = ""
        if signal_type == SignalType.GSM:
            return pickle.load(open(root_path + 'data/gsm_baseband.pkl', 'rb'))
        elif signal_type == SignalType.Barker:
            return pickle.load(open(root_path + 'data/barker_baseband.pkl', 'rb'))
        else:
            raise Exception("NoSuchTypeError")

    @classmethod
    def gen_sequence(cls, signal_type=SignalType.GSM):
        training_seq = TrainingSequence.get(signal_type)
        show_barker_code(training_seq)
        training_seq_fft = fftshift(fft(training_seq))
        len_up_sample = len(training_seq) * 12
        up_sample_training_seq_fft = np.zeros(len_up_sample, dtype=complex)
        up_sample_training_seq_fft[143:169] = training_seq_fft
        up_sample_training_seq_fft = fftshift(up_sample_training_seq_fft)
        up_sample_training_seq = np.real(ifft(up_sample_training_seq_fft))
        up_sample_training_seq = np.r_[up_sample_training_seq, np.zeros(12*14)]
        # min-max normalization
        up_sample_training_seq = up_sample_training_seq / np.max(np.abs(up_sample_training_seq))
        show_up_sample_training_seq(up_sample_training_seq)
        up_sample_training_seq_fc = up_sample_training_seq * AudioUtils.build_cos_or_sin(
            len(up_sample_training_seq), SinOrCosType.Cos)
        training_seq_fc = AudioUtils.band_pass(up_sample_training_seq_fc)
        show_training_seq_fc(training_seq_fc)
        # dump signal
        project_path = os.path.pardir
        os.chdir(project_path)
        pickle.dump(up_sample_training_seq, open(os.path.join("{}//data".format(project_path), signal_type + '_baseband.pkl'), 'wb'))
        pickle.dump(training_seq_fc, open(os.path.join("{}//data".format(project_path), signal_type+'_passband.pkl'), 'wb'))
        training_seq_fc = np.tile(training_seq_fc, (1, 2000))  # 20sec
        empty_seq = np.zeros((1, training_seq_fc.shape[1]))
        # the dim of 0 is top speaker and the dim of 1 is bottom speaker
        audio_signal = np.r_[training_seq_fc, empty_seq].T
        audio_folder_path = os.path.join(os.path.abspath('..'), "audio")
        if not os.path.exists(audio_folder_path):
            os.mkdir(audio_folder_path)
        # AudioUtils.write_audio(audio_signal, os.path.join(audio_folder_path, signal_type + "_frequency.wav"))


def show_barker_code(training_seq):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.stem(np.arange(len(training_seq)), training_seq, linefmt='--', basefmt='k-')
    plt.grid(axis="y", color='lightgray', linestyle='--', zorder=1)
    plt.xlabel('Sample', fontsize=48)
    plt.ylabel('Amplitude', fontsize=48)
    plt.ylim(-1, 1)
    plt.savefig(r"C:\Users\zengq\Desktop\{}.pdf".format("barker_code"),
                dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.show()


def show_up_sample_training_seq(up_sample_training_seq):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.plot(np.arange(len(up_sample_training_seq)), up_sample_training_seq)
    plt.grid(axis="y", color='lightgray', linestyle='--', zorder=1)
    plt.xlabel('Sample', fontsize=48)
    plt.ylabel('Amplitude', fontsize=48)
    plt.ylim(-1, 1)
    plt.show()


def show_training_seq_fc(training_seq_fc):
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 36
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.plot(np.arange(len(training_seq_fc)), training_seq_fc, linewidth=0.5)
    plt.grid(axis="y", color='lightgray', linestyle='--', zorder=1)
    plt.xlabel('Sample', fontsize=48)
    plt.ylabel('Amplitude', fontsize=48)
    plt.ylim(-1, 1)
    plt.show()


if __name__ == '__main__':
    Transmitter.gen_sequence(signal_type=SignalType.Barker)
