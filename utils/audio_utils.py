"""
@Project :acouinput_python
@File ：audio_utils.py
@Date ： 2022/4/7 13:26
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import numpy as np
from scipy import signal
from constants.constants import Fs, Fc, SinOrCosType, LowerFrequency, UpperFrequency, DefaultAlignFramesNum, \
      LowPassFrequency, DeviceType
from scipy.io.wavfile import write, read


class AudioUtils(object):

    @classmethod
    def build_cos_or_sin_with_offset(cls, signal_length, sin_or_cos, fc=Fc):
        pass

    @classmethod
    def build_cos_or_sin(cls, signal_length, sin_or_cos, fc=Fc):
        """
        build the sin or cos of given length
        :param fc:
        :param signal_length: the signal length
        :param sin_or_cos: type
        :return: the cos or sin
        """
        t = np.arange(0, signal_length) / Fs
        if sin_or_cos == SinOrCosType.Sin:
            return np.sin(2 * np.pi * fc * t)
        elif sin_or_cos == SinOrCosType.Cos:
            return np.cos(2 * np.pi * fc * t)

    @classmethod
    def x_corr(cls, x, y):
        """
        calculate the of two signals (same as xcorr in Matlab)
        >> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
        array([0.5, 2., 3.5, 3., 0.])
        :param x: one of the signal
        :param y: one of the signal
        :return: the convolution result of two signals
        """
        return np.correlate(x, y, "full")

    @classmethod
    def band_pass(cls, signals, lower_fre=LowerFrequency, upper_fre=UpperFrequency):
        [b, a] = signal.butter(8, [lower_fre * 2 / Fs, upper_fre * 2 / Fs], btype='bandpass', output='ba')
        return signal.filtfilt(b, a, signals)

    @classmethod
    def low_pass(cls, signals, lower_fre=LowPassFrequency):
        [b, a] = signal.butter(8, lower_fre * 2 / Fs, btype='lowpass', output='ba')
        return signal.filtfilt(b, a, signals)

    @classmethod
    def align_signal(cls, y, template, energy_thr, align_frames_num=DefaultAlignFramesNum):
        """
        get the arrival time of signal by calculate the STE(short time energy) and return the start index
        :param align_frames_num:
        :param y: np.array, stands for the received signal
        :param template: np.array, the template signal in pass band
        :param energy_thr: double, the energy threshold for align
        :return: the start index of signal
        """
        frame_len = template.shape[0]
        # times = 0.2  # 可以只管前0.2s
        times = 1  # 只管前0.2s
        frame_num = int(np.floor(Fs * times / frame_len))  # pick 1s audio to find the start frame
        len_y = y.shape[0]
        if len_y < frame_num * frame_len:  # if the length of audio is less than 1s, we can use y[0:frame_len*frame_num]
            frame_num = int(np.floor(len_y / frame_len))
        # the frame_num is less than align_frames_num
        if frame_num - align_frames_num <= 0:
            max_index = np.argmax(cls.x_corr(y, template))
            start_index = max_index - frame_len + 1
            return start_index
        # [start_index:end_index] stands for start_index to end_index-1
        frame_y = y[:frame_num * frame_len].reshape((frame_len, -1), order='F')
        frame_energy = sum(frame_y * frame_y)  # sum(arr) returns the sum of each column
        energy_thr = max(frame_energy) * energy_thr  # frame_energy is one-dim array
        frame_energy_thr = frame_energy > energy_thr
        frame_start_index = 0
        for i in np.arange(0, frame_num - align_frames_num):
            flag = True
            for j in np.arange(i, i+align_frames_num):
                if not frame_energy_thr[j]:
                    flag = False
                    break
            if flag:
                frame_start_index = i
                break
        start_index = frame_start_index * frame_len
        frame_start = y[start_index:(start_index + frame_len * 2)]  # find the arrive time during two frames
        max_index = np.argmax(cls.x_corr(frame_start, template))  # calculate the xcorr
        if max_index < frame_len - 1:
            start_index = start_index + max_index
        else:
            start_index = start_index + max_index - frame_len + 1
        return start_index

    @classmethod
    def read_audio(cls, file_path, device_type):
        """
        read audio and return the data
        ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        :param device_type:
        :param file_path: the source path of audio
        :return: data in audio top mic data[:, 0] bottom mic data[:, 1]
        """
        sample_rate, data = read(file_path)
        if device_type == DeviceType.HONOR30Pro:
            return data[:, 1]
        if device_type == DeviceType.OPPOWatch2:
            return data


if __name__ == '__main__':
    pass