"""
@Project :acouinput_python
@File ：constants.py
@Date ： 2022/4/7 13:27
@Author ： Qiuyang Zeng
@Software ：PyCharm

"""
import numpy as np


class SinOrCosType(object):
    Sin = "Sin"
    Cos = "Cos"


class SignalType(object):
    Barker = "Barker"
    GSM = "GSM"


class DataType(object):
    AbsDCir = "AbsDCir"
    RealPhase = "RealPhase"
    AbsDCirAndRealPhase = "mixed"


class SignalPaddingType(object):
    ZeroPadding = "ZeroPadding"
    LastValuePadding = "LastValuePadding"


class DatasetLoadType(object):
    ALL = "ALL"
    TrainAndTest = "TrainAndTest"
    TrainValidAndTest = "TrainValidAndTest"
    UniformTrainValidAndTest = "UniformTrainValidAndTest"


class DataSourceType(object):
    IMG = "IMG"
    CSV = "CSV"
    PICKLE = "PICKLE"


class DeviceType(object):
    OPPOWatch2 = "OPPOWatch2"
    HONOR30Pro = "HONOR30Pro"


TrainingSequence = {
    SignalType.Barker: np.array([-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1,
                                 -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1]),
    SignalType.GSM: np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1,
                              1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1])
}

SignalPaddingTypeMap = {
    DataType.AbsDCir: SignalPaddingType.ZeroPadding,
    DataType.RealPhase: SignalPaddingType.LastValuePadding
}

START_INDEX_SHIFT = 5
Fs = 48000  # 48kHz
Fc = 20000  # 20kHz
Up_times = 12
UpperFrequency = 22000
LowerFrequency = 18000
LowPassFrequency = 2000
DCIRLowPassFrequency = 8000
DefaultAlignFramesNum = 3
DefaultPhaseThreshold = 0.6
FRAME_LENGTH = 480

# CIR
TAP_SIZE = 60
WINDOW_SIZE = 40  # 0.4s
STEP = 20  # overlap = 50%
P_VALUE = 192
L_VALUE = 120
Amplitude = np.iinfo(np.int16).max
WaveLength = 340/Fc
DEFAULT_CONFIG = {
    "DataSourceType": DataSourceType.PICKLE,
    "DeviceType": DeviceType.HONOR30Pro,
    "DatasetLoadType": DatasetLoadType.TrainValidAndTest,
    "SignalType": SignalType.Barker,
    "DataType": DataType.AbsDCir,
    "SignalPaddingType": SignalPaddingType.ZeroPadding,
}

LabelVocabulary = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
LABEL_CLASSES = ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


if __name__ == '__main__':
    pass
