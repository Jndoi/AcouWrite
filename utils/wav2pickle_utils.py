"""
@Project : AcouWrite
@File : wav2csv_pickle_utils.py
@Date : 2022/4/18 23:16
@Author : Qiuyang Zeng
@Software : PyCharm

"""
import os
import pickle
from tqdm import tqdm
from receiver import Receiver
from constants import START_INDEX_SHIFT


class DataItem:
    def __init__(self, label, split_abs_d_cir):
        self.label = label  # the index arr of label just like [1, 2] (start with 1)
        self.split_abs_d_cir = split_abs_d_cir


class DataItemWithPhase(DataItem):  # contains real phase
    def __init__(self, label, split_abs_d_cir, split_real_phase):
        super().__init__(label, split_abs_d_cir)
        self.split_real_phase = split_real_phase


def wav2pickle(wav_base_path, dump_path=None,
               start_index_shift=START_INDEX_SHIFT, augmentation_radio=None):
    data = []
    for root, dirs, files in os.walk(wav_base_path):
        for file in tqdm(files, desc=root):
            if os.path.splitext(file)[1] == '.wav':
                label = file.split("_")[0]
                split_abs_d_cir = Receiver.receive(root, file,
                                                   start_index_shift=start_index_shift,
                                                   augmentation_radio=augmentation_radio)
                for i in range(len(label)):
                    label_item_int = ord(label[i]) - ord('a')
                    data.append(DataItem(label_item_int, split_abs_d_cir[i]))
    if dump_path:
        pickle.dump(data, open(dump_path, 'wb'))


def load_data_from_pickle(base_path=None):
    if base_path:
        return pickle.load(open(base_path, 'rb'))
    else:
        raise Exception("Base path is None")


if __name__ == '__main__':
    # Different handwriting distance
    DISTANCE_AWAY_5_CM = 14
    DISTANCE_AWAY_10_CM = 28
    # Different handwriting speed
    SPEED_FASTER = 0.8  # time is 4/5 times the original
    SPEED_SLOWER = 1.25  # time is 5/4 times the original
    pass
