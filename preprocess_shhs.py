import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
import numpy as np
from utils.data import SleepData
from utils.file_loader_shhs import load_npz_files_SHHS
from typing import Tuple,List


def _get_datasets(
                  modals: int,
                  npz_files_pairs: List[Tuple[str,str]],
                  stride: int = 35,
                  two_d: bool = True
) -> List[SleepData]:

    data, labels = load_npz_files_SHHS(
        npz_files_pairs,
        two_d=two_d
    )

    def data_big_group(d: np.ndarray) -> np.ndarray:
        """
        A closure to divide data into big groups to prevent
        data leak in data enhancement.
        """
        return_data = np.array([])
        beg = 0
        while (beg + stride) <= d.shape[0]:
            y = d[beg: beg + stride, ...]
            y = y.reshape((1, 1, 35, 3000, 3))
            return_data = y if beg == 0 else np.append(return_data, y, axis=0)
            beg += stride
        return return_data

    def label_big_group(labels: np.ndarray) -> np.ndarray:
        """
        A closure to divide labels into big groups to prevent
        data leak in data enhancement.
        """
        return_labels = np.array([])
        beg = 0
        while (beg + stride) <= len(labels):
            y = labels[beg: beg + stride]
            """y[y == 5] = 4  # 将标签中的5改为4
            y[y == 4] = 3  # 将标签中的4改为3"""
            y = y[np.newaxis, ...]
            return_labels = y if beg == 0 else np.concatenate(
                (return_labels, y),
                axis=0
            )
            beg += stride
        return return_labels

    with ThreadPoolExecutor(max_workers=4) as executor:
        data = executor.map(data_big_group, data)
        labels = executor.map(label_big_group, labels)
    # ch_nameslist = ['EEG', 'EOG(L)', 'EOG(R)']
    if modals is None:
        datasets = [SleepData(d, l) for d, l in zip(data, labels)]
    elif modals == 12:  # EEG and EOG(R)
        datasets = [SleepData(d[..., [0,2]], l) for d, l in zip(data, labels)]
    elif modals == 13:  # eeg only
        datasets = [SleepData(d[..., [0,0]], l) for d, l in zip(data, labels)]
    elif modals == 14:  # eog only
        datasets=[SleepData(d[..., [2,2]], l) for d, l in zip(data, labels)]
    else:

        datasets = [SleepData(d[..., modals], l) for d, l in zip(data, labels)]

    return datasets
# Datasets corresponding to eeg,eog and (eeg.eog) channel
# Since SalientSleepNet has dual channel inputs,
# double eeg(or eog) here means that both channels use eeg(or eog) as inputs to simulate the case where only the eeg(or eog) channel data is used as input.
get_eeg_and_eog_datasets = partial(_get_datasets, 12)
get_double_eeg_datasets = partial(_get_datasets, 13)
get_double_eog_datasets = partial(_get_datasets, 14)