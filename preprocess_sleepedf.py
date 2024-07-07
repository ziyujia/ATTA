import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
if sys.platform == "linux":
    sys.path.append(r"/mnt/nfs-storage/allthingssalient/sss")
elif sys.platform == "win32":
    sys.path.append(r"D:\Python\MySleepProject")
import numpy as np
from utils.data import SleepData
from utils.file_loader_sleepedf import load_npz_files
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
from utils.data import SleepData
from typing import Tuple,List

def _get_datasets(
                  modals: int,
                  data_dir: str,
                  stride: int = 35,
                  two_d: bool = True
) -> List[SleepData]:
    data, labels = load_npz_files(
        glob.glob(os.path.join(data_dir, '*.npz')),
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
            # y = y[np.newaxis, ...]
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
            y = y[np.newaxis, ...]
            return_labels = y if beg == 0 else np.concatenate(
                (return_labels, y),
                axis=0
            )
            beg += stride
        return return_labels  # [:, np.newaxis, ...]

    with ThreadPoolExecutor(max_workers=4) as executor:
        data = executor.map(data_big_group, data)
        labels = executor.map(label_big_group, labels)

    if modals is None:
        datasets = [SleepData(d, l) for d, l in zip(data, labels)]

    elif modals == 1:
        datasets = [SleepData(d[..., :2], l) for d, l in zip(data, labels)]
    elif modals == 2:
        datasets=[]
        for d, l in zip(data, labels):
            d[...,1]=d[..., 0]
            eeg_dataset=SleepData(d[..., :2], l)
            datasets.append(eeg_dataset)
    elif modals == 3:
        datasets=[]
        for d, l in zip(data, labels):
            d[..., 0] = d[..., 1]
            eog_dataset = SleepData(d[..., :2], l)
            datasets.append(eog_dataset)
    else:
        datasets = [SleepData(d[..., modals], l) for d, l in zip(data, labels)]

    return datasets

# Datasets corresponding to each channel
# Since SalientSleepNet has dual channel inputs,
# double eeg(or eog) here means that both channels use eeg (or eog) as inputs to simulate the case where only the eeg(or eog) channel data is used as input.
get_eeg_and_eog_datasets = partial(_get_datasets, 1)
get_double_eeg_datasets = partial(_get_datasets, 2)
get_double_eog_datasets = partial(_get_datasets, 3)
get_datasets = partial(_get_datasets, None)