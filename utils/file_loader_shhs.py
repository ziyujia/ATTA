import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, lfilter
from functools import partial

def load_npz_file(npz_data_file_name: str,npz_label_file_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load training and validation data from npz files.
    :param npz_file_name: a str of npz filename.
    :return: a tuple of PSG data, labels and sampling rate of the npz file.
    """
    print(f"Loading {npz_data_file_name}.")
    x = np.load(npz_data_file_name) #x:(ne,nc,6000)
    y = np.load(npz_label_file_name)#y:(ne,)
    # low-pass filtering
    nyquist = 200 / 2  # Nyquist frequency is half the sampling rate
    cutoff = 50 / nyquist
    b, a = butter(5, cutoff, btype='low')
    filtered_data = np.empty(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            filtered_data[i, j, :] = lfilter(b, a, x[i, j, :])
    # Downsampling
    x = filtered_data[:, :, ::2]
    x = np.transpose(x,(0,2,1))

    sampling_rate = 100
    return x, y, sampling_rate

def load_npz_files(
        npz_file_pairs: List[Tuple[str,str]],
        workers: int = 4,
        two_d: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data and labels for training and validation.
    Note that we default use 3 channels, if that's changed, need to change the axes stuff code.
    :param two_d: denote data's dimension is adapted by Conv2D else Conv1D.
    :param workers: size of threads pool.
    :param npz_files_name: a list of str contains npz files' name.
    :return: the list of chosen PSG data and labels. Returning with `npz_files_name`'s order.
    """
    assert len(npz_file_pairs) > 0
    data_list, label_list, fs_list = [], [], []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(load_npz_file, *zip(*npz_file_pairs)):
            data_list.append(record[0].astype(np.float32))
            label_list.append(record[1].astype(np.int32))
            fs_list.append(record[2])
            data_list = list(executor.map(lambda x: np.squeeze(x), data_list))
            if two_d:  # Conv2d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, np.newaxis, :, :], data_list))
            else:  # Conv1d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, ...], data_list))

    if len(np.unique(fs_list)) != 1:
        raise Exception("Found mismatch in sampling rate.")

    print(f"load {len(data_list)} files totally.")
    return data_list, label_list



def load_npz_file_SHHS(npz_data_file_name: str,npz_label_file_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load training and validation data from npz files.
    :param npz_file_name: a str of npz filename.
    :return: a tuple of PSG data, labels and sampling rate of the npz file.
    """
    print(f"Loading {npz_data_file_name}.")
    x = np.load(npz_data_file_name)['x']   # x:(ne,3,3000)
    y = np.load(npz_data_file_name)['y']   # y:(ne,)
    x = np.transpose(x, (0, 2, 1))

    sampling_rate = 100
    return x, y, sampling_rate

def load_npz_files_SHHS(
        npz_file_pairs: List[Tuple[str,str]],
        workers: int = 4,
        two_d: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    assert len(npz_file_pairs) > 0
    data_list, label_list, fs_list = [], [], []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(load_npz_file_SHHS, *zip(*npz_file_pairs)):
            data_list.append(record[0].astype(np.float32))
            label_list.append(record[1].astype(np.int32))
            fs_list.append(record[2])
            data_list = list(executor.map(lambda x: np.squeeze(x), data_list))
            if two_d:  # Conv2d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, np.newaxis, :, :], data_list))
            else:      # Conv1d
                data_list = list(executor.map(lambda x: x[:, np.newaxis, ...], data_list))

    if len(np.unique(fs_list)) != 1:
        raise Exception("Found mismatch in sampling rate.")

    print(f"load {len(data_list)} files totally.")
    return data_list, label_list