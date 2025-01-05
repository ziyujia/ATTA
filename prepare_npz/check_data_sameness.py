import argparse
import glob
import os

import numpy as np


def main():
    """This function check if the new data same with old ones
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_data', type=str,
                        default='../sleep_data/old_npzs')
    parser.add_argument('--new_data', type=str,
                        default='../sleep_data/sleepedf/prepared')
    args = parser.parse_args()

    old_ones = glob.glob(os.path.join(args.old_data, '*.npz'))
    new_ones = glob.glob(os.path.join(args.new_data, '*.npz'))
    old_ones.sort()
    new_ones.sort()
    assert len(old_ones) == len(new_ones)

    for old, new in zip(old_ones, new_ones):
        print(f"[{os.path.basename(old).split('.')[0]}]:")
        with np.load(old) as of:
            with np.load(new) as nf:
                ox, nx = of['x'], nf['x']
                oy, ny = of['y'], nf['y']
                ofs, nfs = of['fs'], nf['fs']
                assert ox.shape == nx.shape
                assert oy.shape == ny.shape
                assert ofs == nfs

                print(f"\tIs `y` tags all same: \
                        {np.all(oy == ny)}")
                print(f"\tIs `x` tags all same: \
                        {np.all(ox == nx)}")
                print(f"\tIs the `EEG&EOG` tags all same: \
                        {np.all(ox[:,:,:2] == nx[:,:,:2])}")
        print("\n========================================\
                 ========================================\n")


if __name__ == '__main__':
    main()
