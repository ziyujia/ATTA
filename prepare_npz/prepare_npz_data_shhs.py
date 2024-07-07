import mne
import numpy as np
import os
import xml.etree.ElementTree as ET
import argparse

def read_xml_stages(xml_file_path):
    # Parse XML files to extract sleep stage data.
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    stages = []

    for scored_event in root.findall('.//ScoredEvent'):
        event_type = scored_event.find('EventType').text
        if event_type == 'Stages|Stages':
            event_concept = scored_event.find('EventConcept').text
            stage = event_concept.split('|')[-1]  # Extract the number after '|'.
            if stage=='4':
                stage='3'
                print(stage)
            elif stage=='5':
                stage='4'
                print(stage)
            duration = float(scored_event.find('Duration').text)
            repeat_times = int(duration / 30)  # Count repetitions.
            stages.extend([int(stage)] * repeat_times)  # Repeat the sleep phase the appropriate number of times and add to the list.

    return np.array(stages)

def process_edf_file(edf_file_path, xml_file_path,channels):
    # Read EDF files.
    print(edf_file_path)
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)

    channel_names = channels
    raw.pick_channels(channel_names,ordered=True)

    # Decrease the sample rate of the EEG channel to 100Hz and increase the sample rate of the EOG channel to 100Hz.
    raw.resample(100, npad="auto")

    # Set Epochs parameter: 30 seconds per Epoch.
    epoch_duration = 30  # 30 seconds
    sampling_rate = 100  # 100 Hz
    samples_per_epoch = epoch_duration * sampling_rate  # Number of samples per Epoch

    # Split Epochs.
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration - 1/sampling_rate, baseline=None, preload=True)

    # Extract data.
    epoch_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_samples)

    sleep_stages = read_xml_stages(xml_file_path)
    print(sleep_stages.shape)

    npzpath=edf_file_path.replace('.edf', '.npz')[1:]
    # Save processed data to NPZ file.
    np.savez("./npz"+npzpath, x=epoch_data, y=sleep_stages)

    print(f"Document is saved. Epoch : {epoch_data.shape[0]}, channel number：{epoch_data.shape[2]}, Number of samples per Epoch：{epoch_data.shape[1]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf_dir", "-e", type=str,
                        default=r"../data/SHHS/polysomnography/edfs/shhs1",
                        help="File path to the edf file")
    parser.add_argument("--xml_dir", "-x", type=str,
                        default=r"../data/SHHS/polysomnography/annotations-events-nsrr/shhs1",
                        help="File path to the xml file.")
    parser.add_argument("--select_ch", '-s', type=list,
                        default=[
                            'EEG',
                            'EOG(L)',
                            'EOG(R)'
                        ],
                        help="choose the channels for training.")
    args = parser.parse_args()
    edf_path = args.edf_dir
    xml_path=args.xml_dir
    channels=args.select_ch
    xml_files = {}
    # collect all the .xml files.
    for file in os.listdir(xml_path):
        if file.endswith('.xml'):
            xml_file_key = file[:12]
            xml_files[xml_file_key] = os.path.join(xml_path, file)

    # Iterates through all .edf files and looks for matching .xml files.
    for file in os.listdir(edf_path):
        if file.endswith('.edf'):
            edf_file_key = file[:12]
            edf_file_path = os.path.join(edf_path, file)
            xml_file_path = xml_files.get(edf_file_key)
            if xml_file_path:
                process_edf_file(edf_file_path, xml_file_path,channels)
            else:
                print(f"Corresponding XML file not found : {edf_file_key}.xml")
if __name__ == '__main__':
    main()