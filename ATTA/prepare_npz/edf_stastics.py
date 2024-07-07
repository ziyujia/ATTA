import os
import mne

# Specify the path to the folder containing the EDF file.
edf_folder = "../../../../sleep-telemetry"

# Get a list of filenames of all EDF files in a folder.
edf_files = [f for f in os.listdir(edf_folder) if f.endswith("PSG.edf")]

# Initialize the dictionary to store channel occurrences and channel totals.
channel_counts = {}
total_channel_counts = {}

# For each EDF file.
for edf_file in edf_files:
    edf_file_path = os.path.join(edf_folder, edf_file)

    # Read EDF files with the mne library.
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)

    # Get a list of channel names in this file.
    channel_names = raw.info["ch_names"]

    # Count the occurrences of each channel
    for channel in channel_names:
        if channel in channel_counts:
            channel_counts[channel] += 1
        else:
            channel_counts[channel] = 1

# Iterate over counted channel occurrences
for channel, count in channel_counts.items():
    print(f"Channel: {channel}, Total Count: {count}")

# Count the total number of times each channel is in all EDF files
for edf_file in edf_files:
    edf_file_path = os.path.join(edf_folder, edf_file)
    raw = mne.io.read_raw_edf(edf_file_path, preload=False)
    channel_names = raw.info["ch_names"]
    for channel in channel_names:
        if channel in total_channel_counts:
            total_channel_counts[channel] += 1
        else:
            total_channel_counts[channel] = 1

# Print the total number of times each channel is in all EDF files
print("\nTotal Channel Counts Across All EDF Files:")
for channel, count in total_channel_counts.items():
    print(f"Channel: {channel}, Total Count: {count}")

# Identify channels that are common to all EDF files
common_channels = [channel for channel, count in total_channel_counts.items() if count == len(edf_files)]
print("\nCommon Channels Across All EDF Files:")
print(common_channels)