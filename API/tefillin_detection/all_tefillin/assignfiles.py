import os
import shutil

def get_file_basenames(folder_path):
    """Return a set of file basenames (no extensions) from a given folder."""
    return {os.path.splitext(f)[0] for f in os.listdir(folder_path)}

def copy_matching_photos(labels_path, photos_path, destination_path):
    label_basenames = get_file_basenames(labels_path)

    for filename in os.listdir(photos_path):
        basename, ext = os.path.splitext(filename)
        if basename in label_basenames:
            src_path = os.path.join(photos_path, filename)
            dst_path = os.path.join(destination_path, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename} → {destination_path}")

# SET YOUR PATHS HERE:
labels_folder = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/labels"
photos_folder = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/ALLphotos"
destination_folder = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/GOODphotos"  # <- CHANGE THIS

copy_matching_photos(labels_folder, photos_folder, destination_folder)
