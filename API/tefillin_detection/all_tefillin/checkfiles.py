import os

def get_file_basenames(folder_path):
    """Return a set of file names without extensions from a given folder."""
    return {os.path.splitext(filename)[0] for filename in os.listdir(folder_path)}

def find_unshared_files(labels_path, photos_path):
    labels_files = get_file_basenames(labels_path)
    photos_files = get_file_basenames(photos_path)

    only_in_labels = labels_files - photos_files
    only_in_photos = photos_files - labels_files

    return only_in_labels, only_in_photos

# Replace with your actual path if running outside VS Code
labels_folder = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/labels"
photos_folder = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/photos"

only_in_labels, only_in_photos = find_unshared_files(labels_folder, photos_folder)

print("Files only in 'labels' folder (not in 'photos'):")
for name in sorted(only_in_labels):
    print(f"  - {name}")

print("\nFiles only in 'photos' folder (not in 'labels'):")
for name in sorted(only_in_photos):
    print(f"  - {name}")

