import os
import random
import shutil

# INPUT THESE PATHS
source_images_dir = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/ALLphotos"
source_labels_dir = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/all_tefillin/labels"

train_img_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/train/images"
train_lbl_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/train/labels"
val_img_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/valid/images"
val_lbl_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/valid/labels"
test_img_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/test/images"
test_lbl_dest = "/Users/koviressler/Desktop/DailyTefillin/tefillin_detection/finallyTef/test/labels"

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Get all .jpg or .jpeg image filenames
all_images = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

# Calculate split sizes
total = len(all_images)
train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)
test_count = total - train_count - val_count  # Remaining

# Partition images
train_imgs = all_images[:train_count]
val_imgs = all_images[train_count:train_count + val_count]
test_imgs = all_images[train_count + val_count:]

def copy_files(images, img_dest, lbl_dest):
    for img in images:
        base_name, _ = os.path.splitext(img)
        label_file = base_name + ".txt"

        img_src = os.path.join(source_images_dir, img)
        lbl_src = os.path.join(source_labels_dir, label_file)

        img_dst = os.path.join(img_dest, img)
        lbl_dst = os.path.join(lbl_dest, label_file)

        if os.path.exists(img_src):
            shutil.copy2(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, lbl_dst)

# Copy all sets
copy_files(train_imgs, train_img_dest, train_lbl_dest)
copy_files(val_imgs, val_img_dest, val_lbl_dest)
copy_files(test_imgs, test_img_dest, test_lbl_dest)

print(f"Copied {len(train_imgs)} to train, {len(val_imgs)} to val, and {len(test_imgs)} to test.")
