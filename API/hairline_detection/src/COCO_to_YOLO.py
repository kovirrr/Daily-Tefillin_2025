import json
import os

def coco_to_yolo(coco_file, output_dir):
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Ensure output_dir is a directory, not a file
    if output_dir.endswith('.txt'):
        output_dir = os.path.dirname(output_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a mapping of image id to file name and dimensions
    image_map = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco_data['images']}

    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        img_filename, img_width, img_height = image_map[img_id]

        txt_file = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)
        
        with open(txt_path, 'a') as f:
            # Assuming only one class, so using 0 as class id
            class_id = 0
            segmentation = ann['segmentation'][0]  # Assuming single polygon
            
            # Convert segmentation to YOLO format
            yolo_seg = []
            for i in range(0, len(segmentation), 2):
                yolo_seg.append(segmentation[i] / img_width)
                yolo_seg.append(segmentation[i+1] / img_height)
            
            # Write to file
            f.write(f"{class_id} {' '.join(map(str, yolo_seg))}\n")

    print("Conversion completed.")

# Usage
coco_file = '/Users/koviressler/Desktop/DailyTefillin/hairline_detection/labels_my-project-name_2024-07-18-06-46-50.json'
output_dir = '/Users/koviressler/Desktop/DailyTefillin/hairline_detection/labels'
coco_to_yolo(coco_file, output_dir)