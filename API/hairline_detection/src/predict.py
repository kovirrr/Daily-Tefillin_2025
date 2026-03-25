import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import scipy.ndimage as ndi
from model import bce_dice_loss, dice_coef, iou_score

def detect_hairline(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Remove small objects
    binary_mask = ndi.binary_opening(binary_mask, structure=np.ones((5,5)))
    
    # Find the top edge of the largest connected component
    labeled, num_features = ndi.label(binary_mask)
    sizes = ndi.sum(binary_mask, labeled, range(1, num_features + 1))
    largest_component = sizes.argmax() + 1
    largest_mask = (labeled == largest_component)
    
    hairline = []
    for col in range(binary_mask.shape[1]):
        col_pixels = np.where(largest_mask[:, col])[0]
        if len(col_pixels) > 0:
            hairline.append((col, col_pixels[0]))
    
    return hairline

def predict_and_detect_hairline(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    hairline = detect_hairline(prediction[:,:,0])
    
    return img, prediction, hairline

def check_tefillin_position(image, hairline, tefillin_bbox):
    image_height, image_width = image.shape[:2]
    
    # Check if tefillin is centered
    tefillin_center_x = (tefillin_bbox[0] + tefillin_bbox[2]) / 2
    nose_x = image_width / 2
    is_centered = abs(tefillin_center_x - nose_x) < (image_width * 0.1)  # Within 10% of center
    
    # Check if tefillin is above hairline
    tefillin_bottom_y = tefillin_bbox[3]
    hairline_y_at_tefillin = np.interp(tefillin_center_x, [x for x, y in hairline], [y for x, y in hairline])
    is_above_hairline = tefillin_bottom_y < hairline_y_at_tefillin
    
    return is_centered and is_above_hairline

def main():
    model = load_model('../models/best_model.h5', custom_objects={
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou_score': iou_score
    })
    
    image_path = '../data/images/test_image.jpg'  # Replace with your test image
    img, pred, hairline = predict_and_detect_hairline(model, image_path)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(pred[:,:,0], cmap='gray')
    plt.title('Predicted Mask')
    plt.subplot(133)
    plt.imshow(img)
    plt.plot([x for x, y in hairline], [y for x, y in hairline], 'r-', linewidth=2)
    plt.title('Detected Hairline')
    plt.show()
    
    # Example tefillin bounding box (you would need to implement tefillin detection separately)
    tefillin_bbox = [200, 50, 300, 100]  # [x_min, y_min, x_max, y_max]
    is_correct = check_tefillin_position(img, hairline, tefillin_bbox)
    print(f"Tefillin position is correct: {is_correct}")

if __name__ == "__main__":
    main()