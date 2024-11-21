import cv2
import numpy as np
import random
import os
from multiprocessing import Pool, cpu_count
from utils import read_bounding_boxes, bbox_to_yolo
def adjust_brightness_contrast(image, brightness=10, contrast=40):
    """
    Adjusts the brightness and contrast of an image.

    Parameters:
    ----------
    image : numpy.ndarray
        The input image to be adjusted. The image should be in BGR color format (as loaded by `cv2.imread`).
    brightness : int, optional
        The amount to adjust the brightness. Positive values increase brightness,
        while negative values decrease it. Default is 10.
    contrast : int, optional
        The percentage to adjust the contrast. Positive values increase contrast,
        while negative values decrease it. Default is 40.

    Returns:
    -------
    numpy.ndarray
        The adjusted image in RGB color format.
    """
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 100, beta=brightness)
    return adjusted

def local_contrast_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhances the local contrast of an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters:
    ----------
    image : numpy.ndarray
        The input image to be enhanced. The image should be in BGR color format (as loaded by `cv2.imread`).
    clip_limit : float, optional
        Threshold for contrast limiting. Higher values allow for more contrast enhancement. Default is 2.0.
    tile_grid_size : tuple of int, optional
        Size of the grid for dividing the image into tiles. Each tile is processed independently. 
        Default is (8, 8).

    Returns:
    -------
    numpy.ndarray
        The enhanced image with improved local contrast

    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_image

    
def day_to_night(original_img):
    """
    Simulates a day-to-night transformation for an image.

    Parameters:
    ----------
    original_img : numpy.ndarray
        The input image to be transformed. The image should be in BGR color format (as loaded by `cv2.imread`).

    Returns:
    -------
    numpy.ndarray
        The transformed image with a night-time appearance
    """
    dark_image = cv2.convertScaleAbs(original_img, alpha=0.55, beta=-30)
    blurred_image = cv2.GaussianBlur(dark_image, (7, 7), 0)
    return blurred_image

def rotate_and_blend_roi(image_path, labels_path, output_path, angle=None, scale=1.0):
    """
    Xoay vùng ROI trong ảnh và blend trở lại ảnh gốc.

    Parameters:
    - image_path: Đường dẫn tới ảnh
    - labels_path: Đường dẫn tới file chứa các tọa độ bounding box
    - angle: Góc xoay (độ). Nếu không truyền, chọn ngẫu nhiên từ [0, 15].
    - scale: Tỉ lệ phóng to/thu nhỏ vùng xoay.

    Returns:
    - image_rotated: Ảnh sau khi blend vùng xoay trở lại.
    """
    try:
        img_height = 720
        img_width = 1280
        image = cv2.imread(image_path)

        bboxes = read_bounding_boxes(labels_path)
    #     class_id, bbox = boxes[4]
    #     x_min, y_min, x_max, y_max = bbox
        new_boxes = []
        if angle is None:
            angle = round(random.uniform(15, 20), 1)  # Ngẫu nhiên góc xoay từ 10-20 độ
        direction = ""
        if angle > 0:
            direction = "pos"
        else:
            direction = "neg"
        for class_id, bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            y = round(y_min)
            x = round(x_min)
            w = round(x_max - x_min) 
            h = round(y_max - y_min) 

            # Cắt ROI từ ảnh gốc
            roi = image[y:y+h, x:x+w]

            # Tạo ma trận xoay
            center = (w // 2, h // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

            # Kích thước mới để chứa toàn bộ vùng xoay
            new_w = int(w * abs(np.cos(np.radians(angle))) + h * abs(np.sin(np.radians(angle))))
            new_h = int(w * abs(np.sin(np.radians(angle))) + h * abs(np.cos(np.radians(angle))))

            # Điều chỉnh ma trận xoay
            rotation_matrix[0, 2] += (new_w - w) / 2
            rotation_matrix[1, 2] += (new_h - h) / 2

            # Xoay vùng ROI
            rotated_roi = cv2.warpAffine(roi, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

            # Tạo alpha channel cho vùng xoay
            alpha_channel = np.zeros((new_h, new_w), dtype=np.uint8)
            alpha_channel[rotated_roi[:, :, 0] > 0] = 255
            rotated_roi_with_alpha = cv2.merge((rotated_roi, alpha_channel))

            # Xác định vùng mới trong ảnh gốc
            y_new = max(0, y - (new_h - h) // 2)
            x_new = max(0, x - (new_w - w) // 2)
            y_end = min(img_height, y_new + new_h)
            x_end = min(img_width, x_new + new_w)

            # Blend vùng xoay vào ảnh gốc
            roi_bg = image[y_new:y_end, x_new:x_end]
            roi_fg = rotated_roi_with_alpha[:y_end-y_new, :x_end-x_new]
            b, g, r, a = cv2.split(roi_fg)
            alpha = a / 255.0

            for c in range(3):
                roi_bg[:, :, c] = (alpha * roi_fg[:, :, c] + (1 - alpha) * roi_bg[:, :, c]).astype(np.uint8)

            image[y_new:y_end, x_new:x_end] = roi_bg
            bbox = [x_new, y_new, x_end, y_end]
            x_center, y_center, width, height = bbox_to_yolo(bbox)
            new_boxes.append((class_id, x_center, y_center, width, height))

        # Lưu ảnh mới
        if angle < 0:
            angle = - angle

        os.makedirs(output_path, exist_ok=True) 

        base_name = image_path.split('/')[-1].split('.')[0]
        output_image_path = f"{output_path}/{base_name}_{direction}_{angle}.jpg"
        cv2.imwrite(output_image_path, image)

        # Lưu file labels mới
        output_labels_path = f"{output_path}/{base_name}_{direction}_{angle}.txt"
        with open(output_labels_path, 'w') as f:
            for class_id, x_new, y_new, x_end, y_end in new_boxes:
                f.write(f"{class_id - 1} {x_new} {y_new} {x_end} {y_end}\n")
        # Chuyển sang RGB để hiển thị
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image_rgb
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
        return None