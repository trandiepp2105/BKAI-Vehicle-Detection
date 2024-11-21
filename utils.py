import zipfile
import os


def yolo_to_bbox(yolo_bbox, img_width = 1280, img_height = 720):
    """
    Chuyển đổi bounding box từ định dạng YOLO sang định dạng (x_min, y_min, x_max, y_max).

    Định dạng YOLO bao gồm các thông số:
    - x_center: Tọa độ x của tâm bounding box (tính theo tỷ lệ với chiều rộng hình ảnh).
    - y_center: Tọa độ y của tâm bounding box (tính theo tỷ lệ với chiều cao hình ảnh).
    - width: Chiều rộng của bounding box (tính theo tỷ lệ với chiều rộng hình ảnh).
    - height: Chiều cao của bounding box (tính theo tỷ lệ với chiều cao hình ảnh).

    Args:
        yolo_bbox (list): Danh sách chứa 4 phần tử [x_center, y_center, width, height],
                          trong đó x_center, y_center, width, height đều là số thực.
        img_width (int): Chiều rộng của hình ảnh (pixel).
        img_height (int): Chiều cao của hình ảnh (pixel).

    Returns:
        list: Danh sách chứa 4 phần tử [x_min, y_min, x_max, y_max],
              với x_min, y_min là tọa độ góc trên bên trái,
              và x_max, y_max là tọa độ góc dưới bên phải của bounding box (pixel).
    """
    x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    x_max = (x_center + width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    y_max = (y_center + height / 2) * img_height
    return [x_min, y_min, x_max, y_max]

def bbox_to_yolo(bbox, img_width = 1280, img_height = 720):
    """
    Chuyển đổi bounding box từ định dạng (x_min, y_min, x_max, y_max) sang định dạng YOLO.

    Args:
        bbox (list): Danh sách chứa 4 phần tử [x_min, y_min, x_max, y_max],
                     trong đó x_min, y_min là tọa độ góc trên bên trái,
                     và x_max, y_max là tọa độ góc dưới bên phải của bounding box (pixel).
        img_width (int): Chiều rộng của hình ảnh (pixel).
        img_height (int): Chiều cao của hình ảnh (pixel).

    Returns:
        list: Danh sách chứa 4 phần tử [x_center, y_center, width, height],
              với x_center, y_center là tọa độ tâm bounding box (tính theo tỷ lệ),
              và width, height là chiều rộng và chiều cao của bounding box (tính theo tỷ lệ).
    """
    x_min, y_min, x_max, y_max = bbox
    # Tính toán tọa độ tâm
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    # Tính toán chiều rộng và chiều cao của bounding box
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return [x_center, y_center, width, height]


def read_bounding_boxes(file_path, img_width = 1280, img_height = 720):
    """
    Đọc bounding boxes từ file và chuyển đổi từ định dạng YOLO sang định dạng (x_min, y_min, x_max, y_max).

    Mỗi dòng trong file chứa thông tin về một bounding box, với định dạng:
    <class_id> <x_center> <y_center> <width> <height>

    Args:
        file_path (str): Đường dẫn tới file chứa bounding boxes.
        img_width (int): Chiều rộng của hình ảnh (pixel).
        img_height (int): Chiều cao của hình ảnh (pixel).

    Returns:
        list: Danh sách chứa các bounding box, mỗi bounding box là một tuple với 
              định dạng (class_id, bbox), trong đó bbox là danh sách [x_min, y_min, x_max, y_max].
    """
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            # Giá trị đầu tiên là class_id(cộng thêm 1 để không trùng với background id)    
            class_id = int(parts[0]) + 1 
            yolo_bbox = list(map(float, parts[1:]))  # 4 phần tử tiếp theo
            bbox = yolo_to_bbox(yolo_bbox, img_width, img_height)
            boxes.append((class_id, bbox))
    return boxes

def zip_txt_file(input_file, output_zip):
    """
    Nén một file cụ thể thành một file .zip.
    
    Args:
        input_file (str): Đường dẫn file .txt cần nén.
        output_zip (str): Đường dẫn file ZIP đầu ra.
    """
    # Kiểm tra xem file đầu vào có tồn tại không
    if not os.path.exists(input_file):
        print(f"Lỗi: File không tồn tại: {input_file}")
        return
    
    # Tạo thư mục đích nếu chưa tồn tại
    output_dir = os.path.dirname(output_zip)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Đã tạo thư mục: {output_dir}")
    
    # Nén file vào ZIP
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        arcname = os.path.basename(input_file)  # Tên file trong file ZIP
        zipf.write(input_file, arcname)
        print(f"Đã nén file: {input_file} vào {output_zip}")