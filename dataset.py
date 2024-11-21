import torch
from torch.utils.data import Dataset
import os
import cv2
from utils import read_bounding_boxes
class VehicleDetectionDataset(Dataset):
    def __init__(self, train_folder, transform=None):
        self.train_folder = train_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Đường dẫn ảnh và file nhãn .txt
        image_file = self.image_files[idx]
        image_path = os.path.join(self.train_folder, image_file)
        label_file = os.path.join(self.train_folder, image_file.replace('.jpg', '.txt'))

        # Đọc ảnh
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]

        # Đọc bounding boxes và nhãn từ file .txt
        boxes_data = read_bounding_boxes(label_file, img_width, img_height)
#         boxes_data = read_bounding_boxes(label_file, 224, 224)
        # Tách nhãn và bounding boxes
        labels = [box[0] for box in boxes_data]
        boxes = [box[1] for box in boxes_data]

        # Tạo đối tượng target để dùng cho huấn luyện
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        # Áp dụng transform nếu có
        if self.transform:
            image = self.transform(image)

        return image, target

class VehicleDetectionTestDataset(Dataset):
    def __init__(self, test_folder, transform=None):
        self.test_folder = test_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Lấy tên file ảnh
        image_file = self.image_files[idx]
        # Đường dẫn ảnh
        image_path = os.path.join(self.test_folder, image_file)

        # Đọc ảnh
        original_img = cv2.imread(image_path)

        # Áp dụng transform nếu có
        if self.transform:
            image = self.transform(original_img)

        return image, image_file  # Trả về ảnh và tên tệp ảnh
