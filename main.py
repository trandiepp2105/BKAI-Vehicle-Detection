import os
import shutil
import torch
from torch.utils.data import DataLoader
from download_train_data import download_and_extract_zip
from data_augmentation import adjust_images_color, rotate_images
from dataset import VehicleDetectionDataset, VehicleDetectionTestDataset
from transform_image import transform_image
from vehicle_detection_model import VehicleDetectionModel
from train_model import train_model
from predict import predict_model
from utils import zip_txt_file
def main():

    # Part 1: Tải dữ liệu tập train
    print("PART 1: TẢI DỮ LIỆU TẬP TRAIN!")
    file_id = "1SjMOqzzKDtmkqmiesIyDy2zkEN7xGjbE"    #ID của file zip của tập train
    output_zip = "/data/train_data.zip"
    extract_folder = "/data/train_data"
    
    # Tải và giải nén file
    download_and_extract_zip(file_id, output_zip, extract_folder)

    # Part 2: Tải dữ liệu tập test
    print("PART 2: TẢI DỮ LIỆU TẬP TEST!")
    file_id = "1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ"    #ID của file zip của tập train
    output_zip = "/data/public_test.zip"
    extract_folder = "/data/public_test"
    
    # Tải và giải nén file
    download_and_extract_zip(file_id, output_zip, extract_folder)

    # Part 3: Tăng cường dữ liệu
    print("PART 3: TĂNG CƯỜNG DỮ LIỆU!")
    train_folder = "/data/train_data"
    color_adjusted_train_data_folder = "/data/color_adjusted_train_data"
    rotated_train_data_folder = "/data/rotated_train_data"
    adjust_images_color(input_folder=train_folder,output_folder= color_adjusted_train_data_folder)
    rotate_images(input_dir= color_adjusted_train_data_folder, output_dir= rotated_train_data_folder)

    # Gộp 2 phần dữ liệu sau tăng cường vào chung 1 thư mục
    input_dir = color_adjusted_train_data_folder
    output_dir = rotated_train_data_folder
    for file_name in os.listdir(input_dir):
        source_file = os.path.join(input_dir, file_name)
        target_file = os.path.join(output_dir, file_name)
        
        # Chỉ copy nếu source là file
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)
    print("Đã copy xong toàn bộ dữ liệu vào cùng 1 folder!")

    # Part 4: Tạo dataloader
    train_folder = '/data/rotated_train_data'  # Đường dẫn đến ảnh

    # Khởi tạo dataset và dataloader
    dataset = VehicleDetectionDataset(train_folder, transform=transform_image)

    def custom_collate_fn(batch):
        images, targets = zip(*batch)
        return images, targets

    data_loader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)

    # Part 5: Training model
    print("PART 5: TRAINING MODEL!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VehicleDetectionModel(num_classes=5)
    optimizer = model.configure_optimizer()
    output_path = '/data/fasterrcnn_finetuned_fullday'
    # Gọi hàm train_model
    train_model(model, data_loader, optimizer, device, output_path, 5)

    # Part 6: Predict
    print("PART 6: PREDICT!")
    # tạo dataloader cho tập test
    test_data_folder = "/ata/public_test/public test"
    # Tạo dataset
    test_dataset = VehicleDetectionTestDataset(test_folder=test_data_folder, transform=transform_image)

    # Tạo dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5
    model = VehicleDetectionModel(num_classes=num_classes, pretrained=True)
    # Đường dẫn đến file đã lưu trong thư mục working của Kaggle
    model_path = '/data/fasterrcnn_finetuned_fullday.pth'
    # Tải trọng số vào mô hình
    model.load_state_dict(torch.load(model_path))
    predict_output_path = "/data/predict.txt"
    predict_model(model, test_dataloader, device, threshold = 0.02,output_file= predict_output_path)

    # zip file predict
    zip_path = "/data/predict.zip"
    zip_txt_file(input_file=predict_output_path, output_zip=zip_path)
if __name__ == "__main__":
    main()