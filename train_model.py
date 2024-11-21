import torch
from vehicle_detection_model import VehicleDetectionModel
from torch.utils.data import DataLoader
from dataset import VehicleDetectionDataset
from transform_image import transform_image
def train_model(model, data_loader, optimizer, device, output_path, num_epochs=12):
    """
    Huấn luyện mô hình với dữ liệu và lưu trạng thái mô hình sau mỗi epoch.

    Args:
        model (torch.nn.Module): Mô hình Faster R-CNN.
        data_loader (torch.utils.data.DataLoader): DataLoader chứa dữ liệu huấn luyện.
        optimizer (torch.optim.Optimizer): Optimizer để cập nhật trọng số.
        device (torch.device): Thiết bị để huấn luyện (CPU hoặc GPU).
        output_path (str): Đường dẫn lưu mô hình cuối cùng.
        num_epochs (int, optional): Số lượng epoch. Mặc định là 12.
    """
    model.to(device)
    model.train()  # Đặt model ở chế độ train
    
    for epoch in range(num_epochs):
        epoch_loss = 0  # Tổng loss của từng epoch

        for batch_idx, batch in enumerate(data_loader, start=1):
            images, targets = batch
            images = list(image.to(device) for image in images)

            # Chuyển dữ liệu sang thiết bị (CPU/GPU)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: Tính toán loss từ model
            total_loss, loss_dict = model(images, targets)

            # Reset gradient
            optimizer.zero_grad()

            # Backward pass: Tính gradient và cập nhật trọng số
            total_loss.backward()
            optimizer.step()

            # Cộng dồn loss cho epoch hiện tại
            epoch_loss += total_loss.item()

            # Hiển thị tiến trình
            if batch_idx % 10 == 0:  # Hiển thị mỗi 10 batch
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], "
                    f"Loss: {total_loss.item():.4f}"
                )

        # Kết thúc một epoch, in tổng loss
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed. Total Loss: {epoch_loss:.4f}")

        # Lưu trạng thái mô hình sau mỗi epoch
        epoch_model_path = f"{output_path}_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model for Epoch {epoch+1} saved to {epoch_model_path}.")

    # Lưu trạng thái mô hình cuối cùng
    final_model_path = f"{output_path}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}.")

if __name__ == "__main__":

    train_folder = '/rotated_train_data'  # Đường dẫn đến ảnh

    # Khởi tạo dataset và dataloader
    dataset = VehicleDetectionDataset(train_folder, transform=transform_image)

    def custom_collate_fn(batch):
        images, targets = zip(*batch)
        return images, targets

    data_loader = DataLoader(dataset, batch_size=8, collate_fn=custom_collate_fn)

    # Khởi tạo model, optimizer và dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_frequencies = [65.6, 18.2, 4.3, 11.9]
    model = VehicleDetectionModel(num_classes=5)
    optimizer = model.configure_optimizer()
    output_path = 'fasterrcnn_finetuned_fullday'
    # Gọi hàm train_model
    train_model(model, data_loader, optimizer, device, output_path, 5)
     