import torch
from utils import bbox_to_yolo
# from tqdm import tqdm

def predict_model(model, data_loader, device, threshold = 0.6, output_file='predict.txt'):
    model.to(device)
    model.eval()  # Đặt model ở chế độ eval để dự đoán

    with open(output_file, 'w') as f:
        for batch in data_loader:
            images, image_names = batch
            
            # Đảm bảo images là danh sách tensor
            images = list(image.to(device) for image in images)
            
            # Dự đoán
            with torch.no_grad():
                predictions = model(images)

            # Xử lý từng ảnh và từng dự đoán
            for i, prediction in enumerate(predictions):
                image_name = image_names[i]
                
                img_height, img_width = images[i].shape[1], images[i].shape[2]
                
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy() - 1  # class_id trừ đi 1
                scores = prediction['scores'].cpu().numpy()
                
                # Lặp qua mỗi đối tượng dự đoán trong ảnh
                for j in range(len(boxes)):
                    class_id = labels[j]
                    score = scores[j]
                    bbox = boxes[j]
                    x_center, y_center, width, height = bbox_to_yolo(bbox, img_width, img_height)

                    # Ghi vào file với định dạng yêu cầu
                    if score >= threshold:
                        f.write(f"{image_name} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.3f}\n")
                    else:
                        continue
    print(f"Predictions saved to {output_file}")
