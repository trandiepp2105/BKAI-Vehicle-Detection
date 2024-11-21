import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class VehicleDetectionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        Args:
            num_classes (int): Số lượng lớp (bao gồm cả lớp background).
            class_frequencies (list): Danh sách tần suất xuất hiện của từng lớp trong tập dữ liệu.
            pretrained (bool): Sử dụng mô hình pretrained hay không.
        """
        super().__init__()
        
        # Sử dụng ResNet-50 với FPN mặc định từ torchvision
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=pretrained)

        # Thay thế predictor của Faster R-CNN với số lớp cho bài toán của bạn
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    def forward(self, images, targets=None):
        """
        Forward pass qua mô hình Faster R-CNN.

        Args:
            images (list[Tensor]): Danh sách các ảnh đầu vào.
            targets (list[dict], optional): Các nhãn và box ground-truth cho training. (Default: None).

        Returns:
            Nếu có targets, trả về losses dưới dạng dict.
            Nếu không có targets (inference), trả về kết quả predictions.
        """

        if targets is not None:
            # Forward pass với targets để nhận loss từ Faster R-CNN

            losses = self.model(images, targets)

            # Áp dụng trọng số khác cho các loss còn lại
            loss_classifier_weight = 1.0  # Trọng số cho loss_classifier
            loss_box_reg_weight = 1.0   # Trọng số cho loss_box_reg
            loss_objectness_weight = 0.7  # Trọng số cho loss_objectness
            loss_rpn_box_reg_weight = 0.7  # Trọng số cho loss_rpn_box_reg

            # Tính tổng loss
            total_loss = (
                loss_classifier_weight * losses['loss_classifier'] +
                loss_box_reg_weight * losses['loss_box_reg'] +
                loss_objectness_weight * losses['loss_objectness'] +
                loss_rpn_box_reg_weight * losses['loss_rpn_box_reg']
            )

            return total_loss, losses
        else:
            # Forward pass khi inference
            return self.model(images)

    def configure_optimizer(self, lr=0.0001, weight_decay=0.0005):
        """
        Cấu hình optimizer cho việc huấn luyện.
        
        Args:
            lr (float): Learning rate cho optimizer.
            weight_decay (float): Weight decay cho optimizer.
            
        Returns:
            Optimizer đã được cấu hình cho các tham số của mô hình.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optimizer 