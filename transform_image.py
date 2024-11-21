import torchvision.transforms as T


def transform_image(image):
    """
    Chuyển đổi hình ảnh thành tensor và thêm batch dimension.

    Args:
        image (PIL Image or NumPy array): Hình ảnh đầu vào cần chuyển đổi.

    Returns:
        torch.Tensor: Tensor hình ảnh với định dạng (C, H, W),
                      trong đó C là số kênh, H là chiều cao, và W là chiều rộng.
    """
    transform = T.Compose([
        T.ToTensor(),  # Chuyển đổi hình ảnh thành tensor,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


