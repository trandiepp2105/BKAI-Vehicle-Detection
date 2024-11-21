import gdown
import zipfile
import os

def download_and_extract_zip(file_id, output_zip="/train_data.zip", extract_folder="/train_data"):
    """
    Tải một file ZIP từ Google Drive và giải nén nó.
    
    Args:
        file_url (str): URL của file trên Google Drive.
        output_zip (str): Tên file ZIP sau khi tải xuống.
        extract_folder (str): Thư mục để giải nén nội dung file ZIP.
    """
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    os.makedirs(extract_folder, exist_ok=True)
    # Chuyển đổi URL thành dạng tải xuống
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    # Tải file ZIP từ Google Drive
    print("Đang tải file từ Google Drive...")
    gdown.download(download_url, output_zip, quiet=False)
    print(f"File đã tải xuống: {output_zip}")
    
    # Kiểm tra và giải nén file ZIP
    if zipfile.is_zipfile(output_zip):
        print("Đang giải nén file...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"File đã được giải nén vào thư mục: {extract_folder}")
        os.remove(output_zip)
        print(f"File ZIP {output_zip} đã được xóa.")
    else:
        print("File tải xuống không phải là file ZIP hợp lệ.")

if __name__ == "__main__":
    print("PART 2: TẢI DỮ LIỆU TẬP TEST!")
    file_id = "1BQvwhSoeDm-caCImtlbcAMzhI8MDsrCZ"    #ID của file zip của tập train
    output_zip = "/data/public_test.zip"
    extract_folder = "/data/public_test"
    
    # Tải và giải nén file
    download_and_extract_zip(file_id, output_zip, extract_folder)
