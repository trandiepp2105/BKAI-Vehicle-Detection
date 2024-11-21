from multiprocessing import Pool, cpu_count
import shutil
import os
import cv2
from image_processing import local_contrast_clahe, adjust_brightness_contrast, day_to_night, rotate_and_blend_roi

def adjust_images_color(input_folder, output_folder):
    """
    Tăng cường dữ liệu bằng cách làm sáng ảnh ban đêm và làm tối ảnh ban ngày.

    Parameters:
    - input_folder: Đường dẫn tới thư mục train. Trong thư mục train sẽ chứa 2 
    thư mục daytime và nighttime.
    - output_folder: Đường dẫn tới nơi muốn lưu dữ liệu sau tăng cường.
    """
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_folder, exist_ok=True)

    # Duyệt qua từng folder và xử lý
    for subfolder in ["nighttime", "daytime"]:
        subfolder_path = os.path.join(input_folder, subfolder)
        
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            
            # Copy file gốc (ảnh và .txt) sang thư mục đầu ra với tên gốc
            output_file_path = os.path.join(output_folder, filename)
            shutil.copy(file_path, output_file_path)
            
            # Xử lý nếu là file ảnh .jpg
            if filename.endswith(".jpg"):
                # Đọc ảnh
                original_img = cv2.imread(file_path)
                
                # Áp dụng hàm xử lý tương ứng
                if subfolder == "nighttime":
                    processed_img_1 = local_contrast_clahe(original_img)
                    processed_img_2 = adjust_brightness_contrast(original_img)

                else:  # daytime
                    processed_img_1 = day_to_night(original_img)
                    processed_img_2 = local_contrast_clahe(original_img)

                
                # Lưu phiên bản ảnh đã xử lý với hậu tố _edited
                output_edited_file_path = os.path.join(output_folder, filename.replace(".jpg", "_edited1.jpg"))
                cv2.imwrite(output_edited_file_path, processed_img_1)
                
                output_edited_file_path = os.path.join(output_folder, filename.replace(".jpg", "_edited2.jpg"))
                cv2.imwrite(output_edited_file_path, processed_img_2)
            
            # Xử lý file .txt: thêm hậu tố _edited
            elif filename.endswith(".txt"):
                output_edited_txt_path = os.path.join(output_folder, filename.replace(".txt", "_edited1.txt"))
                shutil.copy(file_path, output_edited_txt_path)
                output_edited_txt_path = os.path.join(output_folder, filename.replace(".txt", "_edited2.txt"))
                shutil.copy(file_path, output_edited_txt_path)


def rotate_images(input_dir, output_dir, angle=15):
    """
    Tăng cường dữ liệu bằng cách xoay bounding box 15 độ.

    Parameters:
    - input_folder: Đường dẫn tới thư mục dữ liệu sau khi được tăng cường ánh sáng.
    - output_folder: Đường dẫn tới nơi muốn lưu dữ liệu sau tăng cường.
    """
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo danh sách tham số
    tasks = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg'):
            image_path = os.path.join(input_dir, file_name)
            labels_path = os.path.join(input_dir, file_name.replace('.jpg', '.txt'))
            
            # Kiểm tra file nhãn tồn tại
            if os.path.exists(labels_path):
                tasks.append((image_path, labels_path, output_dir, angle))
                tasks.append((image_path, labels_path, output_dir, -angle))
    print("task 0: ", tasks[0])
    # Thiết lập multiprocessing với starmap
    num_workers = cpu_count()  # Số luồng CPU khả dụng
    with Pool(num_workers) as pool:
        pool.starmap(rotate_and_blend_roi, tasks)

# if __name__ == "__main__":
#     train_folder = "/train_data"
#     color_adjusted_train_data_folder = "/color_adjusted_train_data"
#     rotated_train_data_folder = "/rotated_train_data"
#     adjust_images_color(input_folder=train_folder,output_folder= color_adjusted_train_data_folder)
#     rotate_images(input_dir= color_adjusted_train_data_folder, output_dir= rotated_train_data_folder)

#     # Gộp 2 phần dữ liệu sau tăng cường vào chung 1 thư mục
#     input_dir = color_adjusted_train_data_folder
#     output_dir = rotated_train_data_folder
#     for file_name in os.listdir(input_dir):
#         source_file = os.path.join(input_dir, file_name)
#         target_file = os.path.join(output_dir, file_name)
        
#         # Chỉ copy nếu source là file
#         if os.path.isfile(source_file):
#             shutil.copy2(source_file, target_file)
#     print("Đã copy xong toàn bộ dữ liệu vào cùng 1 folder!")