FROM python:3.10

WORKDIR /vehicledetection

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ./ ./

ENV PYTHONPATH=/vehicledetection

COPY entrypoint.sh /entrypoint.sh

# Đảm bảo script có quyền thực thi
RUN chmod +x /entrypoint.sh

# Xóa ký tự \r trong file entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

# Thiết lập script entrypoint để chạy khi container khởi động
ENTRYPOINT ["/entrypoint.sh"]
