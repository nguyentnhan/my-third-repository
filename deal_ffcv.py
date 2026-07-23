import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

# Import chuẩn từ FFCV
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
# 1. Các phép biến đổi cơ bản, chuyển đổi định dạng và ép lên GPU
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    ToTorchImage,
    NormalizeImage,
    RandomHorizontalFlip
)
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder,   RandomResizedCropRGBImageDecoder


# Định nghĩa lại Dataset tối giản (KHÔNG có transform) cho FFCV Writer
class FFCVPrepDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB') 
        label = self.labels[idx]
        # FFCV yêu cầu nhãn kiểu int chuẩn, ảnh dạng PIL hoặc numpy
        return img, int(label) 

def export_to_ffcv(path, path_csv, num):
    # (Phần đọc CSV, quét os.scandir và train_test_split giữ nguyên như code của bạn)
    # ... [Giữ nguyên đoạn code xử lý chuỗi của bạn ở đây] ...
    df = pd.read_csv(path_csv, index_col=0)
    df.index = df.index.astype(str)

    image_paths, y_data = [], [] # Chỉ lưu PATH
    dem = 0
    
    with os.scandir(path) as entries:
        for entry in entries:
            if dem >= num: break
            file_id = os.path.splitext(entry.name)[0]
            if str(file_id) in df.index:
                image_paths.append(entry.path) # Lưu đường dẫn, không phải mảng ảnh
                y_data.append(df.loc[file_id].iloc[0])
                dem += 1
    
    # Chia tập dữ liệu dựa trên đường dẫn
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, y_data, test_size=0.2, random_state=42, stratify=y_data
    )
    # Tạo đối tượng dataset thô
    train_ds_raw = FFCVPrepDataset(X_train, y_train)
    test_ds_raw = FFCVPrepDataset(X_test, y_test)
    
    # Đóng gói tập TRAIN thành file train.beton
    writer_train = DatasetWriter("train.beton", {
        'image': RGBImageField(write_mode='raw', max_resolution=32), # CIFAR có size 32x32
        'label': IntField()
    })
    writer_train.from_indexed_dataset(train_ds_raw)
    
    # Đóng gói tập TEST thành file test.beton
    writer_test = DatasetWriter("test.beton", {
        'image': RGBImageField(write_mode='raw', max_resolution=32),
        'label': IntField()
    })
    writer_test.from_indexed_dataset(test_ds_raw)
    
    return len(le.classes_), le
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, RandomHorizontalFlip
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder, RandomResizedCropRGBImageDecoder

# TỰ ĐỊNH NGHĨA LỚP CHUẨN HÓA TRÊN GPU (Thay thế hoàn toàn NormalizeImage của FFCV)
class GPUNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # Chuyển mean, std thành Tensor GPU dạng float32 để tính toán siêu tốc
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda().view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).cuda().view(3, 1, 1)

    def forward(self, x):
        # x lúc này đã là Tensor float32 trên GPU (dải 0-255), ta chỉ việc tính toán:
        return (x - self.mean) / self.std

def dealwith(batch_size=64):
    # Định nghĩa Mean/Std dải 0-255 theo chuẩn ImageNet
    mean = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0]
    std = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0]
    
    this_device = torch.device("cuda:0")
    gpu_normalizer = GPUNormalize(mean, std)
    
    # Pipeline TRAIN
    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((32, 32)),
        RandomHorizontalFlip(),
        ToTensor(),              
        ToTorchImage(),          
        ToDevice(this_device, non_blocking=True), 
        gpu_normalizer # Gọi lớp chuẩn hóa tùy hợp chạy trực tiếp trên GPU ở đây!
    ]
    
    # Pipeline TEST
    test_image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToTorchImage(),          
        ToDevice(this_device, non_blocking=True), 
        gpu_normalizer # Áp dụng tương tự cho tập test
    ]
    
    label_pipeline = [
        IntDecoder(),            
        ToTensor(), 
        ToDevice(this_device, non_blocking=True)
    ]

    train_loader = Loader("train.beton",
                          batch_size=batch_size,
                          num_workers=4,
                          order=OrderOption.RANDOM,
                          drop_last=True, # Giữ nguyên để tránh lỗi kích thước CUDA Graphs
                          pipelines={'image': train_image_pipeline, 'label': label_pipeline})
                          
    test_loader = Loader("test.beton",
                         batch_size=batch_size,
                         num_workers=4,
                         order=OrderOption.SEQUENTIAL,
                         pipelines={'image': test_image_pipeline, 'label': label_pipeline})
                         
    return train_loader, test_loader