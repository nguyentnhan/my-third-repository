import os
import sys
import multiprocessing

# CHIẾN THUẬT 1: Thay vì 'spawn', sử dụng 'fork' (chỉ hoạt động tốt trên Linux/WSL)
# Nếu bạn chạy trên Windows, bắt buộc phải giữ 'spawn' nhưng áp dụng CHIẾN THUẬT 2 bên dưới.
if sys.platform != 'win32':
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass
else:
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
import torch
import torch.nn as nn
import os
from torch.cuda.amp import GradScaler, autocast
from CNN_Resnet import ResNet18
from deal_ffcv import dealwith,export_to_ffcv
import time
import torch._inductor.config as config
config.max_autotune = False
# Cấu hình chung để dùng ở nhiề
WEIGHTS_PATH = "model_weights_tam.pth"


def get_model(device):
    model = ResNet18().to(device)
    if os.path.exists(WEIGHTS_PATH):
        checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        
        # Kiểm tra NaN trong trọng số
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"CẢNH BÁO: Trọng số lớp {name} chứa NaN!")
    return model

def train_model(resume=False):
    # ... (Khởi tạo model, optimizer, scaler như cũ) ...
    epochs = 50
    num = 50000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    model = torch.compile(model, backend="cudagraphs")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    # tạo ra train.beton và test.beton ( chỉ tạo ra trong 1 lần )
    _ , le = export_to_ffcv("cifar/train", "cifar/trainLabels.csv", num)
    train_loader, test_loader = dealwith(128)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        # Khởi tạo total_loss là một Tensor 0 trên GPU
        total_loss = torch.tensor(0.0, device=device)
        
        for i, data in enumerate(train_loader):
            torch.compiler.cudagraph_mark_step_begin()
            images = data[0]
            labels = data[1].flatten().long()
            optimizer.zero_grad(set_to_none=True)

            # Dùng bfloat16 nguyên bản - KHÔNG cần GradScaler
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print(f" Phát hiện NaN tại batch {i}! Dừng huấn luyện.")
                return
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            
            # Clip gradient vẫn hoạt động bình thường mượt mà không cần unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Cộng dồn loss trực tiếp trên GPU an toàn
            total_loss += loss.detach()
            # steps += 1

            # Chỉ gọi .item() khi thực sự cần in ra (giảm thiểu ép CPU-GPU đồng bộ)
        avg_loss_batch = total_loss / (i+1)
        print(f"Epoch {epoch+1} | Loss: {avg_loss_batch.item():.4f}")
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} hoàn thành trong: {epoch_duration:.2f} giây")
        total_loss.zero_() # Reset Tensor về 0 ngay trên GPU

        # Cuối epoch, tính toán một lần cho scheduler
        # Lưu ý: total_loss lúc này chỉ chứa tổng của các batch lẻ còn lại sau lần reset cuối
        # Để chính xác hơn, bạn có thể tạo một biến phụ lưu epoch_loss riêng
        avg_epoch_loss = avg_loss_batch.item() # Lấy giá trị gần nhất để update scheduler
        scheduler.step(avg_epoch_loss)
        
        # Lưu trọng số mô hình dạng chuẩn raw_state_dict (uncompiled)
        # torch.compile thêm tiền tố '_orig_mod.', ta lấy model._orig_mod nếu có, hoặc dùng luôn model nếu dùng bản mới
        raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        torch.save(raw_model.state_dict(), WEIGHTS_PATH)
        
        lr = optimizer.param_groups[0]['lr']
        if lr < 1e-7: 
            print("Chạm giới hạn Learning Rate. Dừng sớm.")
            break

    print("💾 Đã hoàn tất huấn luyện.")
    evaluate(model, test_loader, device)

def evaluate(model, loader, device):
    """Hàm đánh giá độ chính xác chung"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for load in loader:
            images = load[0] # FFCV pipeline đã đẩy lên GPU từ trước
            labels = load[1].flatten().long()
            # squeeze loại bỏ các chiêu = 1

            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # trả về chỉ số vị trí của giá trị max trong mỗi hàng 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0:
        print(" Cảnh báo: test_loader không trả về bất kỳ dữ liệu nào! Kiểm tra lại kích thước batch_size.")
        return 0.0
    accuracy = 100 * correct / total
    print(f" Accuracy: {accuracy:.2f}%")
    # has_zero = (outputs == 0).any()
    # print(f"Có giá trị 0 trong output không? {has_zero.item()}")
    return accuracy, outputs

def use_model(loader):
    """Hàm dùng để dự đoán dữ liệu mới (Sử dụng trọng số đã lưu)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(" Đang dự đoán dữ liệu...")
    evaluate(model, loader, device)

def predict():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    train_loader, test_loader, _ = dealwith("train", "trainLabels.csv", 5, 64)
    use_model(train_loader)

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    # train_loader, test_loader, _ = dealwith("train", "trainLabels.csv", 10000, 64)
    # use_model(train_loader)
    # Chạy huấn luyện
    train_model(resume=False)

# việc sử dụng scaler đã khiến mô hình mắc phải loss NaN ( hoặc cũng có thể do lr cao nữa (0.001))
# vì sgd ko điều chỉnh lr nên nó dễ nhảy qua hố sang mức loss cao hơn bên kia.
# tôi sẽ thử nâng lr lên 0.001 trong 50 epoch tiếp
# với learning rate đầu bằng 0.008, khi chạy mô hình lại lần nữa
# thì lr sẽ được reset lại và tăng mạnh
# lỗi Kịch bản : "Vòng xoáy tử thần" (The Death Spiral) vòng lặp phản hồi dương (positive feedback loop)
# việc tăng trọng số tức thì sẽ đồng thời tăng bước nhảy của tất cả trọng số, 
# khiến cho loss lại tăng nữa, khiến cho tất cả trọng số lại tăng mạnh đồng thời tiếp, 
# và trước khi scheduler giảm lr về mức ổn thì grad đã bùng nổ.
# dù đã giảm lr = 0.000008 nhưng vẫn bị NaN loss có thể là vì:
# NaN lúc này không phải do mô hình "ngu đi", mà do nó "quá tự tin" dẫn đến các con số vượt quá khả năng lưu trữ của kiểu dữ liệu bạn đang dùng
# kết quả sau 100 - 50 - vài epoch (2 lần chạy) -> 96% test, 96% train