# 🚨 HỆ THỐNG PHÁT HIỆN TÉ NGÃ & CẢNH BÁO TELEGRAM

## 📁 1. Cấu trúc thư mục

```text
fall-detection/
├── configs/          # Cấu hình Telegram (Token, Chat ID)
├── data/             # Dữ liệu video & đặc trưng đã trích xuất
├── models/           # Mô hình LSTM (fall_model.keras) & MediaPipe Task
├── reports/          # Biểu đồ đánh giá huấn luyện & Confusion Matrix
├── src/              # Mã nguồn các module (extract, train, detect, alert)
├── main.py           # File chạy chính của chương trình
└── requirements.txt  # Thư viện phụ thuộc
```

---

## 🚀 2. Hướng dẫn cách chạy

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình Telegram Bot
Tạo file `configs/telegram_config.json` (hoặc sao chép từ `configs/telegram_config.example.json`) và điền Token + Chat ID của bạn:
```json
{
    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "chat_id": "YOUR_TELEGRAM_CHAT_ID",
    "cooldown_seconds": 15,
    "enabled": true
}
```

Kiểm tra kết nối Bot:
```bash
python main.py --mode test-telegram
```

### Bước 3: Chạy nhận diện té ngã

#### 🎥 Chạy trực tiếp qua Webcam / DroidCam:
```bash
python main.py --mode detect --source 0
```

#### 🎬 Chạy kiểm thử trên file Video:
```bash
python main.py --mode detect --source "data/Home_01/Home_01/Videos/video (15).avi"
```

#### ⌨️ Phím tắt điều khiển:
- **`q`** hoặc **`ESC`**: Thoát chương trình.
- **`r`**: Reset bộ đệm và xóa cảnh báo.
