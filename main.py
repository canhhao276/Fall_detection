
import sys
import argparse
from pathlib import Path

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đưa thư mục gốc dự án vào PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(
        description="Hệ thống AI Phát hiện Té ngã qua Video & Khung xương MediaPipe Pose",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ câu lệnh:
  python main.py --mode detect --source 0                                      # Chạy với Webcam / DroidCam
  python main.py --mode detect --source "data/Home_01/Home_01/Videos/video (15).avi" # Chạy với Video file
  python main.py --mode test-telegram                                          # Test kết nối Telegram Bot
  python main.py --mode extract                                                # Trích xuất dữ liệu Le2i
  python main.py --mode train                                                  # Huấn luyện lại mô hình LSTM
        """
    )

    parser.add_argument(
        "--mode", 
        choices=["detect", "test-telegram", "extract", "train"], 
        default="detect",
        help="Chế độ thực thi: 'detect' (Mặc định: Nhận diện), 'test-telegram', 'extract', 'train'"
    )
    parser.add_argument(
        "--source", 
        default="0", 
        help="Nguồn video: '0' (Webcam/DroidCam), URL WiFi hoặc đường dẫn file video (.avi/.mp4)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.8, 
        help="Ngưỡng xác suất kích hoạt cảnh báo té ngã (mặc định: 0.8 / 80%)"
    )
    parser.add_argument(
        "--cooldown", 
        type=float, 
        default=15.0, 
        help="Thời gian chờ chống spam tin nhắn Telegram (giây, mặc định: 15s)"
    )
    parser.add_argument(
        "--no-telegram", 
        action="store_true", 
        help="Tắt tính năng gửi cảnh báo qua Telegram Bot"
    )

    args = parser.parse_args()

    if args.mode == "detect":
        from src.realtime_detect import run_realtime_detection
        source = int(args.source) if str(args.source).isdigit() else args.source
        run_realtime_detection(
            source=source, 
            threshold=args.threshold, 
            enable_telegram=not args.no_telegram, 
            cooldown=args.cooldown
        )

    elif args.mode == "test-telegram":
        from src.telegram_alert import TelegramAlertNotifier
        print("="*60)
        print(" [KIỂM TRA KẾT NỐI TELEGRAM BOT CẢNH BÁO TÉ NGÃ] ")
        print("="*60)
        notifier = TelegramAlertNotifier()
        notifier.test_connection()

    elif args.mode == "extract":
        from src.extract_pose import main as run_extract
        run_extract()

    elif args.mode == "train":
        from src.train_lstm import main as run_train
        run_train()

if __name__ == "__main__":
    main()
