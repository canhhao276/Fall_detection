"""
MÔ-ĐUN CẢNH BÁO TÉ NGÃ TỰ ĐỘNG QUA TELEGRAM BOT (ALERT MODULE)
----------------------------------------------------------------------
Chức năng:
1. Tự động chụp khung hình ảnh tại thời điểm phát hiện người té ngã.
2. Gửi ảnh chụp kèm tin nhắn cảnh báo khẩn cấp (Emergency Alert) tới Telegram Bot.
3. Cơ chế Cooldown Timer (thời gian chờ chống spam, mặc định 15s).
4. Cơ chế Đa luồng bất đồng bộ (Asynchronous Background Threading).
5. Hỗ trợ cấu hình tiện lợi qua file 'configs/telegram_config.json'.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
import cv2
import requests

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "telegram_config.json"

class TelegramAlertNotifier:
    """
    Quản lý kết nối và gửi thông báo khẩn cấp kèm ảnh chụp tới Telegram Bot.
    """
    def __init__(self, bot_token=None, chat_id=None, cooldown_seconds=15, enabled=True, config_path=None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.last_alert_time = 0.0

        # Nếu chưa truyền token, tự động nạp từ file cấu hình
        if not self.bot_token or not self.chat_id:
            self.load_config()

        # Kiểm tra tính hợp lệ của cấu hình
        self.is_configured = bool(
            self.bot_token and 
            self.chat_id and 
            "YOUR_TELEGRAM" not in str(self.bot_token) and 
            "YOUR_TELEGRAM" not in str(self.chat_id)
        )

        if self.is_configured and self.enabled:
            print(f"[TELEGRAM] Đã kích hoạt cảnh báo Telegram Bot (Cooldown: {self.cooldown_seconds}s).")
        else:
            print("[TELEGRAM] Telegram Bot chưa cấu hình hoặc bị tắt.")

    def load_config(self):
        """Nạp thông tin cấu hình từ file configs/telegram_config.json."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.bot_token = config.get("bot_token", self.bot_token)
                    self.chat_id = config.get("chat_id", self.chat_id)
                    self.cooldown_seconds = config.get("cooldown_seconds", self.cooldown_seconds)
                    self.enabled = config.get("enabled", self.enabled)
            except Exception as e:
                print(f"[TELEGRAM] Không thể đọc file cấu hình '{self.config_path}': {e}")

    def send_fall_alert_async(self, frame_bgr, fall_prob: float, camera_source: str = "Camera"):
        """
        Gọi gửi cảnh báo ngầm trong luồng riêng biệt (Non-blocking Background Thread).
        """
        if not self.enabled or not self.is_configured:
            return

        current_time = time.time()
        elapsed = current_time - self.last_alert_time

        # Kiểm tra Cooldown chống spam
        if elapsed < self.cooldown_seconds:
            remaining = self.cooldown_seconds - elapsed
            print(f"[TELEGRAM] Bỏ qua gửi tin (Đang trong Cooldown, còn {remaining:.1f}s).")
            return

        self.last_alert_time = current_time
        frame_copy = frame_bgr.copy()

        worker = threading.Thread(
            target=self._send_worker,
            args=(frame_copy, fall_prob, camera_source),
            daemon=True
        )
        worker.start()

    def _get_chat_ids(self):
        """Lấy danh sách Chat ID (hỗ trợ cả 1 ID hoặc danh sách nhiều người/nhóm)."""
        if isinstance(self.chat_id, list):
            return [str(cid).strip() for cid in self.chat_id if str(cid).strip()]
        elif isinstance(self.chat_id, str) and "," in self.chat_id:
            return [cid.strip() for cid in self.chat_id.split(",") if cid.strip()]
        elif self.chat_id:
            return [str(self.chat_id).strip()]
        return []

    def _send_worker(self, frame_bgr, fall_prob: float, camera_source: str):
        """
        Tiến trình ngầm: Nén ảnh JPEG và gửi tới Telegram Bot API.
        """
        chat_ids = self._get_chat_ids()
        if not chat_ids:
            return

        try:
            now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            caption = (
                "🚨 <b>CẢNH BÁO: PHÁT HIỆN SỰ CỐ TÉ NGÃ!</b> 🚨\n\n"
                f"⏰ <b>Thời gian:</b> <code>{now_str}</code>\n"
                f"📊 <b>Độ tin cậy AI:</b> <b>{fall_prob * 100:.1f}%</b>\n"
                f"📹 <b>Nguồn quan sát:</b> <code>{camera_source}</code>\n"
                f"⚠️ <b>Trạng thái:</b> <i>Phát hiện người ngã và đang nằm trên sàn. Vui lòng kiểm tra khẩn cấp!</i>"
            )

            success, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                print("[TELEGRAM LỖI] Không thể mã hóa ảnh JPEG.")
                return

            img_bytes = buffer.tobytes()
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

            for cid in chat_ids:
                try:
                    files = {"photo": ("fall_snapshot.jpg", img_bytes, "image/jpeg")}
                    data = {
                        "chat_id": cid,
                        "caption": caption,
                        "parse_mode": "HTML"
                    }
                    response = requests.post(url, files=files, data=data, timeout=10)
                    if response.status_code == 200:
                        print(f"[TELEGRAM THÀNH CÔNG] Đã gửi ảnh & cảnh báo khẩn cấp tới Chat ID: {cid}!")
                    else:
                        print(f"[TELEGRAM LỖI] Gửi tin tới {cid} thất bại ({response.status_code}): {response.text}")
                except Exception as e:
                    print(f"[TELEGRAM LỖI] Không thể gửi tới {cid}: {e}")

        except Exception as e:
            print(f"[TELEGRAM LỖI] Lỗi trong quá trình gửi cảnh báo: {e}")

    def test_connection(self) -> bool:
        """Kiểm tra kết nối và gửi tin nhắn thử nghiệm tới Telegram."""
        if not self.is_configured:
            print(f"[TELEGRAM] Chưa điền Token và Chat ID vào file '{self.config_path}'.")
            return False

        chat_ids = self._get_chat_ids()
        print(f"[TELEGRAM] Đang kiểm tra kết nối tới {len(chat_ids)} người nhận/nhóm qua Telegram API...")
        all_ok = True
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        for cid in chat_ids:
            try:
                data = {
                    "chat_id": cid,
                    "text": f"✅ <b>HỆ THỐNG PHÁT HIỆN TÉ NGÃ ĐÃ SẴN SÀNG!</b>\n\n🕒 Thời gian kết nối: {now_str}\nTrạng thái: Trực tuyến và sẵn sàng gửi cảnh báo.",
                    "parse_mode": "HTML"
                }
                res = requests.post(url, data=data, timeout=10)
                if res.status_code == 200:
                    print(f"[TELEGRAM THÀNH CÔNG] Đã gửi tin nhắn test thành công tới Chat ID: {cid}!")
                else:
                    print(f"[TELEGRAM LỖI] Chat ID {cid} không hợp lệ: {res.text}")
                    all_ok = False
            except Exception as e:
                print(f"[TELEGRAM LỖI] Lỗi kết nối tới {cid}: {e}")
                all_ok = False

        return all_ok

def main():
    print("="*60)
    print(" [KIỂM TRA KẾT NỐI TELEGRAM BOT CẢNH BÁO TÉ NGÃ] ")
    print("="*60)
    notifier = TelegramAlertNotifier()
    notifier.test_connection()

if __name__ == "__main__":
    main()
