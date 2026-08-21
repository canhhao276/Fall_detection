
import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import cv2
import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tắt bớt log không cần thiết của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Import Telegram Alert Module từ thư mục src
from src.telegram_alert import TelegramAlertNotifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"

MODEL_PATH = MODELS_DIR / "fall_model.keras"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
TELEGRAM_CONFIG_PATH = CONFIGS_DIR / "telegram_config.json"

SEQUENCE_LENGTH = 30     # Số frames trong cửa sổ trượt
FEATURE_DIM = 132        # 33 keypoints * 4
FALL_THRESHOLD = 0.8     # Ngưỡng xác suất kích hoạt cảnh báo té ngã (0.8)
DEFAULT_COOLDOWN = 15.0  # Thời gian chờ chống spam tin nhắn Telegram (giây)

# Danh sách 35 cặp kết nối các khớp xương chuẩn của MediaPipe Pose
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# ======================= LỚP TRÍCH XUẤT POSE =======================
class RealtimePoseDetector:
    """Xử lý trích xuất keypoints và vẽ khung xương theo thời gian thực."""
    def __init__(self):
        self.use_tasks_api = False
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                self.mp_pose = mp.solutions.pose
                self.detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_tasks_api = False
            else:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    running_mode=vision.RunningMode.IMAGE
                )
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.mp = mp
                self.use_tasks_api = True
        except Exception:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp
            base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.mp = mp
            self.use_tasks_api = True

    def process_frame(self, frame_bgr: np.ndarray):
        """Trích xuất 132 keypoints và vẽ khung xương trực tiếp lên frame."""
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated_frame = frame_bgr.copy()

        keypoints = np.zeros(FEATURE_DIM, dtype=np.float32)
        landmarks_data = []

        if not self.use_tasks_api:
            frame_rgb.flags.writeable = False
            results = self.detector.process(frame_rgb)
            if results.pose_landmarks:
                kp = []
                for lm in results.pose_landmarks.landmark:
                    kp.extend([lm.x, lm.y, lm.z, lm.visibility])
                    landmarks_data.append((int(lm.x * w), int(lm.y * h), lm.visibility))
                keypoints = np.array(kp, dtype=np.float32)
                self.draw_skeleton(annotated_frame, landmarks_data)
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_image)
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                first_person = results.pose_landmarks[0]
                kp = []
                for lm in first_person:
                    v = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    kp.extend([lm.x, lm.y, lm.z, v])
                    landmarks_data.append((int(lm.x * w), int(lm.y * h), v))
                keypoints = np.array(kp, dtype=np.float32)
                self.draw_skeleton(annotated_frame, landmarks_data)

        return keypoints, landmarks_data, annotated_frame

    def draw_skeleton(self, frame: np.ndarray, landmarks_data: list):
        """Vẽ các đoạn xương và khớp cơ thể."""
        if not landmarks_data or len(landmarks_data) < 33:
            return

        for start_idx, end_idx in POSE_CONNECTIONS:
            pt1 = (landmarks_data[start_idx][0], landmarks_data[start_idx][1])
            pt2 = (landmarks_data[end_idx][0], landmarks_data[end_idx][1])
            v1 = landmarks_data[start_idx][2]
            v2 = landmarks_data[end_idx][2]

            if v1 > 0.4 and v2 > 0.4:
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

        for x, y, v in landmarks_data:
            if v > 0.4:
                cv2.circle(frame, (x, y), 4, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

# ======================= HÀM PHÂN TÍCH TƯ THẾ CƠ THỂ =======================
def check_body_posture(landmarks_data):
    """Phân tích góc nghiêng thân mình và tỉ lệ khung bao cơ thể."""
    if not landmarks_data or len(landmarks_data) < 33:
        return True, False, 0.0

    ls = landmarks_data[11]  # Vai trái
    rs = landmarks_data[12]  # Vai phải
    lh = landmarks_data[23]  # Hông trái
    rh = landmarks_data[24]  # Hông phải

    shoulder_x = (ls[0] + rs[0]) / 2.0
    shoulder_y = (ls[1] + rs[1]) / 2.0
    hip_x = (lh[0] + rh[0]) / 2.0
    hip_y = (lh[1] + rh[1]) / 2.0

    dx = abs(shoulder_x - hip_x)
    dy = abs(shoulder_y - hip_y)
    
    torso_angle = float(np.degrees(np.arctan2(dy, dx + 1e-6)))

    valid_pts = [(pt[0], pt[1]) for pt in landmarks_data if pt[2] > 0.3]
    if len(valid_pts) >= 6:
        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]
        bbox_w = max(xs) - min(xs) + 1e-6
        bbox_h = max(ys) - min(ys) + 1e-6
        aspect_ratio = bbox_h / bbox_w
    else:
        aspect_ratio = 0.5

    is_lying = (torso_angle < 45.0) or (aspect_ratio < 1.0)
    is_standing = (torso_angle > 60.0) and (aspect_ratio > 1.25)

    return is_lying, is_standing, torso_angle

# ======================= GIAO DIỆN HUD RESPONSIVE =======================
def draw_hud(frame: np.ndarray, status: str, fall_prob: float, fps: float, buffer_len: int, threshold: float, fall_elapsed: float = 0.0, is_fall_active: bool = False, tg_active: bool = False):
    """Vẽ giao diện hiển thị chuyên nghiệp (HUD Banner, Thanh xác suất, FPS, Trạng thái, Telegram)."""
    h, w, _ = frame.shape
    scale = max(0.45, min(w / 640.0, 1.0))

    if is_fall_active:
        banner_color = (30, 30, 220)       # Đỏ tươi (Báo động ngã)
        status_text = f"FALL! PERSON DOWN ({fall_elapsed:.0f}s)"
    elif fall_prob >= 0.5:
        banner_color = (0, 140, 255)       # Cam (Cảnh báo tiềm ẩn)
        status_text = "WARNING"
    else:
        banner_color = (40, 180, 50)       # Xanh lá (Bình thường)
        status_text = "NORMAL"

    # Viền cảnh báo đỏ toàn màn hình
    if is_fall_active:
        border_thickness = max(2, int(4 * scale))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), border_thickness)

    # Khung HUD chính
    hud_w = int(240 * scale)
    hud_h = int(75 * scale)
    pad = int(8 * scale)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + hud_w, pad + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (pad, pad), (pad + hud_w, pad + hud_h), banner_color, max(1, int(1.5 * scale)))

    # Text trạng thái
    font_scale = 0.42 * scale
    cv2.putText(frame, f"STATUS: {status_text}", (pad + int(8 * scale), pad + int(20 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, banner_color, max(1, int(1.5 * scale)), cv2.LINE_AA)

    # Thanh hiển thị xác suất
    prob_text = f"Prob: {fall_prob * 100:.0f}%"
    cv2.putText(frame, prob_text, (pad + int(8 * scale), pad + int(42 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, (240, 240, 240), 1, cv2.LINE_AA)

    bar_x = pad + int(85 * scale)
    bar_y = pad + int(32 * scale)
    bar_w = int(140 * scale)
    bar_h = int(10 * scale)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * np.clip(fall_prob, 0.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), banner_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    thresh_x = int(bar_x + bar_w * threshold)
    cv2.line(frame, (thresh_x, bar_y - 1), (thresh_x, bar_y + bar_h + 1), (0, 255, 255), 1)

    buffer_str = f"Buf: {buffer_len}/{SEQUENCE_LENGTH}" if buffer_len < SEQUENCE_LENGTH else "Buf: OK"
    tg_str = "TG: ON" if tg_active else "TG: OFF"
    info_text = f"FPS: {fps:.1f} | {buffer_str} | {tg_str}"
    cv2.putText(frame, info_text, (pad + int(8 * scale), pad + int(64 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, (180, 180, 180), 1, cv2.LINE_AA)

# ======================= HÀM MỞ NGUỒN VIDEO THÔNG MINH =======================
def open_video_source(source):
    """Mở nguồn video: Ưu tiên File video -> URL IP -> Camera Index."""
    if isinstance(source, str) and os.path.exists(source):
        print(f"[INFO] Đang mở File Video: '{source}' ...")
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"[THÀNH CÔNG] Đã mở file video '{source}' (Tổng {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames)!")
            return cap

    if isinstance(source, str):
        if source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://"):
            if ":4747" in source and not source.endswith("/video") and not source.endswith("/video.force"):
                source = source.rstrip("/") + "/video"
            print(f"[INFO] Đang kết nối tới DroidCam IP Stream: {source} ...")
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print("[THÀNH CÔNG] Đã kết nối DroidCam IP Stream!")
                    return cap
                cap.release()
        elif ":" in source:
            url = f"http://{source}/video"
            print(f"[INFO] Đang kết nối tới DroidCam IP: {url} ...")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print("[THÀNH CÔNG] Đã kết nối DroidCam IP Stream!")
                    return cap
                cap.release()

    cam_id = 0
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cam_id = int(source)

    print(f"[INFO] Đang mở Camera ID: [{cam_id}] ...")
    cap = cv2.VideoCapture(cam_id)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"[THÀNH CÔNG] Đã mở Camera ID: {cam_id}!")
            return cap
        cap.release()

    if sys.platform == "win32":
        try:
            cap_dshow = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cap_dshow.isOpened():
                ret, _ = cap_dshow.read()
                if ret:
                    print(f"[THÀNH CÔNG] Đã mở Camera ID qua DirectShow: {cam_id}!")
                    return cap_dshow
                cap_dshow.release()
        except Exception:
            pass

    return None

# ======================= VÒNG LẶP SUY LUẬN CHÍNH =======================
def run_realtime_detection(source=0, threshold=FALL_THRESHOLD, enable_telegram=True, cooldown=DEFAULT_COOLDOWN):
    """Khởi chạy hệ thống suy luận thời gian thực qua Video Stream (Webcam, DroidCam hoặc File)."""
    print("="*65)
    print(" [BƯỚC 3 & 4: SUY LUẬN REAL-TIME & CẢNH BÁO TELEGRAM BOT] ")
    print("="*65)

    if not MODEL_PATH.exists():
        print(f"[LỖI] Không tìm thấy file mô hình '{MODEL_PATH}'. Vui lòng chạy Bước 2 trước!")
        return

    # Khởi tạo Telegram Alert Notifier
    telegram_notifier = TelegramAlertNotifier(
        config_path=TELEGRAM_CONFIG_PATH,
        cooldown_seconds=cooldown,
        enabled=enable_telegram
    )

    print(f"[INFO] Đang tải mô hình Deep Learning từ '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    @tf.function(reduce_retracing=True)
    def predict_step(tensor):
        return model(tensor, training=False)

    dummy_input = tf.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=tf.float32)
    _ = predict_step(dummy_input)
    print("[INFO] Đã biên dịch Graph Mode & Khởi động mô hình thành công!")

    print("[INFO] Đang khởi tạo MediaPipe Pose...")
    pose_detector = RealtimePoseDetector()
    print("[INFO] MediaPipe Pose đã sẵn sàng!")

    cap = open_video_source(source)
    if not cap or not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn video: {source}")
        return

    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    status = "INITIALIZING"
    fall_prob = 0.0
    fps = 0.0
    prev_time = time.time()
    frame_count = 0

    is_fall_active = False
    fall_start_time = 0.0
    stand_up_frames = 0
    REQUIRED_STAND_FRAMES = 20

    print("\n" + "-"*55)
    print(" [HƯỚNG DẪN ĐIỀU KHIỂN]")
    print("  - Nhấn phím 'q' hoặc 'ESC' trên cửa sổ video để THOÁT.")
    print("  - Nhấn phím 'r' để RESET bộ đệm chuỗi và xóa cảnh báo.")
    print(f"  - Ngưỡng phát hiện ngã (Threshold): {threshold * 100:.0f}%")
    print(f"  - Cảnh báo Telegram: {'BẬT (Async)' if telegram_notifier.is_configured and enable_telegram else 'TẮT / Chưa cấu hình'}")
    print("-" * 55 + "\n")

    window_name = "AI Fall Detection System (Vision-based)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Luồng video kết thúc hoặc không còn frame.")
                break

            keypoints, landmarks_data, annotated_frame = pose_detector.process_frame(frame)
            frame_buffer.append(keypoints)

            current_time = time.time()

            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_tensor = tf.convert_to_tensor(np.expand_dims(frame_buffer, axis=0), dtype=tf.float32)
                pred_prob = predict_step(input_tensor).numpy()[0][0]
                fall_prob = float(pred_prob)

                if fall_prob >= threshold:
                    if not is_fall_active:
                        is_fall_active = True
                        fall_start_time = current_time
                        print(f"\n[🚨 CẢNH BÁO KHẨN CẤP] Phát hiện sự cố té ngã! Độ tin cậy: {fall_prob*100:.1f}%")
                        
                        telegram_notifier.send_fall_alert_async(
                            frame_bgr=annotated_frame, 
                            fall_prob=fall_prob, 
                            camera_source=str(source)
                        )
            else:
                status = f"BUFFERING ({len(frame_buffer)}/{SEQUENCE_LENGTH})"
                fall_prob = 0.0

            if is_fall_active:
                is_lying, is_standing, torso_angle = check_body_posture(landmarks_data)

                if is_standing:
                    stand_up_frames += 1
                    if stand_up_frames >= REQUIRED_STAND_FRAMES:
                        is_fall_active = False
                        stand_up_frames = 0
                        status = "NORMAL (Person Recovered)"
                        print("[INFO] Người đã tự đứng dậy an toàn. Tắt cảnh báo té ngã.")
                else:
                    stand_up_frames = 0
                    status = "FALL DETECTED"

                fall_elapsed = current_time - fall_start_time
            else:
                fall_elapsed = 0.0
                if len(frame_buffer) == SEQUENCE_LENGTH:
                    status = "NORMAL"

            frame_count += 1
            if current_time - prev_time >= 0.5:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            draw_hud(annotated_frame, status, fall_prob, fps, len(frame_buffer), threshold, fall_elapsed, is_fall_active, tg_active=telegram_notifier.is_configured and enable_telegram)
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Người dùng nhấn thoát hệ thống.")
                break
            elif key == ord('r'):
                frame_buffer.clear()
                is_fall_active = False
                stand_up_frames = 0
                print("[INFO] Đã làm mới bộ đệm chuỗi và xóa trạng thái cảnh báo.")

    except KeyboardInterrupt:
        print("\n[INFO] Nhận tín hiệu ngắt từ bàn phím.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã giải phóng Camera và đóng toàn bộ cửa sổ an toàn.")

def main():
    parser = argparse.ArgumentParser(description="Chương trình suy luận phát hiện té ngã thời gian thực & Cảnh báo Telegram")
    parser.add_argument("--source", default="0", 
                        help="Nguồn video: Đường dẫn file video (ví dụ: 'data/Home_01/Home_01/Videos/video (3).avi'), hoặc '0' cho Webcam / DroidCam")
    parser.add_argument("--threshold", type=float, default=FALL_THRESHOLD, 
                        help="Ngưỡng xác suất kích hoạt cảnh báo té ngã (mặc định: 0.8)")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN, 
                        help="Thời gian chờ chống spam tin nhắn Telegram tính bằng giây (mặc định: 15s)")
    parser.add_argument("--no-telegram", action="store_true", 
                        help="Tắt tính năng gửi cảnh báo Telegram")
    args = parser.parse_args()

    run_realtime_detection(
        source=args.source, 
        threshold=args.threshold, 
        enable_telegram=not args.no_telegram, 
        cooldown=args.cooldown
    )

if __name__ == "__main__":
    main()
