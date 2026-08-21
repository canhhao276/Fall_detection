
import os
import sys
import glob
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 30   # Số frames trong một chuỗi hành động (~1.2s ở 25 FPS)
STEP = 10              # Bước trượt của cửa sổ (overlap = 20 frames)
FEATURE_DIM = 33 * 4   # 33 keypoints * 4 thuộc tính (x, y, z, visibility) = 132 chiều

OUTPUT_X_PATH = PROCESSED_DIR / "X_data.npy"
OUTPUT_Y_PATH = PROCESSED_DIR / "y_data.npy"

MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def download_model_if_needed():
    """Tự động tải model MediaPipe Pose Landmarker nếu chưa có."""
    if not MODEL_PATH.exists():
        print(f"[INFO] Đang tải mô hình MediaPipe Pose Landmarker ({MODEL_PATH})...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("[INFO] Tải mô hình thành công!")

class PoseExtractor:
    """Lớp bọc (Wrapper) xử lý MediaPipe Pose hỗ trợ cả Tasks API và Legacy."""
    def __init__(self):
        download_model_if_needed()
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
                base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
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
            base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
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

    def extract_landmarks(self, frame_bgr):
        """Trích xuất vector 132 chiều từ 1 frame ảnh."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not self.use_tasks_api:
            frame_rgb.flags.writeable = False
            results = self.detector.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                return np.array(landmarks, dtype=np.float32)
            else:
                return np.zeros(FEATURE_DIM, dtype=np.float32)
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_image)
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                first_person = results.pose_landmarks[0]
                landmarks = []
                for lm in first_person:
                    v = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
                    landmarks.extend([lm.x, lm.y, lm.z, v])
                return np.array(landmarks, dtype=np.float32)
            else:
                return np.zeros(FEATURE_DIM, dtype=np.float32)

def parse_annotation_file(txt_path):
    """Đọc file annotation .txt để xác định khoảng frame ngã [start_fall, end_fall]."""
    if not os.path.exists(txt_path):
        return None, None
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return None, None
        
        # Nếu dòng đầu là số nguyên -> start_frame
        try:
            start_frame = int(lines[0])
        except ValueError:
            return None, None
        
        # Nếu có dòng 2 -> end_frame
        if len(lines) >= 2:
            try:
                end_frame = int(lines[1])
            except ValueError:
                end_frame = start_frame + 40
        else:
            end_frame = start_frame + 40
            
        return start_frame, end_frame
    except Exception as e:
        print(f"[CẢNH BÁO] Không thể đọc file annotation {txt_path}: {e}")
        return None, None

def find_annotation_for_video(video_path):
    """Tìm file annotation .txt tương ứng với file video .avi."""
    v_path = Path(video_path)
    base_name = v_path.stem
    parent_dir = v_path.parent.parent
    
    # 1. Tìm trong thư mục Annotation_files cùng cấp
    anno_dir = parent_dir / "Annotation_files"
    if anno_dir.exists():
        candidates = [
            anno_dir / f"{base_name}.txt",
            anno_dir / f"{base_name.lower()}.txt",
            anno_dir / f"{base_name.replace(' ', '')}.txt",
            anno_dir / f"{base_name.replace('_', ' ')}.txt"
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        txt_files = list(anno_dir.glob("*.txt"))
        for tf in txt_files:
            if base_name.lower() in tf.stem.lower() or tf.stem.lower() in base_name.lower():
                return str(tf)

    return None

def process_single_video(video_path, pose_extractor):
    """Xử lý 1 video: Đọc toàn bộ frames, trích xuất Pose và cắt cửa sổ chuỗi 30 frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[CẢNH BÁO] Không thể mở video: {video_path}")
        return [], []

    frames_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lm = pose_extractor.extract_landmarks(frame)
        frames_landmarks.append(lm)
    cap.release()

    total_frames = len(frames_landmarks)
    if total_frames < SEQUENCE_LENGTH:
        return [], []

    # Tìm annotation để xác định khoảng frame ngã
    anno_path = find_annotation_for_video(video_path)
    start_fall, end_fall = parse_annotation_file(anno_path) if anno_path else (None, None)

    video_sequences = []
    video_labels = []

    for i in range(0, total_frames - SEQUENCE_LENGTH + 1, STEP):
        seq = frames_landmarks[i : i + SEQUENCE_LENGTH]
        window_start = i + 1
        window_end = i + SEQUENCE_LENGTH

        # Gán nhãn: Nếu cửa sổ giao với khoảng ngã >= 5 frames -> Nhãn 1 (Fall), ngược lại Nhãn 0 (No Fall)
        if start_fall is not None and end_fall is not None:
            overlap = max(0, min(window_end, end_fall) - max(window_start, start_fall) + 1)
            label = 1 if overlap >= 5 else 0
        else:
            label = 0

        video_sequences.append(seq)
        video_labels.append(label)

    return video_sequences, video_labels

def main():
    print("="*65)
    print(" [BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG MEDIAPIPE POSE TỪ LE2I DATASET] ")
    print("="*65)

    if not DATA_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục '{DATA_DIR}'!")
        return

    # Quét toàn bộ video .avi trong thư mục data/
    video_files = list(DATA_DIR.rglob("*.avi")) + list(DATA_DIR.rglob("*.mp4"))
    if not video_files:
        print(f"[LỖI] Không tìm thấy video .avi / .mp4 nào trong '{DATA_DIR}'!")
        return

    print(f"[INFO] Tìm thấy tổng cộng {len(video_files)} video trong bộ dữ liệu.")
    print(f"[INFO] Cấu hình trích xuất: Window = {SEQUENCE_LENGTH} frames, Step = {STEP} frames, Feature = {FEATURE_DIM} chiều.")

    pose_extractor = PoseExtractor()

    all_sequences = []
    all_labels = []

    print("\n[INFO] Bắt đầu trích xuất Pose và tạo dữ liệu huấn luyện...")
    for v_path in tqdm(video_files, desc="Trích xuất Pose"):
        seqs, labels = process_single_video(v_path, pose_extractor)
        all_sequences.extend(seqs)
        all_labels.extend(labels)

    if not all_sequences:
        print("[LỖI] Không trích xuất được chuỗi dữ liệu nào!")
        return

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)

    fall_count = int(np.sum(y == 1))
    no_fall_count = int(np.sum(y == 0))

    print("\n" + "="*65)
    print(" [KẾT QUẢ TRÍCH XUẤT ĐẶC TRƯNG THÀNH CÔNG] ")
    print("="*65)
    print(f"📁 File X_data: {OUTPUT_X_PATH} | Shape: {X.shape} | Dung lượng: {os.path.getsize(OUTPUT_X_PATH)/(1024**2):.2f} MB")
    print(f"📁 File y_data: {OUTPUT_Y_PATH} | Shape: {y.shape}")
    print(f"📊 Phân bố mẫu: Ngã (Fall = 1): {fall_count} mẫu | Bình thường (No-Fall = 0): {no_fall_count} mẫu")
    print("="*65)

if __name__ == "__main__":
    main()
