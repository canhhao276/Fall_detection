
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

X_DATA_PATH = PROCESSED_DIR / "X_data.npy"
Y_DATA_PATH = PROCESSED_DIR / "y_data.npy"
MODEL_SAVE_PATH = MODELS_DIR / "fall_model.keras"
HISTORY_PLOT_PATH = REPORTS_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# Siêu tham số (Hyperparameters)
SEQUENCE_LENGTH = 30
FEATURE_DIM = 132
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

def load_data():
    """Nạp dữ liệu mảng numpy đã trích xuất ở Bước 1."""
    if not X_DATA_PATH.exists() or not Y_DATA_PATH.exists():
        print(f"[LỖI] Không tìm thấy file dữ liệu tại '{PROCESSED_DIR}'. Vui lòng chạy Bước 1 trước!")
        sys.exit(1)

    print(f"[INFO] Đang nạp dữ liệu từ '{PROCESSED_DIR}'...")
    X = np.load(X_DATA_PATH)
    y = np.load(Y_DATA_PATH)
    print(f"[INFO] Nạp thành công: X = {X.shape}, y = {y.shape}")
    return X, y

def build_lstm_model(input_shape):
    """Xây dựng kiến trúc mô hình Deep Learning LSTM 2 tầng."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ], name="FallDetection_LSTM")

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def plot_history(history):
    """Vẽ và lưu đồ thị Loss, Accuracy, Recall, Precision qua các Epochs."""
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ĐỒ THỊ HUẤN LUYỆN MÔ HÌNH LSTM PHÁT HIỆN TÉ NGÃ', fontsize=15, fontweight='bold')

    # Loss
    axs[0, 0].plot(epochs_range, history.history['loss'], label='Train Loss', color='#1f77b4', lw=2)
    axs[0, 0].plot(epochs_range, history.history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2, linestyle='--')
    axs[0, 0].set_title('Loss (Binary Crossentropy)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle=':')
    axs[0, 0].legend()

    # Accuracy
    axs[0, 1].plot(epochs_range, history.history['accuracy'], label='Train Acc', color='#2ca02c', lw=2)
    axs[0, 1].plot(epochs_range, history.history['val_accuracy'], label='Val Acc', color='#d62728', lw=2, linestyle='--')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True, linestyle=':')
    axs[0, 1].legend()

    # Recall
    axs[1, 0].plot(epochs_range, history.history['recall'], label='Train Recall', color='#9467bd', lw=2)
    axs[1, 0].plot(epochs_range, history.history['val_recall'], label='Val Recall', color='#8c564b', lw=2, linestyle='--')
    axs[1, 0].set_title('Recall (Độ nhạy phát hiện ngã)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].grid(True, linestyle=':')
    axs[1, 0].legend()

    # Precision
    axs[1, 1].plot(epochs_range, history.history['precision'], label='Train Precision', color='#e377c2', lw=2)
    axs[1, 1].plot(epochs_range, history.history['val_precision'], label='Val Precision', color='#17becf', lw=2, linestyle='--')
    axs[1, 1].set_title('Precision (Độ chính xác cảnh báo)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].grid(True, linestyle=':')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[INFO] Đã lưu đồ thị huấn luyện tại: {HISTORY_PLOT_PATH}")

def plot_confusion_matrix(cm):
    """Vẽ và lưu ma trận nhầm lẫn (Confusion Matrix)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['No Fall (0)', 'Fall (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='MA TRẬN NHẦM LẪN (CONFUSION MATRIX)',
           ylabel='Nhãn thực tế (True Label)',
           xlabel='Nhãn dự đoán (Predicted Label)')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()
    print(f"[INFO] Đã lưu ma trận nhầm lẫn tại: {CONFUSION_MATRIX_PATH}")

def main():
    print("="*65)
    print(" [BƯỚC 2: HUẤN LUYỆN MÔ HÌNH DEEP LEARNING LSTM PHÁT HIỆN TÉ NGÃ] ")
    print("="*65)

    X, y = load_data()

    # 1. Phân chia dữ liệu Stratified Train/Val Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Tập huấn luyện (Train): {X_train.shape[0]} mẫu | Tập kiểm thử (Val): {X_val.shape[0]} mẫu")

    # 2. Xử lý mất cân bằng dữ liệu với Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    print(f"[INFO] Trọng số phân lớp tự động (Class Weights): {class_weight_dict}")

    # 3. Xây dựng mô hình
    model = build_lstm_model(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM))
    model.summary()

    # 4. Thiết lập Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Huấn luyện mô hình
    print("\n[INFO] Bắt đầu quá trình huấn luyện mô hình Deep Learning...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Đánh giá chi tiết trên tập kiểm thử độc lập
    print("\n" + "="*65)
    print(" [ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP VALIDATION] ")
    print("="*65)
    
    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Độ chính xác (Accuracy): {val_acc*100:.2f}%")
    print(f"✅ Diện tích dưới đường cong (ROC-AUC): {val_auc:.4f}")
    print(f"✅ Độ nhạy phát hiện ngã (Recall / Sensitivity): {val_rec*100:.2f}%")
    print(f"✅ Độ chính xác cảnh báo (Precision): {val_prec*100:.2f}%")

    # 7. Dự đoán và in Classification Report + Confusion Matrix
    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n" + "-"*55)
    print(" [BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT)] ")
    print("-"*55)
    print(classification_report(y_val, y_pred, target_names=['No Fall (0)', 'Fall (1)']))

    cm = confusion_matrix(y_val, y_pred)
    plot_history(history)
    plot_confusion_matrix(cm)

    print("\n" + "="*65)
    print(f"🎉 Huấn luyện thành công! Mô hình đã được lưu tại: {MODEL_SAVE_PATH}")
    print("="*65)

if __name__ == "__main__":
    main()
