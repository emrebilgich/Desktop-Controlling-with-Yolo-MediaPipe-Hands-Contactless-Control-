from ultralytics import YOLO
import torch


def train_best_model():
    if torch.cuda.is_available():
        print(f"Başlıyoruz! Ekran Kartı: {torch.cuda.get_device_name(0)}")

   
    model = YOLO('yolov8n.pt')

    # Eğitimi başlat
    model.train(
        data=r'C:\Users\yunus\OneDrive\Masaüstü\dataset3\data.yaml',  
        epochs=150,  
        imgsz=640,  
        device=0,  
        batch=16,  
        patience=30, 
        optimizer='AdamW',  
        lr0=0.01,
        cos_lr=True,  
        label_smoothing=0.1, 
        name='yolo_final_proje'
    )


if __name__ == '__main__':
    train_best_model()
