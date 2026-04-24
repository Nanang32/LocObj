import cv2
import torch
from ultralytics import RTDETR

# 1. Inisialisasi model dan pindahkan ke GPU
# RT-DETR sangat powerful saat dijalankan di GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RTDETR('rtdetr-l.pt')
model.to(device)  # Memaksa model menggunakan CUDA

# 2. Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Jalankan tracking dengan optimasi GPU
    # half=True: Menggunakan FP16 (Floating Point 16) untuk kecepatan 2x lipat tanpa mengurangi akurasi secara signifikan
    results = model.track(
        source=frame, 
        persist=True, 
        classes=[0], 
        tracker="bytetrack.yaml", 
        conf=0.3,
        device=device,  # Pastikan proses inferensi di GPU
        half=(device == 'cuda') # Hanya gunakan half jika di CUDA
    )

    for r in results:
        if r.boxes.id is not None:
            # Pindahkan koordinat ke CPU hanya saat akan menggambar dengan OpenCV
            boxes = r.boxes.xyxy.int().cpu().tolist()
            ids = r.boxes.id.int().cpu().tolist()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                
                # Visualisasi (Tetap di CPU karena OpenCV menggunakan CPU)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label_y = max(y1, 30) # Pencegahan teks terpotong di atas
                cv2.rectangle(frame, (x1, label_y - 30), (x1 + 60, label_y), (0, 255, 0), -1)
                cv2.putText(
                    frame, 
                    f"ID: {obj_id}", 
                    (x1 + 5, label_y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )

    cv2.imshow("RT-DETR - CUDA Accelerated", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()