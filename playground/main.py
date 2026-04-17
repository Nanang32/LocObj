import cv2
from ultralytics import RTDETR

# 1. Inisialisasi model
model = RTDETR('rtdetr-l.pt')

# 2. Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Jalankan tracking (classes=[0] untuk orang)
    results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", conf=0.3)

    for r in results:
        # Pastikan ada objek yang terdeteksi dan memiliki ID
        if r.boxes.id is not None:
            # Ambil koordinat kotak (xyxy), ID, dan kelas
            boxes = r.boxes.xyxy.int().cpu().tolist()
            ids = r.boxes.id.int().cpu().tolist()

            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                
                # --- GAMBAR MANUAL DENGAN OPENCV ---
                # 1. Gambar Bounding Box (Warna Hijau, Ketebalan 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2. Gambar Background Kecil untuk nomor ID agar mudah dibaca
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + 50, y1), (0, 255, 0), -1)

                # 3. Tulis hanya Nomor ID saja (Warna Putih)
                cv2.putText(
                    frame, 
                    f"ID: {obj_id}", 
                    (x1 + 5, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )

    # Tampilkan hasil yang sudah digambar manual
    cv2.imshow("RT-DETR - Only ID Number", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()