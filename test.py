import cv2
from ultralytics import RTDETR, YOLO

# Pilih salah satu model (RT-DETR cenderung lebih stabil untuk single object)
model = RTDETR('rtdetr-l.pt')  # Gunakan model = YOLO('yolov9c.pt') untuk YOLOv9

# Buka kamera
cap = cv2.VideoCapture(0)

print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Jalankan deteksi
    # stream=True agar hemat memori pada video stream
    results = model(frame, stream=True)

    for r in results:
        # Cek jika ada objek yang terdeteksi
        if len(r.boxes) > 0:
            # Cari indeks box dengan confidence tertinggi
            top_idx = r.boxes.conf.argmax()
            
            # Ambil data box tunggal tersebut
            top_box = r.boxes[top_idx]
            
            # Reset isi r.boxes hanya dengan top_box agar r.plot() hanya menggambar satu box
            r.boxes = top_box 
            annotated_frame = r.plot()
        else:
            # Jika tidak ada objek, tampilkan frame asli tanpa box
            annotated_frame = frame

    # Tampilkan jendela
    cv2.imshow("Detection: Single Object Mode", annotated_frame)

    # Berhenti jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
