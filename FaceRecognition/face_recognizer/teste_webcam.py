"""Testa se a webcam abre (Windows: DirectShow, Linux: V4L2, macOS: AVFoundation)."""

import sys

import cv2

from facerec.camera import abrir_camera

cap = abrir_camera(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
if not cap.isOpened():
    sys.exit("Não foi possível abrir a webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar da webcam")
        break
    print(f"\rframe {frame.shape[1]}x{frame.shape[0]}", end="")
    cv2.imshow("Webcam Teste", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
