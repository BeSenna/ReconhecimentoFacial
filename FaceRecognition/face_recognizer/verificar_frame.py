import sys
import cv2

def abrir_camera(indice=0):
    """DirectShow (CAP_DSHOW) só existe no Windows; no Linux usa V4L2."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    return cv2.VideoCapture(indice, backend)

cap = abrir_camera(0)

ret, frame = cap.read()
if not ret or frame is None:
    print("Erro ao capturar frame.")
else:
    print("Shape:", frame.shape)
    print("Tipo:", frame.dtype)
    print("Canais:", frame.shape[2] if len(frame.shape) > 2 else "Monocromático")

cap.release()
