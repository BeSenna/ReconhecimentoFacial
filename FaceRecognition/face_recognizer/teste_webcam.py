import sys
import cv2

def abrir_camera(indice=0):
    """DirectShow (CAP_DSHOW) só existe no Windows; no Linux usa V4L2."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    return cv2.VideoCapture(indice, backend)

cap = abrir_camera(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar da webcam")
        break

    cv2.imshow("Webcam Teste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
