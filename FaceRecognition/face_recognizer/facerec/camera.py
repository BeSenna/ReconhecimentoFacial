import sys

import cv2


def abrir_camera(indice=0):
    """DirectShow (CAP_DSHOW) só existe no Windows; no Linux usa V4L2,
    no macOS AVFoundation."""
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice)  # tenta o backend automático
    return cap
