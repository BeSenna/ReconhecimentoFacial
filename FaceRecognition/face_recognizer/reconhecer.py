import sys
import cv2
import face_recognition
import pickle
from pathlib import Path

def abrir_camera(indice=0):
    """DirectShow (CAP_DSHOW) só existe no Windows; no Linux usa V4L2."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    return cv2.VideoCapture(indice, backend)

ENCODINGS_PATH = Path("output/encodings.pkl")


with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]


video_capture = abrir_camera(0)
if not video_capture.isOpened():
    raise SystemExit("Não foi possível abrir a webcam (verifique /dev/video0 e o grupo 'video').")

print("Reconhecimento facial iniciado. Pressione 'q' para sair.")

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Erro ao capturar frame da câmera.")
        continue

    try:
        
        if frame.dtype != 'uint8':
            frame = frame.astype('uint8')

        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        
        if rgb_small_frame is None or rgb_small_frame.dtype != 'uint8' or len(rgb_small_frame.shape) != 3 or rgb_small_frame.shape[2] != 3:
            print("Imagem convertida inválida.")
            continue

        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Desconhecido"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

        
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)

    except Exception as e:
        print("Erro durante o processamento do frame:", str(e))
        continue

    
    cv2.imshow("Reconhecimento Facial", frame)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
