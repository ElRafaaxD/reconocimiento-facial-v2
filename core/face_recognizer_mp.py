import cv2
import face_recognition
import numpy as np
from multiprocessing import Process, Queue

class FaceRecognizerProcess(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue, face_comparator):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.face_comparator = face_comparator
        self.daemon = True  # <- esto permite que se cierre si el padre muere

    def run(self):
        while True:
            if not self.input_queue.empty():
                frame = self.input_queue.get()

                if isinstance(frame, str) and frame == "STOP":
                    print("🛑 [Proceso] Detenido por comando.")
                    break

                try:
                    # Reducir tamaño para acelerar
                    small_frame = frame[:, :, ::-1]  # BGR to RGB
                    small_frame = cv2.resize(small_frame, (0, 0), fx=0.25, fy=0.25)

                    # Detectar rostros y codificar
                    locations_small = face_recognition.face_locations(small_frame, model="hog")
                    encodings_small = face_recognition.face_encodings(small_frame, locations_small)

                    locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in locations_small]

                    resultados = []
                    for (top, right, bottom, left), encoding in zip(locations, encodings_small):
                        nombre, distancia = self.face_comparator.comparar_encoding(encoding)
                        resultados.append(((top, right, bottom, left), nombre, distancia))

                    self.output_queue.put(resultados)
                except Exception as e:
                    print(f"[❌] Error en proceso hijo: {e}")