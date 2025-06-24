import csv
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

class FaceComparator:
    def __init__(self, csv_path: str):
        self.encodings = []
        self.names = []
        self.csv_path = csv_path
        self._cargar_csv()

    def _cargar_csv(self):
        with open(self.csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) != 129:
                    print(f"[!] Fila inválida: {row[:5]}... ({len(row)} columnas)")
                    continue

                name = row[0]
                try:
                    encoding = np.array(row[1:], dtype=np.float64)
                    self.names.append(name)
                    self.encodings.append(encoding)
                except Exception as e:
                    print(f"Error cargando encoding de {name}: {e}")

        print(f"[✔] Se cargaron {len(self.names)} rostros conocidos.")

    def comparar_imagen(self, ruta_imagen: str) -> Optional[Tuple[str, float]]:
        imagen = face_recognition.load_image_file(ruta_imagen)
        codificaciones = face_recognition.face_encodings(imagen)

        if not codificaciones:
            print("[!] No se detecto rostro en la imagen.")
            return None

        rostro_actual = codificaciones[0]
        distancias = face_recognition.face_distance(self.encodings, rostro_actual)
        indice_mas_cercano = np.argmin(distancias)
        distancia_minima = distancias[indice_mas_cercano]
        nombre_mas_cercano = self.nombres[indice_mas_cercano]

        print(f"Comparado con: {nombre_mas_cercano} (distancia: {distancia_minima:.4f})")

        if distancia_minima < 0.6:
            return nombre_mas_cercano, distancia_minima
        else:
            return None, None
    
    def comparar_encoding(self, encoding):
        if not self.encodings:
            return (None, None)

        distances = face_recognition.face_distance(self.encodings, encoding)
        min_distance = np.min(distances)
        best_index = np.argmin(distances)

        if min_distance < 0.6:
            return (self.names[best_index], min_distance)
        else:
            return (None, min_distance)




