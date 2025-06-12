import csv
import face_recognition
import numpy as np
from typing import List, Tuple, Optional

class FaceComparator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.nombres: List[str] = []
        self.encodings: List[np.ndarray] = []
        self._cargar_csv()

    def _cargar_csv(self):
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nombre = row['nombre']
                vector = np.array(eval(row['vector']))
                self.nombres.append(nombre)
                self.encodings.append(vector)
        print(f"[✔] Se cargaron {len(self.nombres)} rostros conocidos.")

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
