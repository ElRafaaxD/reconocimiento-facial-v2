import os
import csv
import face_recognition

class FaceIndexer:
    def __init__(self, faces_folder: str, output_csv: str):
        self.faces_folder = faces_folder
        self.output_csv = output_csv
        self.rostros = []

    def procesar_faces(self):
        for persona in os.listdir(self.faces_folder):
            ruta_persona = os.path.join(self.faces_folder, persona)
            if not os.path.isdir(ruta_persona):
                continue

            for imagen in os.listdir(ruta_persona):
                ruta_imagen = os.path.join(ruta_persona, imagen)
                imagen_cargada = face_recognition.load_image_file(ruta_imagen)
                codificaciones = face_recognition.face_encodings(imagen_cargada)

                if not codificaciones:
                    print(f"[!] No se detectó rostro en {ruta_imagen}")
                    continue

                encoding = codificaciones[0]  # Tomamos el primero
                self.rostros.append({
                    "nombre": persona,
                    "vector": encoding.tolist()  # Convertimos a lista para CSV
                })

    def guardar_csv(self):
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["nombre", "vector"])  # Encabezados
            for rostro in self.rostros:
                writer.writerow([rostro["nombre"], rostro["vector"]])
        print(f"[✔] Rostros guardados en {self.output_csv}")