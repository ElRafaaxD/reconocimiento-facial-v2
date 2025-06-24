import os
import csv
import face_recognition

class FaceIndexer:
    def __init__(self, faces_folder: str, output_csv: str):
        self.faces_folder = faces_folder
        self.output_csv = output_csv
        self.rostros = []

    def procesar_faces(self):
        # recorrer los directorios
        for persona in os.listdir(self.faces_folder):
            ruta_persona = os.path.join(self.faces_folder, persona)
            if not os.path.isdir(ruta_persona):
                continue

            # recorrer imagenes de un directorio
            for imagen in os.listdir(ruta_persona):
                ruta_imagen = os.path.join(ruta_persona, imagen)
                imagen_cargada = face_recognition.load_image_file(ruta_imagen)
                codificaciones = face_recognition.face_encodings(imagen_cargada)

                if not codificaciones:
                    print(f"[!] No se detectó rostro en {ruta_imagen}")
                    continue

                encoding = codificaciones[0]
                self.rostros.append([persona] + encoding.tolist())  # nombre + vector

    def guardar_csv(self):
        #abrir el objecto csv
        with open(self.output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            for rostro in self.rostros:
                writer.writerow(rostro)
        print(f"Rostros guardados en {self.output_csv}")
