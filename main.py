import cv2
import face_recognition
from core.face_comparator import FaceComparator

def main():
    comparador = FaceComparator(csv_path="data/rostros_indexados.csv")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] No se pudo abrir la camara.")
        return

    print("Camara activada. Presiona 'q' para salir...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] No se pudo leer el frame.")
            break

        # Convertir de BGR a RGB para face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Detectar rostros y codificarlos
        ubicaciones = face_recognition.face_locations(rgb_frame)
        codificaciones = face_recognition.face_encodings(rgb_frame, ubicaciones)

        for (top, right, bottom, left), codificacion in zip(ubicaciones, codificaciones):
            # Comparar con conocidos
            nombre, distancia = comparador.comparar_encoding(codificacion)
            if nombre:
                texto = f"{nombre} ({distancia:.2f})"
                color = (0, 255, 0)
            else:
                texto = "Desconocido"
                color = (0, 0, 255)

            # Dibujar resultados
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, texto, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar en ventana
        cv2.imshow("Reconocimiento Facial en Vivo", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
