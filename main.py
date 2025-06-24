import cv2
from multiprocessing import Queue
from core.face_comparator import FaceComparator
from core.face_recognizer_mp import FaceRecognizerProcess
from core.face_indexer import FaceIndexer

def main():
    '''
    indexador = FaceIndexer(faces_folder="faces", output_csv="data/rostros_indexados.csv")
    indexador.procesar_faces()
    indexador.guardar_csv()
    '''

    comparador = FaceComparator(csv_path="data/rostros_indexados.csv")
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)

    recognizer_process = FaceRecognizerProcess(input_queue, output_queue, comparador)
    recognizer_process.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] No se pudo abrir la camara.")
        return

    print("Camara activada. Presiona 'q' para salir...")

    ultimos_resultados = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if input_queue.empty():
                input_queue.put(frame)

            if not output_queue.empty():
                ultimos_resultados = output_queue.get()

            for (top, right, bottom, left), nombre, distancia in ultimos_resultados:
                if nombre:
                    texto = f"{nombre} ({distancia:.2f})"
                    color = (0, 255, 0)
                else:
                    texto = "Desconocido"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, texto, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Reconocimiento Facial Multiproceso", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Cerrando proceso de reconocimiento...")
        if input_queue:
            input_queue.put("STOP")

        recognizer_process.join(timeout=3)

        cap.release()
        cv2.destroyAllWindows()
        print("Todo cerrado correctamente.")

if __name__ == "__main__":
    main()
