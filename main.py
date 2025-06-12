from core.face_comparator import FaceComparator
from core.face_indexer import FaceIndexer

def main():
    '''
    indexador = FaceIndexer(faces_folder="faces", output_csv="data/rostros_indexados.csv")
    indexador.procesar_faces()
    indexador.guardar_csv()
    '''
    
    ruta_imagen_prueba = "faces_pruebas/enrique.jpeg" #imagen a comparar
    comparador = FaceComparator(csv_path="data/rostros_indexados.csv") #
    resultado = comparador.comparar_imagen(ruta_imagen_prueba)

    if resultado:
        nombre, distancia = resultado
        print(f"Coincidencia: {nombre} (distancia: {distancia:.4f})")
    else:
        print("[!] No se encontró coincidencia.")
    

if __name__ == "__main__":
    main()
