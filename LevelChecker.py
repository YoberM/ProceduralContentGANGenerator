import heapq
import numpy as np
import time
import random

def a_star(mapa, minimunlocation=5):
    filas = len(mapa)
    columnas = len(mapa[0])

    def heuristica(x, y):
        # Usamos la distancia Manhattan al borde derecho
        return columnas - 1 - y

    def vecinos(x, y):
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in direcciones:
            nx, ny = x + dx, y + dy
            if 0 <= nx < filas and 0 <= ny < columnas and mapa[nx][ny] == '-':
                yield nx, ny

    # Encontrar todas las posiciones iniciales (primera columna)
    inicio = [(i, 0) for i in range(filas) if mapa[i][0] == '-' and i >= minimunlocation]
    
    # Cola de prioridad para A*
    cola = []
    for x, y in inicio:
        heapq.heappush(cola, (0 + heuristica(x, y), 0, x, y, [(x, y)]))

    visitados = set()

    while cola:
        _, costo, x, y, camino = heapq.heappop(cola)

        # Si llegamos a la última columna, hemos encontrado un camino
        if y == columnas - 1:
            # return camino
            return True

        if (x, y) in visitados:
            continue
        visitados.add((x, y))

        for nx, ny in vecinos(x, y):
            if (nx, ny) not in visitados:
                nuevo_costo = costo + 1
                heapq.heappush(cola, (nuevo_costo + heuristica(nx, ny), nuevo_costo, nx, ny, camino + [(nx, ny)]))

    return False  # No se encontró camino

def verificar_primera_columna(mapa):
    """Verifica que la primera columna no tenga una fila completa de '-'"""
    if all(fila[0] == '-' for fila in mapa):
        return False
    return True

def verificar_bloques_pregunta(mapa, symbols=['?', 'a'], minimunlocation=11):
    """Verifica que los bloques '?' siempre estén dos bloques por encima de otro bloque que no sea '-'"""
    filas = len(mapa)
    columnas = len(mapa[0])


    for x in range(filas):
        for y in range(columnas):
            if mapa[x][y] in symbols:
                if x >= minimunlocation:
                    return False
                if x + 2 >= filas or mapa[x + 1][y] != '-' or mapa[x + 2][y] != '-':
                    return False
    return True

def verificar_similitud_matriz(mapa, matriz_objetivo, tolerancia=1e-5):
    """Verifica si el mapa es similar a una matriz objetivo dentro de una tolerancia."""
    mapa_np = np.array(mapa)
    matriz_objetivo_np = np.array(matriz_objetivo)
    return np.allclose(mapa_np, matriz_objetivo_np, atol=tolerancia)

def leer_mapa(archivo):
    """Lee el mapa desde un archivo de texto y lo convierte en una lista de listas."""
    with open(archivo, 'r') as f:
        return [list(line.strip()) for line in f.readlines()]

def probar_mapa(mapa,pruebas = [
               verificar_primera_columna,
               verificar_bloques_pregunta,
               a_star
               ]
               ):
    esvalido = True
    salidas = []
    for prueba in pruebas:
        pruebavalida = prueba(mapa)
        esvalido = esvalido and pruebavalida
        salidas.append(pruebavalida)
        
    return esvalido, salidas


print(__name__)
if __name__ == "__main__":
    # Leer el mapa desde un archivo
    mapa = leer_mapa('GamePreviewMap0.txt')
    # Probar el mapa
    mapavalido, salidas = probar_mapa(mapa)

    if (mapavalido):
        print("si es")
    else:
        print("No es")
    print("salidas",salidas)

    
    # Generar una matriz de ejemplo de 14x14
    matrix_to_compare = np.random.rand(14, 14)

    # Generar 2600 matrices de ejemplo de 14x14
    matrices = np.random.rand(2600, 14, 14)

    # Función para comparar matrices
    def compare_matrices(matrix1, matrix2):
        return np.allclose(matrix1, matrix2)
    # Comparar la matriz con todas las matrices en el conjunto
    start_time = time.time()
    for i in range(1, 20):  # Probar con diferentes tamaños
        size = 2600 * i
        results = [compare_matrices(matrix_to_compare, matrices[j % 2600]) for j in range(size)]
        end_time = time.time()
        print(f"Tiempo para tamaño {size}: {end_time - start_time} segundos")
        start_time = time.time()  # Reiniciar el tiempo para la siguiente iteración
