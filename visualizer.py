import numpy as np


# Characters
# Mapeo de caracteres
mapeo_caracteres = {'X': "GamePreviewTiles/TileX.png", 
                    'S': "GamePreviewTiles/TileS.png", 
                    '-': "GamePreviewTiles/Tile-.png", 
                    '?': "GamePreviewTiles/TileQ.png", 
                    'Q': "GamePreviewTiles/TileQUsed.png", 
                    'E': "GamePreviewTiles/TileE.png", 
                    '<': "GamePreviewTiles/TileLess.png", 
                    '>': "GamePreviewTiles/TileMore.png", 
                    '[': "GamePreviewTiles/Tile[.png", 
                    ']': "GamePreviewTiles/Tile].png", 
                    'o': "GamePreviewTiles/Tileo.png", 
                    'B': "GamePreviewTiles/TileBM.png", 
                    'b': "GamePreviewTiles/Tileb.png"}
mapeo_caracteres_inverso = {valor: clave for clave, valor in mapeo_caracteres.items()}
map_number = len(mapeo_caracteres.keys())
dataRGBrange = [ 255/(map_number-1) * i for i in range(map_number)]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def generar_preview(matriz):
    # Obtener dimensiones de la matriz
    filas, columnas = len(matriz), len(matriz[0])
    
    # Crear una imagen vacía con las dimensiones adecuadas
    imagen = np.zeros((filas * 16, columnas * 16, 3), dtype=np.uint8)
    
    # Rellenar la imagen con los tiles correspondientes
    for i in range(filas):
        for j in range(columnas):
            caracter = matriz[i][j]
            if caracter in mapeo_caracteres:
                if caracter == 'Q':
                    caracter ='-'
                tile_path = mapeo_caracteres[caracter]
                tile_img = mpimg.imread(tile_path)
                imagen[i*16:(i+1)*16, j*16:(j+1)*16, :] = tile_img[:, :, :3] * 255
    
    return imagen
