import numpy as np
import os
from PIL import Image
from collections import Counter
mapeo_caracteres = {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6, '>': 7, '[': 8, ']': 9, 'o': 10, 'B': 11, 'b': 12}
windows_size = 14
map_number = len(mapeo_caracteres.keys())

def create_windowsCSV(text, window_size):
    rows, cols = text.shape
    windows = []

    for i in range(0, cols - window_size-1):
        window = text[0:(window_size),i:(i + window_size)]
        window = np.array(window)
        window = window.flatten()
        windows.append(window)

    return windows

def create_headerCSV():
    texto = "label"
    for i in range(windows_size**2):
        texto = texto + ",pixel" + str(i+1) 
    return texto

def create_attributes_normalized(window,imageTag):
    counts = Counter()

    # Count the frequency of each character in the window
    rows, cols = window.shape
    for i in range(rows):
        for j in range(cols):
            char = window[i, j]
            counts[char] += 1

    # Calculate and print the normalized frequency for each character in reverse order
    texto = ""
    for value in reversed(mapeo_caracteres.values()):
        number_of_value = counts[value]
        normalized_frequency = round(number_of_value / (rows * cols),4)
        texto = str(normalized_frequency) + " " + texto
    texto = str(imageTag) + texto
    return texto
    


def create_attributes_counted(window,imageTag):
    counts = Counter()

    # Count the frequency of each character in the window
    rows, cols = window.shape
    for i in range(rows):
        for j in range(cols):
            char = window[i, j]
            counts[char] += 1

    # Calculate and print the normalized frequency for each character in reverse order
    texto = ""
    for value in reversed(mapeo_caracteres.values()):
        number_of_value = counts[value]
        texto = str(number_of_value) + " " + texto
    texto = str(imageTag) + texto
    return texto
    


# files = ['mario-1-1.txt','mario-1-2.txt','mario-1-3.txt','mario-2-1.txt','mario-2-2.txt','mario-2-3.txt','mario-3-1.txt','mario-3-2.txt','mario-3-3.txt',]
files = []
path = "data/"
labels = []
# Lectura de los niveles
for carpeta in os.listdir(path):
    if os.path.isdir(path+carpeta):
        for textImage in os.listdir(path+carpeta+"/"):
            # Escribir la información en el CSV
            labels.append(str(carpeta))
            files.append(path+carpeta+"/"+textImage)

print (files)
print (labels)
print ("\n\n")
# Supongamos que tienes un texto de 14x202 (como una cadena de caracteres, por ejemplo).
# texto = "XS-?QE<>[]oBb"  # Reemplaza esto con tu propio texto
path = "generatedimg/"
file_counter = 0
globalcounter = 0
num_of_windows = 0
data_per_window = []
for file in files:
    texto = 0
    # Leer el texto desde un archivo
    with open(file, 'r') as file:
        texto = (file.read().split('\n'))
    
    ## Procesing from TVLGC to 0 .. 12 format
    texto_matriz = []
    for row in texto:
        newrow = []
        for char in row:
            newrow.append(mapeo_caracteres[char])
        texto_matriz.append(np.array(newrow))
    texto_matriz = np.array(texto_matriz)
    # Especifica el tamaño de la ventana deseada (14x14)

    # Crea las ventanas
    ventanas = create_windowsCSV(texto_matriz, windows_size)
    for ventana in ventanas:
        level = labels[file_counter] 
        data_num = ventana.shape
        for i in range(data_num[0]):
            level = level + "," +str(round (float(ventana[i])/float(map_number-1) * 255))
        data_per_window.append(level)
        # print(ventana)
    file_counter += 1
    

header = create_headerCSV()
with open("data_attributes.txt", 'w') as archivo:
    archivo.write(f"{header}\n")
    for i in data_per_window:
        archivo.write(f"{i}\n")


    # Ahora tendrás archivos de imagen (ventana_0.png, ventana_1.png, etc.) que representan las ventanas 14x14 de tu texto.
