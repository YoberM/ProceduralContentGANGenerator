import numpy as np
from PIL import Image
from collections import Counter

mapeo_caracteres = {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6, '>': 7, '[': 8, ']': 9, 'o': 10, 'B': 11, 'b': 12}
windows_size = 14
map_number = len(mapeo_caracteres.keys())

def create_windows(text, window_size):
    rows, cols = text.shape
    windows = []

    for i in range(0, cols - window_size-1):
        window = text[0:(window_size),i:(i + window_size)]
        windows.append(window)

    return windows

def create_windowsCSV(text, window_size):
    rows, cols = text.shape
    windows = []

    for i in range(0, cols - window_size-1):
        window = text[0:(window_size),i:(i + window_size)]
        window = np.array(window)
        window = window.flatten()
        windows.append(window)

    return windows

def create_header():
    texto = ""
    values = reversed(mapeo_caracteres.keys())
    for value in values:
        texto = value  + " " + texto 
    return texto

def create_headerCSV():
    texto = "label"
    for i in range(windows_size**2):
        texto = texto + ",pixel" + str(i+1) 
    return texto

def create_attributes_normalized(window, imageTag):
    counts = Counter()
    rows, cols = window.shape
    for i in range(rows):
        for j in range(cols):
            char = window[i, j]
            counts[char] += 1

    texto = ""
    for value in reversed(mapeo_caracteres.values()):
        number_of_value = counts[value]
        normalized_frequency = round(number_of_value / (rows * cols),4)
        texto = str(normalized_frequency) + " " + texto
    texto = str(imageTag) + texto
    return texto

def create_attributes_counted(window, imageTag):
    counts = Counter()
    rows, cols = window.shape
    for i in range(rows):
        for j in range(cols):
            char = window[i, j]
            counts[char] += 1

    texto = ""
    for value in reversed(mapeo_caracteres.values()):
        number_of_value = counts[value]
        texto = str(number_of_value) + " " + texto
    texto = str(imageTag) + texto
    return texto

path = "generatedimg/"
files = ['mario-1-1.txt', 'mario-1-2.txt', 'mario-1-3.txt']
file_counter = 0
globalcounter = 0
num_of_windows = 0
data_per_window = []

for file in files:
    with open(file, 'r') as file:
        texto = file.read().split('\n')

    texto_matriz = []
    for row in texto:
        newrow = []
        for char in row:
            newrow.append(mapeo_caracteres[char])
        texto_matriz.append(np.array(newrow))
    texto_matriz = np.array(texto_matriz)

    ventanas = create_windows(texto_matriz, windows_size)
    num_of_windows += len(ventanas)
    extension = ".jpg"
    for i, ventana in enumerate(ventanas):
        imagen_ventana = Image.fromarray((ventana/(map_number-1) * 255).astype(np.uint8))
        imagen_ventana = imagen_ventana.convert("RGB")
        imagen_ventana = imagen_ventana.resize([windows_size*2, windows_size*2], resample=Image.NEAREST)
        imagen_ventana = imagen_ventana.resize([windows_size*4, windows_size*4], resample=Image.NEAREST)
        imagen_ventana = imagen_ventana.resize([windows_size*8, windows_size*8], resample=Image.NEAREST)
        imagen_ventana.save(path+f"{globalcounter}"+extension)
        imagen_ventana.show()
        data_per_window.append(create_attributes_normalized(ventana, str(globalcounter)+extension+" "))
        globalcounter += 1
    file_counter += 1

header = create_header()
with open("data_attributes.txt", 'w') as archivo:
    archivo.write(f"{num_of_windows}\n")
    archivo.write(f"{header}\n")
    for i in data_per_window:
        archivo.write(f"{i}\n")
