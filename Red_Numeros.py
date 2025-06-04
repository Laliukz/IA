import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# tabla de números
numeros = {
    '111101101101111': 0,
    '001001001001001': 1,
    '111001111100111': 2,
    '111001111001111': 3,
    '101101111001001': 4,
    '111100111001111': 5,
    '111100111101111': 6,
    '111001001001001': 7,
    '111101111101111': 8,
    '111101111001111': 9
}

# activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
    return x * (1 - x)

# pesos
tamaño_entrada = 15 #entradas
tamaño_medio = 10 #capa media
tamaño_salida = 10 #salidas
np.random.seed(1)
pesos_entrada_medio = 2 * np.random.random((tamaño_entrada, tamaño_medio)) - 1
pesos_medio_salida = 2 * np.random.random((tamaño_medio, tamaño_salida)) - 1

# red neuronal
def entrenar(X, y, iteraciones):
    global pesos_entrada_medio, pesos_medio_salida
    for iteracion in range(iteraciones):
        # propagación hacia adelante
        entrada_capa_media = np.dot(X, pesos_entrada_medio)
        salida_capa_media = sigmoid(entrada_capa_media)
        
        entrada_capa_salida = np.dot(salida_capa_media, pesos_medio_salida)
        salida_capa_salida = sigmoid(entrada_capa_salida)
        
        # error
        error = y - salida_capa_salida
        
        # propagación hacia atrás
        delta_capa_salida = error * derivada_sigmoid(salida_capa_salida)
        error_capa_media = delta_capa_salida.dot(pesos_medio_salida.T)
        delta_capa_media = error_capa_media * derivada_sigmoid(salida_capa_media)
        
        # actualización de pesos
        pesos_medio_salida += salida_capa_media.T.dot(delta_capa_salida)
        pesos_entrada_medio += X.T.dot(delta_capa_media)

# datos de entrada y salida
X = np.array([list(map(int, list(key))) for key in numeros.keys()])
y = np.eye(10)[list(numeros.values())]

entrenar(X, y, 10000) # entrenamiento de la red

# predecir número
def predecir(imagen):
    entrada_capa_media = np.dot(imagen, pesos_entrada_medio)
    salida_capa_media = sigmoid(entrada_capa_media)
    
    entrada_capa_salida = np.dot(salida_capa_media, pesos_medio_salida)
    salida_capa_salida = sigmoid(entrada_capa_salida)
    
    return np.argmax(salida_capa_salida)

def imagen_a_matriz(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert('L')
    imagen = imagen.resize((3, 5)) # 3x5
    matriz = np.array(imagen)
    matriz = (matriz < 128).astype(int)
    return matriz.flatten()

def seleccionar_imagen():
    ruta_archivo = filedialog.askopenfilename()
    if ruta_archivo:
        imagen = imagen_a_matriz(ruta_archivo)
        prediccion = predecir(imagen)
        resultado_label.config(text=f'Resultado: {prediccion}')
        # mostrar imagen
        img = Image.open(ruta_archivo)
        img = img.resize((150, 250))
        img = ImageTk.PhotoImage(img)
        imagen_label.config(image=img)
        imagen_label.image = img

# interfaz
raiz = tk.Tk()
raiz.title("Reconocimiento de Números")
raiz.geometry("500x500")
boton_subir = tk.Button(raiz, text="Cargar Imagen", command=seleccionar_imagen)
boton_subir.pack()
resultado_label = tk.Label(raiz, text="Resultado: ")
resultado_label.pack()
imagen_label = tk.Label(raiz)
imagen_label.pack()
raiz.mainloop()

