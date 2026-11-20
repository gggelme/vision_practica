"""Colab 1 — LeNet-5 sobre MNIST: arquitectura, entrenamiento y pizarra interactiva.

Un solo modelo recorre todo el práctico:
  1. Se carga MNIST y se arma la LeNet-5 (pesos al azar: todavía no sabe nada).
  2. Se entrena con Adam y se guarda en disco.
  3. Esa MISMA red entrenada clasifica los dígitos que se dibujan con el mouse.

Controles de la pizarra: dibujar con el botón izquierdo, C borra, + / - grosor, Q o ESC sale.
"""
import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # oculta los mensajes informativos de TensorFlow

import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from tensorflow.keras import layers, models

MODELO = Path(__file__).parent / 'lenet_mnist.keras'
VENTANA = 'Pizarra LeNet'
ESCALA_VENTANA = 2      # la pizarra se muestra al doble de tamaño (se dibuja más cómodo)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_mnist():
    # 60 000 dígitos de entrenamiento y 10 000 de prueba, de 28x28 en gris (0-255).
    # Keras lo descarga la primera vez y lo guarda en ~/.keras/datasets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 0-255 -> 0-1: la red aprende mejor con entradas chicas y en la misma escala
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # (N, 28, 28) -> (N, 28, 28, 1): Conv2D espera un eje de canales (1 = gris)
    return np.expand_dims(X_train, -1), y_train, np.expand_dims(X_test, -1), y_test


# ---------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------
def crear_lenet():
    # Sin relleno (padding 'valid'), una convolución KxK achica la imagen: N -> N - K + 1.
    # El pooling 2x2 la divide por 2. Los tamaños quedan: 28 -> 24 -> 12 -> 8 -> 4
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # Extracción de características: los filtros se aprenden solos
        layers.Conv2D(6, (5, 5), activation='relu'),       # 6 filtros 5x5   -> 24x24x6
        layers.AveragePooling2D((2, 2)),                    # promedio 2x2    -> 12x12x6
        layers.Conv2D(16, (5, 5), activation='relu'),      # 16 filtros 5x5  -> 8x8x16
        layers.AveragePooling2D((2, 2)),                    #                 -> 4x4x16
        # Clasificación: a partir de las características decide qué dígito es
        layers.Flatten(),                                   # 4*4*16 = 256 números
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax'),             # una probabilidad por dígito
    ])
    # Adam ajusta los pesos; sparse_categorical_crossentropy porque las etiquetas son
    # enteros 0-9 (no vectores one-hot)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def describir(model):
    # Una línea por capa: tamaño de salida y cantidad de parámetros (pesos que se aprenden)
    print(f'  {"capa":<22} {"salida":<16} {"parámetros":>10}')
    for capa in model.layers:
        salida = 'x'.join(str(d) for d in capa.output.shape[1:])
        print(f'  {capa.name:<22} {salida:<16} {capa.count_params():>10,}')
    print(f'  {"TOTAL":<39} {model.count_params():>10,}')


# ---------------------------------------------------------------------------
# Entrenamiento (una sola vez: después se carga del archivo)
# ---------------------------------------------------------------------------
def obtener_modelo(X_train, y_train, X_test, y_test, epocas=5):
    # Si ya hay un modelo entrenado guardado se usa ESE (siempre el mismo); si no, se
    # entrena y se guarda. Para reentrenar: borrar lenet_mnist.keras
    if MODELO.exists():
        print(f'  Cargando el modelo ya entrenado: {MODELO.name}')
        return tf.keras.models.load_model(MODELO)
    print(f'  No hay modelo guardado: entrenando {epocas} épocas (tarda alrededor de un minuto)...')
    model = crear_lenet()
    model.fit(X_train, y_train, epochs=epocas, batch_size=32,
              validation_data=(X_test, y_test), verbose=2)
    model.save(MODELO)
    print(f'  Guardado en {MODELO.name}')
    return model


# ---------------------------------------------------------------------------
# Preprocesamiento: dejar el dibujo "como un dígito de MNIST"
# ---------------------------------------------------------------------------
def preprocesar_como_mnist(img):
    # Recibe el dibujo en gris (trazo blanco sobre negro, uint8) y devuelve 28x28 en [0, 1],
    # o None si está vacío. Los dígitos de MNIST tienen el trazo en una caja de 20x20,
    # centrada por su centro de masa en una imagen de 28x28: hay que imitar eso, si no la
    # red ve algo distinto a lo que aprendió
    # 1. Recortar al rectángulo que contiene el trazo
    filas, cols = np.nonzero(img > 30)
    if len(filas) == 0:
        return None
    img = img[filas.min():filas.max() + 1, cols.min():cols.max() + 1]

    # 2. Escalar para que el lado mayor mida 20 px (manteniendo proporciones)
    alto, ancho = img.shape
    escala = 20.0 / max(alto, ancho)
    nuevo_alto, nuevo_ancho = max(1, round(alto * escala)), max(1, round(ancho * escala))
    img = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA).astype(np.float32)

    # 3. Pegarlo en el centro de un lienzo negro de 28x28
    lienzo = np.zeros((28, 28), np.float32)
    arriba, izq = (28 - nuevo_alto) // 2, (28 - nuevo_ancho) // 2
    lienzo[arriba:arriba + nuevo_alto, izq:izq + nuevo_ancho] = img

    # 4. Correrlo para que su centro de masa quede en el centro (14, 14)
    cy, cx = ndimage.center_of_mass(lienzo)
    lienzo = ndimage.shift(lienzo, (14 - cy, 14 - cx), order=1, mode='constant')
    return np.clip(lienzo, 0, 255) / 255.0


# ---------------------------------------------------------------------------
# Pizarra interactiva
# ---------------------------------------------------------------------------
class Pizarra:
    # Se dibuja con el mouse en un lienzo negro de 280x280; al soltar el botón, el dibujo se
    # preprocesa y la red lo clasifica. A la derecha: lo que ve la red (28x28), el dígito
    # predicho y la probabilidad de cada clase
    LADO = 280

    def __init__(self, model):
        self.model = model
        self.lienzo = np.zeros((self.LADO, self.LADO), np.uint8)
        self.grosor = 18
        self.ultimo = None          # último punto del trazo (None = no se está dibujando)
        self.img28 = None
        self.probs = None

    def mouse(self, evento, x, y, flags, _):
        # El mouse da coordenadas en la ventana agrandada: se pasan al lienzo de 280x280
        x, y = int(x / ESCALA_VENTANA), int(y / ESCALA_VENTANA)
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.ultimo = (x, y)
            cv2.circle(self.lienzo, (x, y), self.grosor // 2, 255, -1)
        elif evento == cv2.EVENT_MOUSEMOVE and self.ultimo is not None:
            cv2.line(self.lienzo, self.ultimo, (x, y), 255, self.grosor, cv2.LINE_AA)
            self.ultimo = (x, y)
        elif evento == cv2.EVENT_LBUTTONUP and self.ultimo is not None:
            self.ultimo = None
            self.clasificar()

    def clasificar(self):
        self.img28 = preprocesar_como_mnist(self.lienzo)
        if self.img28 is None:
            self.probs = None
            return
        entrada = self.img28.reshape(1, 28, 28, 1)      # misma forma que X_train: (lote, 28, 28, 1)
        self.probs = self.model(entrada, training=False).numpy()[0]

    def borrar(self):
        self.lienzo[:] = 0
        self.img28 = self.probs = None

    def dibujar(self):
        panel = np.full((self.LADO, 330, 3), 30, np.uint8)
        # Lo que ve la red, agrandado x4 (vecino más cercano: se ven los píxeles)
        cv2.putText(panel, 'lo que ve la red', (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if self.img28 is not None:
            mini = cv2.resize((self.img28 * 255).astype(np.uint8), (112, 112), interpolation=cv2.INTER_NEAREST)
            panel[26:138, 10:122] = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (9, 25), (122, 138), (120, 120, 120), 1)

        if self.probs is not None:
            pred = int(np.argmax(self.probs))
            cv2.putText(panel, str(pred), (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (10, 113, 232), 6, cv2.LINE_AA)
            cv2.putText(panel, f'{self.probs[pred] * 100:.1f} %', (25, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            # Barras: una por dígito, la predicha en naranja
            for i, p in enumerate(self.probs):
                y = 18 + i * 26
                color = (10, 113, 232) if i == pred else (232, 115, 26)
                cv2.putText(panel, str(i), (140, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
                cv2.rectangle(panel, (158, y), (158 + 110, y + 14), (70, 70, 70), -1)
                cv2.rectangle(panel, (158, y), (158 + int(p * 110), y + 14), color, -1)
                cv2.putText(panel, f'{p * 100:5.1f}%', (272, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, 'dibuja un digito', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # El texto guía y el borde se dibujan sobre una COPIA: el lienzo que ve la red queda limpio
        pizarra = cv2.cvtColor(self.lienzo, cv2.COLOR_GRAY2BGR)
        if not self.lienzo.any():
            cv2.putText(pizarra, 'dibuja aca', (75, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 2, cv2.LINE_AA)
        cv2.rectangle(pizarra, (0, 0), (self.LADO - 1, self.LADO - 1), (120, 120, 120), 1)

        vista = np.hstack([pizarra, panel])
        ayuda = np.full((26, vista.shape[1], 3), 50, np.uint8)
        cv2.putText(ayuda, f'C borra | + / - grosor ({self.grosor}) | Q o ESC sale', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        vista = np.vstack([vista, ayuda])
        return cv2.resize(vista, None, fx=ESCALA_VENTANA, fy=ESCALA_VENTANA, interpolation=cv2.INTER_LINEAR)

    def correr(self):
        cv2.namedWindow(VENTANA)
        cv2.setMouseCallback(VENTANA, self.mouse)
        while True:
            cv2.imshow(VENTANA, self.dibujar())
            tecla = cv2.waitKey(15) & 0xFF
            if tecla in (ord('q'), 27):
                break
            if tecla == ord('c'):
                self.borrar()
            elif tecla in (ord('+'), ord('=')):
                self.grosor = min(40, self.grosor + 2)
            elif tecla == ord('-'):
                self.grosor = max(6, self.grosor - 2)
            # Si se cerró la ventana con la X, salir
            if cv2.getWindowProperty(VENTANA, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('PASO 1 — MNIST y arquitectura LeNet-5')
    X_train, y_train, X_test, y_test = cargar_mnist()
    print(f'  Entrenamiento: {X_train.shape[0]} imágenes de {X_train.shape[1]}x{X_train.shape[2]}')
    print(f'  Prueba:        {X_test.shape[0]} imágenes')
    print(f'  Forma del tensor de entrada: {X_train.shape} -> (imágenes, alto, ancho, canales)')
    print(f'  Valores entre {X_train.min():.0f} y {X_train.max():.0f} (normalizados)')
    print(f'  Imágenes por dígito: {np.bincount(y_train).tolist()}')

    print('\n  Arquitectura:')
    describir(crear_lenet())

    print('\nPASO 2 — Entrenamiento')
    model = obtener_modelo(X_train, y_train, X_test, y_test)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'  Precisión en las 10 000 imágenes de prueba: {acc * 100:.2f} %')

    print('\nPASO 3 — Pizarra interactiva')
    print('  Dibujá un dígito con el mouse; al soltar el botón, la red lo clasifica.')
    print('  C borra, + / - cambian el grosor del trazo, Q o ESC sale.')
    print('  Probá: un dígito chico en una esquina (el preprocesamiento lo centra), o algo que')
    print('  NO sea un número (una carita): softmax igual reparte el 100 % entre los 10 dígitos.')
    Pizarra(model).correr()
