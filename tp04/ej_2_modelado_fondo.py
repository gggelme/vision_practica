"""Ejercicio 2 — Modelado de fondo: de la diferencia de cuadros a MOG2, KNN y CNT.

Todos los métodos responden la misma pregunta para cada píxel: ¿es fondo o primer plano?
Cambia cómo recuerdan el fondo:
  - Diferencia de cuadros: el "fondo" es el cuadro anterior.
  - Promedio móvil: el fondo es un promedio que se actualiza de a poco.
  - MOG2: cada píxel es una mezcla de gaussianas (puede recordar varios colores de fondo).
  - KNN: cada píxel guarda muestras recientes; es fondo si se parece a suficientes de ellas.
  - CNT: cuenta cuántos cuadros seguidos un píxel se mantiene estable (muy liviano).
    Necesita opencv-contrib-python (cv2.bgsegm); si no está, se saltea.

Controles: ESPACIO pausa/reanuda, Q o ESC sale.
"""
import time

import cv2
import numpy as np

# Del ejercicio 1 se reutiliza la lectura del video y los títulos de los paneles.
# El import NO ejecuta el main del ejercicio 1: está protegido por if __name__ == '__main__'
from ej_1_preprocesamiento import VIDEO, con_titulo, leer_frames

VENTANA = 'Modelado de fondo'


def gris(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Métodos simples (implementados a mano)
# ---------------------------------------------------------------------------
# Los dos tienen un método apply(frame) -> máscara, igual que los sustractores de OpenCV.
# Así el resto del código los usa sin saber cuál es cuál
class DiferenciaCuadros:
    # B_t(p) = |I_t(p) - I_{t-1}(p)| > eps
    # Sólo detecta lo que cambió respecto del cuadro ANTERIOR
    def __init__(self, eps=20):
        self.eps, self.prev = eps, None

    def apply(self, f):
        # Gris + suavizado: se compara intensidad y se atenúa el ruido
        g = cv2.GaussianBlur(gris(f), (5, 5), 0)
        if self.prev is None:           # el primer cuadro no tiene con quién compararse
            self.prev = g
            return np.zeros_like(g)
        d = cv2.absdiff(g, self.prev)
        self.prev = g
        return ((d > self.eps) * 255).astype(np.uint8)


class PromedioMovil:
    # Fondo:   B_t = (1 - a) B_{t-1} + a I_t
    # Máscara: |I_t - B_t| > eps
    # alfa chico = el fondo cambia lento (tarda en "olvidar" y en aprender)
    def __init__(self, alfa=0.02, eps=25):
        self.a, self.eps, self.bg = alfa, eps, None

    def apply(self, f):
        # float32: el promedio necesita decimales (en uint8 se redondearía en cada paso)
        g = cv2.GaussianBlur(gris(f), (5, 5), 0).astype(np.float32)
        if self.bg is None:             # el fondo arranca siendo el primer cuadro
            self.bg = g.copy()
        d = np.abs(g - self.bg)
        self.bg = (1 - self.a) * self.bg + self.a * g
        return ((d > self.eps) * 255).astype(np.uint8)

    def fondo(self):
        # Imagen del fondo que el método tiene en memoria (para visualizarla)
        return self.bg.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sustractores de OpenCV
# ---------------------------------------------------------------------------
def hay_cnt():
    # CNT vive en cv2.bgsegm, que sólo trae el paquete opencv-contrib-python
    return hasattr(cv2, 'bgsegm')


def crear_sustractor(nombre):
    # MOG2 y KNN con detectShadows=True: la máscara tiene 3 valores
    #   0 = fondo, 127 = sombra (gris), 255 = primer plano
    if nombre == 'MOG2':
        # varThreshold = distancia (al cuadrado, en desvíos) para que un píxel NO sea fondo
        return cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    if nombre == 'KNN':
        # dist2Threshold = distancia al cuadrado a las muestras guardadas para ser "parecido"
        return cv2.createBackgroundSubtractorKNN(history=300, dist2Threshold=400, detectShadows=True)
    if nombre == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if nombre == 'Dif. cuadros':
        return DiferenciaCuadros()
    if nombre == 'Promedio movil':
        return PromedioMovil()
    raise ValueError(nombre)


def porcentaje_primer_plano(m):
    # Sólo 255: las sombras (127) no cuentan como primer plano
    return (m == 255).mean() * 100


# ---------------------------------------------------------------------------
# Post-proceso: de máscara a detecciones
# ---------------------------------------------------------------------------
K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def limpiar_mascara(m):
    # 1. Umbral > 200: se queda sólo con el primer plano (255) y descarta las sombras (127)
    _, b = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
    # 2. Apertura (erosión + dilatación): borra puntos sueltos más chicos que K_OPEN
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, K_OPEN)
    # 3. Cierre (dilatación + erosión): rellena huecos y une partes de una misma persona
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, K_CLOSE)


def detectar_blobs(mascara, area_min=120):
    # Cada región blanca conectada es un "blob"; se descartan las más chicas que area_min (ruido).
    # Devuelve una lista de cajas (x, y, w, h)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= area_min]


def dibujar_cajas(f, cajas, color=(0, 255, 0)):
    g = f.copy()
    for x, y, w, h in cajas:
        cv2.rectangle(g, (x, y), (x + w, y + h), color, 2)
    return g


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def grilla(paneles, cols=3):
    # Acomoda los paneles en filas de `cols`; si la última fila queda incompleta se
    # rellena con negro (np.hstack necesita que todas las filas tengan el mismo ancho)
    vacio = np.zeros_like(paneles[0])
    while len(paneles) % cols:
        paneles = paneles + [vacio]
    filas = [np.hstack(paneles[k:k + cols]) for k in range(0, len(paneles), cols)]
    return np.vstack(filas)


# ---------------------------------------------------------------------------
# Recorrido del video
# ---------------------------------------------------------------------------
def reproducir(frames, fps, metodos, ver_fondo=False):
    # metodos: {nombre: objeto con apply(frame)}. Cada cuadro pasa por todos los métodos.
    # Se mide cuánto tarda cada apply (sólo el método, no el dibujo) y el % de primer plano.
    # Devuelve {nombre: (ms por cuadro, % primer plano promedio)}
    tiempo = {nombre: 0.0 for nombre in metodos}
    pct = {nombre: [] for nombre in metodos}
    i, pausa = 0, False
    while i < len(frames):
        if not pausa:
            f = frames[i]
            mascaras = {}
            for nombre, m in metodos.items():
                t0 = time.perf_counter()
                mascaras[nombre] = m.apply(f)
                tiempo[nombre] += time.perf_counter() - t0
                pct[nombre].append(porcentaje_primer_plano(mascaras[nombre]))

            paneles = [con_titulo(f, f'cuadro {i}')]
            paneles += [con_titulo(m, nombre) for nombre, m in mascaras.items()]
            if ver_fondo and 'Promedio movil' in metodos:
                paneles.append(con_titulo(metodos['Promedio movil'].fondo(), 'fondo del promedio movil'))
            cv2.imshow(VENTANA, grilla(paneles))
            i += 1

        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa
    cv2.destroyAllWindows()
    # i = cuadros procesados (puede ser menos que el total si se salió con Q)
    return {nombre: (1000 * tiempo[nombre] / max(i, 1), np.mean(pct[nombre]) if pct[nombre] else 0.0)
            for nombre in metodos}


def reproducir_detecciones(frames, fps, metodos):
    # Igual que reproducir(), pero cada máscara pasa por el post-proceso y se dibujan las cajas.
    # Una columna por método: arriba la máscara limpia, abajo las detecciones sobre el cuadro.
    # Devuelve {nombre: cajas promedio por cuadro}
    n_cajas = {nombre: [] for nombre in metodos}
    i, pausa = 0, False
    while i < len(frames):
        if not pausa:
            f = frames[i]
            arriba, abajo = [], []
            for nombre, m in metodos.items():
                limpia = limpiar_mascara(m.apply(f))
                cajas = detectar_blobs(limpia)
                n_cajas[nombre].append(len(cajas))
                arriba.append(con_titulo(limpia, f'{nombre} - mascara limpia'))
                abajo.append(con_titulo(dibujar_cajas(f, cajas), f'{nombre} - {len(cajas)} detecciones'))
            cv2.imshow(VENTANA, np.vstack([np.hstack(arriba), np.hstack(abajo)]))
            i += 1

        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa
    cv2.destroyAllWindows()
    return {nombre: np.mean(v) if v else 0.0 for nombre, v in n_cajas.items()}


if __name__ == '__main__':
    frames, fps = leer_frames(VIDEO)
    print(f'{len(frames)} cuadros a {fps:.0f} fps')

    print('\nPASO 1 — Métodos simples: diferencia de cuadros y promedio móvil')
    print('  Ventana: cuadro | diferencia de cuadros | promedio móvil | fondo del promedio móvil')
    print('  Mirá a una persona caminando: en la diferencia sólo aparecen sus BORDES (lo que cambió')
    print('  de un cuadro al siguiente); en el promedio móvil aparece la silueta entera.')
    print('  ESPACIO pausa, Q o ESC sale.')
    metodos = {nombre: crear_sustractor(nombre) for nombre in ('Dif. cuadros', 'Promedio movil')}
    reproducir(frames, fps, metodos, ver_fondo=True)

    print('\nPASO 2 — Sustractores de OpenCV: MOG2, KNN y CNT')
    nombres = ['Dif. cuadros', 'Promedio movil', 'MOG2', 'KNN']
    if hay_cnt():
        nombres.append('CNT')
    else:
        print('  (CNT no disponible: falta opencv-contrib-python. Se compara sin CNT)')
    print('  Ventana: cuadro | Dif. cuadros | Promedio movil')
    print('           MOG2   | KNN          | CNT')
    print('  En MOG2 y KNN las sombras aparecen en GRIS (127), el primer plano en blanco (255).')
    # Métodos nuevos: arrancan de cero (los del paso 1 ya aprendieron el video)
    metodos = {nombre: crear_sustractor(nombre) for nombre in nombres}
    res = reproducir(frames, fps, metodos)

    print(f'\n  {"Método":<16} {"ms/cuadro":>10} {"% primer plano":>15}')
    for nombre, (ms, pct) in res.items():
        print(f'  {nombre:<16} {ms:>10.2f} {pct:>14.2f} %')

    print('\nPASO 3 — Post-proceso: umbral + apertura + cierre + blobs')
    print('  La máscara cruda tiene sombras, ruido y personas partidas en pedazos. Se limpia así:')
    print('    umbral > 200 (saca sombras) -> apertura 3x3 (saca puntos) -> cierre 9x9 (une pedazos)')
    print('    -> findContours: cada región con área >= 120 px es una detección (caja verde)')
    print('  Ventana: una columna por método | arriba = máscara limpia | abajo = detecciones')
    print('  Una persona = una caja es lo ideal. Mirá cuándo una persona sale partida en dos cajas')
    print('  o cuándo dos personas juntas quedan en una sola.')
    nombres = [n for n in ('MOG2', 'KNN', 'CNT') if n != 'CNT' or hay_cnt()]
    metodos = {nombre: crear_sustractor(nombre) for nombre in nombres}
    cajas = reproducir_detecciones(frames, fps, metodos)
    print('\n  Detecciones promedio por cuadro:')
    for nombre, n in cajas.items():
        print(f'    {nombre:<6} {n:5.1f}')
