"""Ejercicio 2 — Registración con características: SIFT + homografía + RANSAC.

Se alinean dos fotos de una MISMA escena plana (una pared con grafitis) tomadas desde ángulos
muy distintos. Como la escena es un plano, una homografía (matriz 3x3) lleva cada punto de una
foto a su lugar en la otra. La homografía se calcula a partir de puntos característicos
emparejados entre las dos imágenes.

Datos: graf1.png / graf3.png de los ejemplos de OpenCV (se descargan solos la primera vez).
Para usar fotos propias: dejarlas en datos/ y cambiar REFERENCIA y A_ALINEAR.

Controles: cualquier tecla cierra la ventana.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
URL_OPENCV = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/{}'
REFERENCIA = 'graf1.png'    # imagen fija: a ella se alinea la otra
A_ALINEAR = 'graf3.png'     # imagen que se deforma para que coincida con la referencia
ESCALA_VISTA = 0.75         # sólo para mostrar: dos imágenes de 800 px no entran en pantalla


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def descargar(nombre):
    # Devuelve la ruta en datos/; si no está y es un ejemplo de OpenCV, lo descarga
    ruta = DATOS / nombre
    if not ruta.exists():
        DATOS.mkdir(exist_ok=True)
        print(f'  Descargando {nombre}...')
        urllib.request.urlretrieve(URL_OPENCV.format(nombre), ruta)
    return ruta


def cargar(nombre):
    ruta = descargar(nombre)
    img = cv2.imread(str(ruta))
    if img is None:
        raise FileNotFoundError(f'No se pudo leer {ruta}')
    return img


# ---------------------------------------------------------------------------
# Puntos característicos y emparejamiento
# ---------------------------------------------------------------------------
def detectar(img, detector):
    # keypoints: dónde está cada punto distintivo (esquinas, manchas) y su escala/orientación.
    # descriptores: un vector por punto que describe cómo se ve su vecindario (en SIFT,
    # 128 números con histogramas de gradientes). Puntos parecidos -> descriptores cercanos
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detector.detectAndCompute(gris, None)


def emparejar(des_mov, des_ref, ratio=0.75):
    # Para cada descriptor de `mov` se buscan los 2 más cercanos en `ref` (distancia euclídea).
    # Ratio test de Lowe: el mejor tiene que ser claramente mejor que el segundo
    # (d1 < ratio * d2). Si los dos están casi igual de cerca, el punto es ambiguo
    # (ej. un patrón que se repite en la pared) y se descarta
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pares = bf.knnMatch(des_mov, des_ref, k=2)
    buenos = [m for m, n in pares if m.distance < ratio * n.distance]
    return buenos, len(pares)


# ---------------------------------------------------------------------------
# Homografía robusta (RANSAC)
# ---------------------------------------------------------------------------
def estimar_homografia(kp_mov, kp_ref, pares, umbral=5.0):
    # Coordenadas (x, y) de cada par: queryIdx indexa `mov`, trainIdx indexa `ref`
    pts_mov = np.float32([kp_mov[m.queryIdx].pt for m in pares]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in pares]).reshape(-1, 1, 2)
    # Una homografía queda definida con 4 pares. RANSAC repite muchas veces:
    #   1. elige 4 pares al azar y calcula la H que los cumple
    #   2. cuenta cuántos pares cumplen esa H: llevando el punto de `mov` con H, cae a menos
    #      de `umbral` píxeles de su pareja en `ref` (inliers)
    # Se queda con la H con más inliers y la recalcula usando todos ellos.
    # Los pares equivocados (outliers) no están de acuerdo con ninguna H común -> quedan afuera
    H, mascara = cv2.findHomography(pts_mov, pts_ref, cv2.RANSAC, ransacReprojThreshold=umbral)
    return H, mascara.ravel().astype(bool)


# ---------------------------------------------------------------------------
# Alineación y evaluación
# ---------------------------------------------------------------------------
def alinear(mov, H, tam_ref):
    # Deforma `mov` con H y la dibuja sobre una imagen del tamaño de la referencia.
    # Lo que queda fuera de `mov` (no hay de dónde sacar color) queda negro
    alto, ancho = tam_ref
    return cv2.warpPerspective(mov, H, (ancho, alto))


def superponer(a, b):
    # Mezcla 50/50: donde las dos coinciden se ve nítido, donde no, se ve "doble".
    # Donde `b` es negro (la foto deformada no cubre esa zona) se deja `a` sola: si no,
    # quedaría mezclada con negro y se vería oscura
    mezcla = cv2.addWeighted(a, 0.5, b, 0.5, 0)
    sin_b = ~b.any(axis=2)
    mezcla[sin_b] = a[sin_b]
    return mezcla


def tablero(a, b, lado=80):
    # Tablero de ajedrez: cuadrados alternados de `a` y `b`. Si están bien alineadas, las
    # líneas del dibujo siguen derecho al pasar de un cuadrado al vecino; si no, se cortan.
    # Donde `b` no tiene contenido se muestra `a`
    alto, ancho = a.shape[:2]
    filas, cols = np.indices((alto, ancho))
    de_b = ((filas // lado + cols // lado) % 2 == 1) & b.any(axis=2)
    salida = a.copy()
    salida[de_b] = b[de_b]
    return salida


def homografia_verdadera(nombre='H1to3p.xml'):
    # OpenCV publica la homografía exacta entre graf1 y graf3. Va de graf1 a graf3, o sea
    # de la referencia a la que se alinea: la nuestra es la inversa
    fs = cv2.FileStorage(str(descargar(nombre)), cv2.FILE_STORAGE_READ)
    H13 = fs.getNode('H13').mat()
    fs.release()
    return np.linalg.inv(H13)


def error_homografia(H, H_real, tam_mov, tam_ref, paso=20):
    # Se toma una grilla de puntos sobre `mov` (uno cada `paso` px), se lleva cada punto con
    # las dos homografías y se mide cuánto distan los resultados. Sólo cuentan los puntos que
    # caen dentro de la referencia (los de afuera no se ven en la alineación)
    alto, ancho = tam_mov
    xs, ys = np.meshgrid(np.arange(0, ancho, paso), np.arange(0, alto, paso))
    pts = np.float32(np.stack([xs.ravel(), ys.ravel()], axis=1)).reshape(-1, 1, 2)
    p_nuestra = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    p_real = cv2.perspectiveTransform(pts, H_real).reshape(-1, 2)
    alto_r, ancho_r = tam_ref
    dentro = ((p_real[:, 0] >= 0) & (p_real[:, 0] < ancho_r)
              & (p_real[:, 1] >= 0) & (p_real[:, 1] < alto_r))
    return np.linalg.norm(p_nuestra - p_real, axis=1)[dentro]


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def con_titulo(img, texto, escala=ESCALA_VISTA):
    img = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == '__main__':
    print('PASO 1 — Las dos fotos de la escena plana')
    ref = cargar(REFERENCIA)
    mov = cargar(A_ALINEAR)
    print(f'  Referencia ({REFERENCIA}): {ref.shape[1]}x{ref.shape[0]} px')
    print(f'  A alinear  ({A_ALINEAR}): {mov.shape[1]}x{mov.shape[0]} px')
    print('  Es la misma pared vista desde otro ángulo: los rectángulos de una se ven como')
    print('  trapecios en la otra. Eso es una transformación de perspectiva (homografía).')

    print('\nPASO 2 — Puntos SIFT y emparejamiento con ratio test')
    RATIO = 0.75
    sift = cv2.SIFT_create()
    t0 = time.perf_counter()
    kp_ref, des_ref = detectar(ref, sift)
    kp_mov, des_mov = detectar(mov, sift)
    t_det = time.perf_counter() - t0
    print(f'  Puntos detectados -> referencia: {len(kp_ref)} | a alinear: {len(kp_mov)} '
          f'({t_det * 1000:.0f} ms)')
    buenos, total = emparejar(des_mov, des_ref, RATIO)
    print(f'  Cada punto de "a alinear" se empareja con el más parecido de la referencia: {total} pares')
    print(f'  Tras el ratio test ({RATIO}): {len(buenos)} pares ({len(buenos) / total * 100:.1f} %)')
    print('  Aun así, algunos de esos pares están MAL (unen puntos que no son el mismo): RANSAC')
    print('  los va a descartar en el paso 3. La ventana muestra los 60 más parecidos, que suelen')
    print('  estar bien: las líneas siguen todas la misma "deformación".')

    # Las 60 mejores (menor distancia entre descriptores); con todas no se ve nada
    mejores = sorted(buenos, key=lambda m: m.distance)[:60]
    vis = cv2.drawMatches(mov, kp_mov, ref, kp_ref, mejores, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Emparejamientos', con_titulo(vis, f'a alinear  <->  referencia  '
                                                  f'(60 mejores de {len(buenos)} pares)'))
    print('  (cualquier tecla cierra la ventana y sigue con el paso 3)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 3 — Homografía con RANSAC')
    UMBRAL = 5.0
    H, inlier = estimar_homografia(kp_mov, kp_ref, buenos, UMBRAL)
    print(f'  Umbral de reproyección: {UMBRAL} px')
    print(f'  Inliers (cumplen la H): {inlier.sum()} / {len(buenos)} ({inlier.mean() * 100:.1f} %)')
    print(f'  Outliers (pares equivocados descartados): {(~inlier).sum()}')
    print('  H (lleva un punto de "a alinear" a su lugar en la referencia):')
    for fila in H:
        print('    [' + '  '.join(f'{v:11.5f}' for v in fila) + ']')

    # Verde = inliers, rojo = outliers. Se dibujan hasta 50 de cada uno
    ins = [m for m, ok in zip(buenos, inlier) if ok][:50]
    outs = [m for m, ok in zip(buenos, inlier) if not ok][:50]
    vis = cv2.drawMatches(mov, kp_mov, ref, kp_ref, outs, None, matchColor=(0, 0, 255),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis = cv2.drawMatches(mov, kp_mov, ref, kp_ref, ins, vis, matchColor=(0, 255, 0),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                          | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    cv2.imshow('RANSAC', con_titulo(vis, 'verde = inliers (cumplen H)   rojo = outliers (descartados)'))
    print('  (cualquier tecla cierra la ventana y sigue con el paso 4)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 4 — Alinear con H y evaluar contra la homografía verdadera')
    alineada = alinear(mov, H, ref.shape[:2])
    # "Antes": las dos fotos superpuestas tal cual, sin ninguna transformación
    antes = superponer(ref, cv2.resize(mov, (ref.shape[1], ref.shape[0])))
    despues = superponer(ref, alineada)
    print('  Ventana: arriba  = referencia | a alinear (original) | a alinear deformada con H')
    print('           abajo  = superpuestas ANTES | superpuestas DESPUES | tablero DESPUES')
    print('  Superpuestas 50/50: antes se ve todo doble; después parece UNA sola foto, porque')
    print('  coinciden punto por punto. En el tablero se alternan cuadrados de cada foto: si')
    print('  están alineadas, las líneas del grafiti siguen derecho de un cuadrado al otro.')

    if REFERENCIA == 'graf1.png' and A_ALINEAR == 'graf3.png':
        err = error_homografia(H, homografia_verdadera(), mov.shape[:2], ref.shape[:2])
        print(f'  Error contra la H verdadera ({len(err)} puntos de una grilla, en px):')
        print(f'    medio {err.mean():.2f} | mediano {np.median(err):.2f} | máximo {err.max():.2f}')
        print('  -> un punto típico de la foto alineada queda a esa distancia de donde debería estar')
    else:
        print('  (con fotos propias no hay homografía verdadera: se evalúa mirando la superposición)')

    # 6 paneles a la mitad de tamaño (400x320 cada uno)
    cv2.imshow('Antes | Despues', np.vstack([
        np.hstack([con_titulo(ref, 'referencia', 0.5),
                   con_titulo(mov, 'a alinear (original)', 0.5),
                   con_titulo(alineada, 'a alinear deformada con H', 0.5)]),
        np.hstack([con_titulo(antes, 'ANTES: superpuestas 50/50', 0.5),
                   con_titulo(despues, 'DESPUES: superpuestas 50/50', 0.5),
                   con_titulo(tablero(ref, alineada), 'DESPUES: tablero', 0.5)]),
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
