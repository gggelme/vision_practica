"""Ejercicio 1 — Preprocesamiento: señal vs. ruido y reducción de carga computacional.

Si el ruido no se filtra, el modelo de fondo lo detecta como movimiento falso.

Controles: ESPACIO pausa/reanuda, Q o ESC sale.
"""
import time
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
VIDEO = DATOS / 'vtest.avi'
ESCALA = 0.6            # los cuadros se reducen al 60 % (como en el notebook)


def reducir(frame, escala=ESCALA):
    return cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def leer_frames(ruta, max_frames=400, escala=ESCALA):
    # Lee hasta max_frames cuadros del video y los devuelve reducidos, junto con los fps
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(reducir(f, escala))
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# Ruido del sensor
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)  # semilla fija: el ruido es el mismo en cada corrida


def ruido_sal_pimienta(f, p=0.01):
    # Simula un sensor defectuoso: una proporción p de los píxeles queda "quemada".
    # La mitad se va a negro (pimienta) y la otra mitad a blanco (sal).
    # Se sortea un número por píxel en [0, 1): los menores a p/2 son pimienta, los mayores
    # a 1 - p/2 son sal. Cada llamada sortea de nuevo -> el ruido cambia en cada cuadro
    r = rng.random(f.shape[:2])
    g = f.copy()
    g[r < p / 2] = 0
    g[r > 1 - p / 2] = 255
    return g


# ---------------------------------------------------------------------------
# Filtros (prefiltro antes de MOG2)
# ---------------------------------------------------------------------------
def filtro_mediana(f):
    # Reemplaza cada píxel por la mediana de su vecindario 5x5. Un píxel de sal o pimienta
    # es un valor extremo aislado: queda en la punta del ordenamiento y la mediana lo ignora
    return cv2.medianBlur(f, 5)


def filtro_gaussiano(f):
    # Promedio ponderado del vecindario 5x5. El píxel quemado no desaparece: se reparte
    # entre sus vecinos como una manchita más suave
    return cv2.GaussianBlur(f, (5, 5), 0)


# ---------------------------------------------------------------------------
# Modelo de fondo (MOG2)
# ---------------------------------------------------------------------------
def crear_mog2():
    # history = cuántos cuadros recuerda el modelo de fondo
    # varThreshold = qué tan distinto tiene que ser un píxel para ser primer plano
    # Sin sombras (detectShadows=False): la máscara es sólo 0 (fondo) o 255 (primer plano)
    return cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)


def mascara_final(frames, con_ruido=False, prefiltro=None, n=120):
    # Le pasa a MOG2 los primeros n cuadros (con o sin ruido) y devuelve la máscara del último.
    # Los primeros cuadros sirven para que el modelo aprenda el fondo; en el cuadro n ya
    # está estable y la máscara muestra lo que considera primer plano.
    # Orden: primero se agrega el ruido (lo hace el sensor) y después se filtra (lo hacemos nosotros)
    mog = crear_mog2()
    for f in frames[:n]:
        if con_ruido:
            f = ruido_sal_pimienta(f)
        if prefiltro:
            f = prefiltro(f)
        m = mog.apply(f)
    return m


def porcentaje_primer_plano(m):
    return (m > 0).mean() * 100


# ---------------------------------------------------------------------------
# Reducción de carga computacional
# ---------------------------------------------------------------------------
def roi(f, y0=0.25, y1=0.85, x0=0.1, x1=0.9):
    # Recorta la región de interés (por defecto 60 % del alto y 80 % del ancho, centrada abajo,
    # donde camina la gente). Sólo se procesa lo que importa
    alto, ancho = f.shape[:2]
    return f[int(alto * y0):int(alto * y1), int(ancho * x0):int(ancho * x1)]


def tiempo_mog2(frames, n=100):
    # Tiempo promedio (ms) que tarda MOG2 en procesar UN cuadro. Se mide sólo el apply,
    # no la preparación de los cuadros (recorte, resize), que ya viene hecha
    mog = crear_mog2()
    n = min(n, len(frames))
    t0 = time.perf_counter()
    for f in frames[:n]:
        mog.apply(f)
    return 1000 * (time.perf_counter() - t0) / n


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def con_titulo(img, texto):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Recorrido del video
# ---------------------------------------------------------------------------
def reproducir(frames, fps, desde=20):
    # Cuatro MOG2 en paralelo, uno por variante. Los tres con ruido reciben EXACTAMENTE el
    # mismo cuadro ruidoso: así la única diferencia entre ellos es el filtro.
    # Devuelve el % de primer plano promedio de cada variante, sin contar los primeros
    # `desde` cuadros (ahí el modelo todavía está aprendiendo el fondo y marca todo)
    mogs = {nombre: crear_mog2() for nombre in ('sin ruido', 'con ruido', 'mediana', 'gaussiano')}
    porcentajes = {nombre: [] for nombre in mogs}
    i, pausa = 0, False

    while i < len(frames):
        if not pausa:
            f = frames[i]
            ruidoso = ruido_sal_pimienta(f)
            mediana = filtro_mediana(ruidoso)
            m = {
                'sin ruido': mogs['sin ruido'].apply(f),
                'con ruido': mogs['con ruido'].apply(ruidoso),
                'mediana': mogs['mediana'].apply(mediana),
                'gaussiano': mogs['gaussiano'].apply(filtro_gaussiano(ruidoso)),
            }
            if i >= desde:
                for nombre, mascara in m.items():
                    porcentajes[nombre].append(porcentaje_primer_plano(mascara))

            # Arriba: lo que ve cada MOG2 | Abajo: lo que detecta
            arriba = np.hstack([con_titulo(ruidoso, f'cuadro {i} con ruido'),
                                con_titulo(m['sin ruido'], 'MOG2 sin ruido (referencia)'),
                                con_titulo(m['con ruido'], 'MOG2 con ruido, sin filtrar')])
            abajo = np.hstack([con_titulo(mediana, 'cuadro con ruido + mediana'),
                               con_titulo(m['mediana'], 'MOG2 + mediana 5x5'),
                               con_titulo(m['gaussiano'], 'MOG2 + gaussiano 5x5')])
            cv2.imshow('Preprocesamiento', np.vstack([arriba, abajo]))
            i += 1

        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa

    cv2.destroyAllWindows()
    return {nombre: np.mean(v) if v else 0.0 for nombre, v in porcentajes.items()}


if __name__ == '__main__':
    print('PASO 1 — Lectura del video')
    frames, fps = leer_frames(VIDEO)
    alto, ancho = frames[0].shape[:2]
    print(f'  -> {len(frames)} cuadros de {ancho}x{alto} a {fps:.0f} fps')

    print('\nPASO 2 — Ruido sal y pimienta')
    P = 0.01
    original = frames[0]
    ruidoso = ruido_sal_pimienta(original, P)
    distintos = np.any(original != ruidoso, axis=2).mean() * 100
    print(f'  Se "queman" ~{P * 100:.0f} % de los píxeles -> cambiaron {distintos:.2f} %')
    print('  (puede dar un poco menos: si un píxel ya era negro o blanco puro, no cambia)')

    print('\nPASO 3 — MOG2 con y sin ruido')
    N = 120
    m_limpia = mascara_final(frames, con_ruido=False, n=N)
    m_ruido = mascara_final(frames, con_ruido=True, n=N)
    print(f'  Máscara del cuadro {N - 1} (blanco = "se movió"):')
    print(f'    sin ruido: {porcentaje_primer_plano(m_limpia):5.2f} % de primer plano')
    print(f'    con ruido: {porcentaje_primer_plano(m_ruido):5.2f} % de primer plano')
    print('  Lo que sobra en la máscara con ruido es movimiento falso: puntos que cambian de un')
    print('  cuadro a otro sin que nada se mueva.')

    print('\nPASO 4 — Filtrar antes de MOG2 (mediana vs. gaussiano)')
    print('  Ventana: arriba = cuadro con ruido | MOG2 sin ruido | MOG2 con ruido sin filtrar')
    print('           abajo  = cuadro filtrado con mediana | MOG2 + mediana | MOG2 + gaussiano')
    print('  ESPACIO pausa, Q o ESC sale.')
    res = reproducir(frames, fps)
    print('  % de primer plano promedio (cuanto más cerca de "sin ruido", mejor):')
    for nombre, pct in res.items():
        print(f'    {nombre:<10} {pct:5.2f} %')

    print('\nPASO 5 — Reducción de carga: ROI, submuestreo y diezmado')
    casos = {
        'Original': frames,
        'ROI (60 % x 80 % del cuadro)': [roi(f) for f in frames],
        'Submuestreo espacial x0.5': [reducir(f, 0.5) for f in frames],
        'Diezmado temporal /2': frames[::2],    # un cuadro de cada dos
    }
    tiempos = {nombre: tiempo_mog2(lista) for nombre, lista in casos.items()}
    ref = tiempos['Original']
    print(f'  {"Método":<30} {"ms/cuadro":>10} {"vs. original":>13}')
    for nombre, ms in tiempos.items():
        print(f'  {nombre:<30} {ms:>10.2f} {ref / ms:>12.1f}x')
    print('  El diezmado tarda lo mismo POR CUADRO (los cuadros son iguales), pero procesa la mitad:')
    print(f'  para cubrir los mismos {len(frames) / fps:.0f} s de video hace la mitad del trabajo.')
