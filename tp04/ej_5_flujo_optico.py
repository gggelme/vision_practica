"""Ejercicio 5 — Flujo óptico: denso (Farnebäck, DIS) y disperso (GFTT + Lucas-Kanade).

El flujo óptico responde "¿cómo se mueve cada punto de la imagen entre dos cuadros?".
Se comparan tres métodos sobre el mismo par de cuadros y después Lucas-Kanade a lo largo del video:
  - Farnebäck: denso, un vector por píxel (expansión polinomial). Preciso, costoso.
  - DIS: denso, búsqueda inversa de parches + densificación. Mucho más rápido.
  - Lucas-Kanade + GFTT: disperso, sólo sigue esquinas. El más liviano.
Los densos se visualizan en HSV (matiz = dirección, brillo = magnitud) y con flechas;
Lucas-Kanade con flechas en cada esquina y, en el video, con estelas.

Controles: en las imágenes cualquier tecla sigue; en el video ESPACIO pausa, Q o ESC sale.
"""
import time
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
SALIDAS = Path(__file__).parent / 'salidas'
VIDEO = DATOS / 'vtest.avi'
VENTANA = 'Flujo optico'


def leer_cuadro(ruta, indice, escala=0.6):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, indice) #ponemos en el indice al cuadro
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'El video no tiene el cuadro {indice}')
    return cv2.resize(f, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def gris(f):
    # Los métodos de flujo óptico trabajan sobre la intensidad, no sobre el color
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def flujo_farneback(a, b):
    # Devuelve una matriz (alto, ancho, 2): para cada píxel, (fx, fy) = cuánto se movió.
    # Farnebäck aproxima la vecindad de cada píxel con un polinomio y estima su desplazamiento.
    # Parámetros: 0.5 = reducción entre niveles de pirámide, 3 = niveles, 15 = ventana,
    #             3 = iteraciones, 5 y 1.2 = vecindad y suavizado del polinomio, 0 = flags
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# DIS se crea una sola vez: crearlo en cada llamada sumaría ese costo al tiempo medido
_DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flujo_dis(a, b):
    # Dense Inverse Search: devuelve lo mismo que Farnebäck, una matriz (alto, ancho, 2).
    # En vez de analizar cada píxel, busca parches en una grilla gruesa (búsqueda inversa,
    # con pirámide de imágenes) y después densifica, interpolando entre los parches.
    # El preset MEDIUM equilibra velocidad y calidad (también hay ULTRAFAST y FAST)
    return _DIS.calc(a, b, None)


def detectar_esquinas(g, max_pts=150):
    # GFTT elige hasta `max_pts` esquinas (puntos con textura en todas las direcciones,
    # los únicos que se pueden reconocer sin ambigüedad), separadas al menos 8 px
    return cv2.goodFeaturesToTrack(g, max_pts, 0.01, 8, blockSize=7) #busca esquinas (cambios en todos los sentidos)


def seguir_puntos(a, b, p0, win=21, nivel=3):
    # Lucas-Kanade busca dónde quedó cada punto de p0 en el cuadro b, comparando una ventana
    # de `win` x `win` alrededor del punto. Con `nivel` niveles de pirámide empieza en la
    # imagen reducida (donde el desplazamiento es chico) y refina hasta la original
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        a, b, p0, None, winSize=(win, win), maxLevel=nivel,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    # st = 1 si encontró el punto; nos quedamos sólo con esos
    ok = st.ravel() == 1
    # Devuelve dos listas (N, 2): posición de cada punto encontrado en a y en b
    return p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)


def flujo_lucas_kanade(a, b, max_pts=150, win=21, nivel=3):
    # Flujo DISPERSO: no calcula todos los píxeles, sólo unos pocos puntos confiables.
    # 1. esquinas en a (GFTT)   2. dónde quedó cada una en b (Lucas-Kanade)
    return seguir_puntos(a, b, detectar_esquinas(a, max_pts), win, nivel)


def medir(fn, a, b, repeticiones=10):
    # Calcula el flujo y mide cuánto tarda. Se ejecuta varias veces y se toma la mediana,
    # porque una sola medición varía mucho según lo que esté haciendo la computadora.
    # Farnebäck y DIS devuelven una matriz (alto, ancho, 2);
    # Lucas-Kanade devuelve los puntos (p0, p1), porque sólo sigue esquinas
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        flujo = fn(a, b)
        tiempos.append(time.perf_counter() - t0)
    return flujo, 1000 * np.median(tiempos)


def con_titulo(img, texto):
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def flujo_a_hsv(flujo):
    # 1. Cada vector (fx, fy) pasa a polares: magnitud (cuánto) y ángulo (hacia dónde)
    mag, ang = cv2.cartToPolar(flujo[..., 0], flujo[..., 1]) #lopasamos a polares

    hsv = np.zeros((*flujo.shape[:2], 3), np.uint8)
    # 2. Matiz = dirección. El ángulo está en [0, 2π) y OpenCV usa H en [0, 180)
    hsv[..., 0] = ang * 180 / np.pi / 2
    # 3. Saturación al máximo para que los colores se vean intensos
    hsv[..., 1] = 255
    # 4. Brillo = magnitud, normalizada al percentil 99.5 para que unos pocos
    #    vectores enormes no dejen todo lo demás oscuro
    ref = max(np.percentile(mag, 99.5), 1e-3)
    hsv[..., 2] = np.clip(mag / ref * 255, 0, 255)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rueda_colores(tam=90):
    # Leyenda del mapa HSV: un flujo "falso" donde cada píxel apunta desde el centro hacia afuera,
    # así cada dirección queda pintada con su color (el centro, sin movimiento, es negro)
    y, x = np.mgrid[-1:1:tam * 1j, -1:1:tam * 1j]
    rueda = flujo_a_hsv(np.dstack([x, y]).astype(np.float32))
    rueda[np.hypot(x, y) > 1] = 255
    return rueda


def flechas_flujo(frame, flujo, paso=14, escala=4, umbral=0.6):
    # Otra forma de ver el mismo flujo: flechas sobre la imagen.
    # paso   -> una flecha cada `paso` píxeles (una por píxel taparía toda la imagen)
    # escala -> la flecha se dibuja `escala` veces más larga que el desplazamiento real
    # umbral -> no se dibujan los vectores menores a `umbral` px (ruido del fondo quieto)
    vis = frame.copy()
    for y in range(paso // 2, frame.shape[0], paso):
        for x in range(paso // 2, frame.shape[1], paso):
            fx, fy = flujo[y, x]
            if np.hypot(fx, fy) > umbral: #para que no se vean muchas flechas
                fin = (int(x + fx * escala), int(y + fy * escala))
                cv2.arrowedLine(vis, (x, y), fin, (0, 255, 255), 1, tipLength=0.4)
    return vis


def flechas_puntos(frame, p0, p1, escala=4, umbral=0.6):
    # Versión dispersa de flechas_flujo: las flechas salen de cada esquina seguida
    # (no de una grilla). Punto rojo = esquina quieta, flecha amarilla = esquina que se movió
    vis = frame.copy()
    for (x0, y0), (x1, y1) in zip(p0, p1):
        if np.hypot(x1 - x0, y1 - y0) > umbral:
            fin = (int(x0 + (x1 - x0) * escala), int(y0 + (y1 - y0) * escala))
            cv2.arrowedLine(vis, (int(x0), int(y0)), fin, (0, 255, 255), 1, tipLength=0.4)
        else:
            cv2.circle(vis, (int(x0), int(y0)), 2, (0, 0, 255), -1)
    return vis


def dibujar_esquinas(frame, p0):
    vis = frame.copy()
    for x, y in p0:
        cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------------
# Lucas-Kanade a lo largo del video
# ---------------------------------------------------------------------------
def leer_frames(ruta, inicio, n, escala=0.6):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, fps


def lucas_kanade_video(frames, max_pts=150, win=21, nivel=3, umbral=0.7, desvanecer=0.9, min_pts=40):
    # A diferencia de los métodos densos, que calculan cada par de cuadros desde cero,
    # acá se siguen LOS MISMOS puntos cuadro tras cuadro: lo que se encontró en el cuadro k
    # es el punto de partida para el k+1. Eso permite dibujar el recorrido (estela) de cada uno.
    #   desvanecer -> la estela anterior se multiplica por este valor en cada cuadro
    #                 (0.9 = se va apagando; 1.0 = queda el recorrido completo)
    #   min_pts    -> si quedan menos puntos, se vuelven a detectar esquinas (re-detección)
    g_ant = gris(frames[0])
    p0 = detectar_esquinas(g_ant, max_pts)
    estelas = np.zeros_like(frames[0])      # imagen negra donde se van dibujando los recorridos
    salida, seguidos, redetecciones, t_total = [], [], 0, 0.0

    for f in frames[1:]:
        g_act = gris(f)
        t0 = time.perf_counter()
        a_pts, b_pts = seguir_puntos(g_ant, g_act, p0, win, nivel)
        t_total += time.perf_counter() - t0

        # Las estelas viejas se apagan y se agrega el tramo nuevo de cada punto que se movió
        estelas = (estelas * desvanecer).astype(np.uint8)
        for (x0, y0), (x1, y1) in zip(a_pts, b_pts):
            if np.hypot(x1 - x0, y1 - y0) > umbral:
                cv2.line(estelas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1)

        # Cuadro actual + estelas encima + posición actual de cada punto en rojo
        vis = cv2.add(f, estelas)
        for x, y in b_pts:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.putText(vis, f'puntos seguidos: {len(b_pts)}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        salida.append(vis)
        seguidos.append(len(b_pts))

        # Los puntos encontrados pasan a ser los de partida del cuadro siguiente
        p0 = b_pts.reshape(-1, 1, 2)
        g_ant = g_act
        if len(p0) < min_pts:
            t0 = time.perf_counter()
            p0 = detectar_esquinas(g_ant, max_pts)
            t_total += time.perf_counter() - t0
            redetecciones += 1

    stats = {'ms_cuadro': 1000 * t_total / (len(frames) - 1), 'seguidos': seguidos,
             'redetecciones': redetecciones}
    return salida, stats


def reproducir(cuadros, fps):
    # Reproduce en loop: ESPACIO pausa/reanuda, Q o ESC sale
    i, pausa = 0, False
    while True:
        cv2.imshow(VENTANA, cuadros[i])
        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa
        if not pausa:
            i = (i + 1) % len(cuadros)


def guardar_video(cuadros, fps, nombre):
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / nombre
    h, w = cuadros[0].shape[:2]
    vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for c in cuadros:
        vw.write(c)
    vw.release()
    return ruta


if __name__ == '__main__':
    ESCALA, UMBRAL = 4, 0.6   # flechas x4 más largas; no se dibuja lo que se mueve menos de 0.6 px
    I0, SEP = 250, 2          # se comparan los cuadros I0 e I0 + SEP
    PASO_GRILLA = 14          # métodos densos: una flecha cada 14 px

    print('PASO 1 — Leer dos cuadros del video')
    f0, f1 = leer_cuadro(VIDEO, I0), leer_cuadro(VIDEO, I0 + SEP) #leemos los dos frames
    a, b = gris(f0), gris(f1) #leemos en gris
    print(f'  cuadros {I0} y {I0 + SEP}, tamaño {a.shape[1]}x{a.shape[0]}')
    print(f'  (separados {SEP} cuadros para que el movimiento sea más visible)')

    # ---------------- métodos DENSOS: un vector por píxel ----------------
    print('\nPASO 2 — Flujo DENSO: Farnebäck y DIS')
    print('  Los dos devuelven una matriz (alto, ancho, 2): un vector (fx, fy) por píxel.')
    print('  Farnebäck aproxima cada vecindad con un polinomio; DIS busca parches en una grilla')
    print('  gruesa y después densifica. Cada uno se ejecuta 10 veces (mediana del tiempo).')
    tiempos, paneles = {}, []
    for nombre, fn in (('Farneback', flujo_farneback), ('DIS', flujo_dis)):
        flujo, ms = medir(fn, a, b) #retorna una imagen de igual tamaño en dos canales (fx, fy)
        tiempos[nombre] = ms
        mag = np.hypot(flujo[..., 0], flujo[..., 1]) #calculamos la magnitud
        centro_ = flujo[a.shape[0] // 2, a.shape[1] // 2]
        print(f'  -> {nombre:<10} {ms:6.1f} ms | píxel del centro: fx = {centro_[0]:.2f}, fy = {centro_[1]:.2f} | '
              f'máx {mag.max():.1f} px | {100 * (mag > 1).mean():.1f} % de los píxeles se mueve > 1 px')

        hsv = flujo_a_hsv(flujo)
        rueda = rueda_colores()
        hsv[-rueda.shape[0] - 5:-5, -rueda.shape[1] - 5:-5] = rueda    # leyenda abajo a la derecha
        flechas = flechas_flujo(f0, flujo, PASO_GRILLA, ESCALA, UMBRAL)
        paneles.append(np.hstack([con_titulo(flechas, f'{nombre} - {ms:.1f} ms - flechas (x{ESCALA})'),
                                  con_titulo(hsv, f'{nombre} - HSV: color = direccion, brillo = magnitud')]))

    print('\n  HSV: negro = quieto | brillante = se mueve mucho | color = dirección (ver la rueda)')
    print(f'  flechas: una cada {PASO_GRILLA} px, sólo donde se mueve más de {UMBRAL} px, {ESCALA} veces más largas')
    print('  Ventana: arriba Farnebäck, abajo DIS. Las flechas apuntan hacia donde indica el color.')
    print('  Cualquier tecla para seguir.')
    cv2.imshow(VENTANA, np.vstack(paneles))
    cv2.waitKey(0)

    # ---------------- método DISPERSO: sólo esquinas ----------------
    print('\nPASO 3 — Flujo DISPERSO: GFTT + Lucas-Kanade (mismos dos cuadros)')
    print('  1. GFTT elige hasta 150 esquinas en el primer cuadro (puntos con textura en todas')
    print('     las direcciones: bordes de ventanas, carteles, conos, personas)')
    print('  2. Lucas-Kanade busca dónde quedó cada una en el segundo cuadro')
    (p0, p1), ms = medir(flujo_lucas_kanade, a, b)
    tiempos['Lucas-Kanade'] = ms
    desp = np.hypot(*(p1 - p0).T)                     # cuánto se movió cada esquina
    print(f'  -> {ms:.1f} ms | {len(p0)} esquinas seguidas (los densos calculan {a.size} vectores)')
    print(f'  -> desplazamiento máximo {desp.max():.1f} px | {(desp > UMBRAL).sum()} de {len(p0)} esquinas se mueven')
    print('  No hay mapa HSV: no hay un vector en cada píxel para pintar.')
    print('  Ventana: esquinas detectadas | flechas (amarilla = se movió, punto rojo = quieta).')
    print('  Cualquier tecla para seguir.')
    cv2.imshow(VENTANA, np.hstack([con_titulo(dibujar_esquinas(f0, p0), f'GFTT - {len(p0)} esquinas'),
                                   con_titulo(flechas_puntos(f0, p0, p1, ESCALA, UMBRAL),
                                              f'Lucas-Kanade - {ms:.1f} ms')]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 4 — Comparación de tiempos (mismo par de cuadros)')
    rapido = min(tiempos.values())
    for nombre, t in tiempos.items():
        print(f'  -> {nombre:<13}{t:8.1f} ms   ({t / rapido:4.1f}x el más rápido)   ~{1000 / t:4.0f} cuadros/s')
    print('  Los densos dan el movimiento de toda la imagen (siluetas completas); Lucas-Kanade sólo el')
    print('  de algunos puntos confiables, pero es el único que alcanza tiempo real con margen.')
    print('  (El tiempo de Lucas-Kanade incluye buscar las esquinas; en video eso se hace una sola vez.)')

    # ---------------- Lucas-Kanade a lo largo del video ----------------
    INICIO, N = 200, 80
    print(f'\nPASO 5 — Lucas-Kanade a lo largo del video ({N} cuadros desde el {INICIO})')
    print('  Los densos calculan cada par de cuadros desde cero; acá se siguen LOS MISMOS puntos:')
    print('  lo encontrado en un cuadro es el punto de partida del siguiente, así que cada punto')
    print('  mantiene su identidad y se puede dibujar su recorrido (estela).')
    print('  Los que se pierden (st = 0) se descartan; si quedan menos de 40 se vuelven a detectar esquinas.')
    frames, fps = leer_frames(VIDEO, INICIO, N)
    cuadros, st = lucas_kanade_video(frames)
    seg = st['seguidos']
    print(f'  -> {st["ms_cuadro"]:.2f} ms por cuadro (sólo el seguimiento)')
    print(f'  -> puntos seguidos: {seg[0]} al principio, {seg[-1]} al final, mínimo {min(seg)}; '
          f're-detecciones: {st["redetecciones"]}')
    ruta = guardar_video(cuadros, fps, 'lucas_kanade.mp4')
    print(f'  -> video guardado en {ruta}')
    print('  Ventana: punto rojo = posición actual | estela verde = recorrido reciente (se apaga 10 % por cuadro)')
    print('  ESPACIO pausa, Q o ESC cierra.')
    reproducir(cuadros, fps)
    cv2.destroyAllWindows()
