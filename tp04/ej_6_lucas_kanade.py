"""Ejercicio 6 — Estabilización de video (Lucas-Kanade + RANSAC).

El video de muestra fue grabado con cámara fija, así que primero se le agrega un temblor
artificial conocido; el video original sirve de referencia para evaluar la estabilización.
  1. Seguir esquinas (GFTT + Lucas-Kanade) entre cuadros consecutivos.
  2. Estimar el movimiento global de la cámara (similaridad con estimateAffinePartial2D + RANSAC).
  3. Acumular los movimientos: trayectoria de la cámara.
  4. Suavizar la trayectoria con una media móvil.
  5. Corregir cada cuadro con la diferencia entre la trayectoria suavizada y la observada.
  6. Comparar la dispersión de la trayectoria antes y después.

Controles del reproductor: ESPACIO pausa/reanuda, Q o ESC sale.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATOS = Path(__file__).parent / 'datos'
SALIDAS = Path(__file__).parent / 'salidas'
VIDEO = DATOS / 'vtest.avi'
VENTANA = 'Estabilizacion'


def leer_frames(ruta, max_frames=200):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


# ---------------------------------------------------------------------------
# Temblor artificial
# ---------------------------------------------------------------------------
def matriz_temblor(tx, ty, ang, w, h):
    # Matriz 3x3 (2x3 + fila [0 0 1]) de rotación alrededor del centro + traslación.
    # La fila extra permite componer transformaciones multiplicando matrices
    M = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), [0, 0, 1]])
    M[:2, 2] += (tx, ty)
    return M


def agregar_temblor(frames, amp=8.0, amp_rot=1.2, seed=0):
    # Cada cuadro se desplaza (tx, ty) píxeles y se rota `ang` grados alrededor del centro.
    # El 0.5 * valor_anterior hace que el temblor tenga cierta continuidad entre cuadros
    # (como una cámara real que vibra) en vez de saltar al azar en cada cuadro.
    # Devuelve los cuadros temblorosos y el temblor aplicado, una fila (tx, ty, ang) por cuadro.
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    tx = ty = ang = 0.0 #desplazamietno para toda la imagen
    salida, temblor = [], []
    for f in frames:
        tx = 0.5 * tx + rng.normal(0, amp * 0.5)
        ty = 0.5 * ty + rng.normal(0, amp * 0.5)
        ang = 0.5 * ang + rng.normal(0, amp_rot * 0.4)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0) #armamos una matriz de rotacion
        # el centro donde se gira y cuantos grados se estira (sin zoom)
        M[:, 2] += (tx, ty) #le suma el temblor (solo la ultima columna)
        # BORDER_REPLICATE rellena el borde que queda vacío al mover la imagen
        salida.append(cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)) #rellena los bordes para que no se vea negro
        temblor.append((tx, ty, ang))
    return salida, np.array(temblor)


# ---------------------------------------------------------------------------
# 1-2. Movimiento de la cámara entre cuadros (GFTT + Lucas-Kanade + RANSAC)
# ---------------------------------------------------------------------------
def gris(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def detectar_esquinas(g, max_puntos=300):
    # GFTT: busca puntos donde la imagen cambia en todas las direcciones (esquinas),
    # que son los únicos que se pueden ubicar sin ambigüedad en el cuadro siguiente.
    # 0.01 = calidad mínima relativa a la mejor esquina; 12 = distancia mínima entre puntos
    return cv2.goodFeaturesToTrack(g, max_puntos, 0.01, 12, blockSize=7)


def seguir_esquinas(g_ant, g_act, p0):
    # Lucas-Kanade piramidal: para cada punto de g_ant busca a dónde se movió en g_act.
    # st[i] = 1 si lo encontró; nos quedamos sólo con esos pares (antes, después)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g_ant, g_act, p0, None)
    ok = st.ravel() == 1
    return p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)


def parametros(M):
    # Desarma una matriz 2x3 de similaridad en (dx, dy, ángulo en grados)
    return M[0, 2], M[1, 2], np.degrees(np.arctan2(M[1, 0], M[0, 0]))


def estimar_movimiento(p0, p1, umbral=3.0):
    # Una sola transformación (traslación + rotación + escala) que lleve p0 a p1.
    # RANSAC prueba con pares al azar y se queda con la que más puntos explica;
    # los puntos a más de `umbral` px de esa transformación son outliers (personas).
    # Supone que el fondo domina la escena: si casi todo se mueve, la estimación falla
    M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=umbral)
    if M is None:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float64), np.zeros(len(p0), bool)
    return M, inliers.ravel() == 1


def movimientos_camara(frames, umbral=3.0):
    # Movimiento entre cada par de cuadros consecutivos (0->1, 1->2, ...).
    # Devuelve una fila (dx, dy, ang) por paso: N cuadros dan N-1 movimientos
    movs = []
    g_ant = gris(frames[0])
    for f in frames[1:]:
        g_act = gris(f)
        p0, p1 = seguir_esquinas(g_ant, g_act, detectar_esquinas(g_ant))
        M, _ = estimar_movimiento(p0, p1, umbral)
        movs.append(parametros(M))
        g_ant = g_act
    return np.array(movs)


# ---------------------------------------------------------------------------
# 3-5. Trayectoria, suavizado y corrección
# ---------------------------------------------------------------------------
def trayectoria(movs):
    # Posición de la cámara en cada cuadro = suma de todos los pasos hasta ese cuadro.
    # Se agrega una fila de ceros al principio: en el cuadro 0 la cámara está en el origen
    return np.vstack([np.zeros(3), np.cumsum(movs, axis=0)])


def suavizar(tray, radio=15):
    # Media móvil: cada punto pasa a ser el promedio de los `radio` cuadros anteriores y
    # posteriores. Elimina las vibraciones rápidas y conserva los movimientos lentos (paneos).
    # Los bordes se rellenan repitiendo el primer/último valor para no perder cuadros
    k = np.ones(2 * radio + 1) / (2 * radio + 1)
    pad = np.pad(tray, ((radio, radio), (0, 0)), mode='edge')
    return np.column_stack([np.convolve(pad[:, j], k, mode='valid') for j in range(tray.shape[1])])


def matriz_correccion(dx, dy, ang):
    # Matriz 3x3 de rotación (en grados) + traslación, con la misma convención que parametros()
    a = np.radians(ang)
    return np.array([[np.cos(a), -np.sin(a), dx], [np.sin(a), np.cos(a), dy], [0, 0, 1]])


def estabilizar(frames, movs, radio=15):
    # corrección = trayectoria deseada (suave) - trayectoria observada (con temblor):
    # lo que hay que mover cada cuadro para llevarlo de donde está a donde "debería" estar
    tray = trayectoria(movs)
    tray_suave = suavizar(tray, radio)
    correcciones = [matriz_correccion(*c) for c in tray_suave - tray]

    h, w = frames[0].shape[:2]
    salida = [cv2.warpAffine(f, C[:2], (w, h), borderMode=cv2.BORDER_REPLICATE)
              for f, C in zip(frames, correcciones)]
    return salida, correcciones, tray, tray_suave


# ---------------------------------------------------------------------------
# 6. Evaluación y salida
# ---------------------------------------------------------------------------
def evaluar(temblor, w, h, correcciones=None):
    # Como conocemos el temblor, sabemos exactamente cómo quedó cada cuadro respecto del
    # original: R_i = corrección_i · temblor_i (sin corrección, R_i = temblor_i).
    # Medimos dónde terminan las 4 esquinas y el centro de la imagen:
    #   salto      = cuánto saltan esos puntos de un cuadro al siguiente [px/cuadro]
    #   dispersión = cuánto se alejan, en promedio, de su posición media [px]
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1], [w / 2, h / 2, 1]], float).T
    pos = []
    for i, t in enumerate(temblor):
        R = matriz_temblor(*t, w, h)
        if correcciones is not None:
            R = correcciones[i] @ R
        pos.append((R @ pts)[:2].T)
    pos = np.array(pos)                                       # (cuadros, 5 puntos, xy)
    salto = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean()
    dispersion = np.linalg.norm(pos - pos.mean(axis=0), axis=2).mean()
    return salto, dispersion


def graficar_trayectoria(tray, tray_suave, radio):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for j, (ax, nombre) in enumerate(zip(axs, ['x [px]', 'y [px]', 'ángulo [°]'])):
        ax.plot(tray[:, j], label='con temblor', lw=1)
        ax.plot(tray_suave[:, j], label=f'suavizada (radio {radio})', lw=3, alpha=0.6)
        ax.set_ylabel(nombre)
        ax.grid(alpha=0.3)
    axs[0].legend()
    axs[0].set_title('Trayectoria de la cámara: observada y deseada')
    axs[-1].set_xlabel('cuadro')
    plt.tight_layout()
    plt.show()


def recortar_bordes(f, zoom=1.08):
    # Zoom leve desde el centro para esconder los bordes replicados que deja la corrección
    h, w = f.shape[:2]
    return cv2.warpAffine(f, cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom), (w, h))


def lado_a_lado(izq, der, titulo_izq, titulo_der):
    vis = np.hstack([izq, der])
    for texto, x in ((titulo_izq, 10), (titulo_der, izq.shape[1] + 10)):
        cv2.putText(vis, texto, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis


def guardar_video(cuadros, fps, nombre):
    SALIDAS.mkdir(exist_ok=True)
    ruta = SALIDAS / nombre
    h, w = cuadros[0].shape[:2]
    vw = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for c in cuadros:
        vw.write(c)
    vw.release()
    return ruta


def reproducir(cuadros, fps):
    # Reproduce en loop una lista de cuadros ya armados
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


if __name__ == '__main__':
    N_CUADROS = 200
    RADIO = 15              # radio de la media móvil (cuadros hacia cada lado)
    UMBRAL_RANSAC = 3.0     # ransacReprojThreshold [px]

    print('PASO 1 — Video con temblor artificial')
    frames, fps = leer_frames(VIDEO, max_frames=N_CUADROS) # leemos los cuadros del video original
    h, w = frames[0].shape[:2]
    temblorosos, temblor = agregar_temblor(frames) # le agregamos el temblor artificial
    print('  El video original es de cámara fija: a cada cuadro se le aplica una traslación + rotación')
    print('  aleatoria (con continuidad entre cuadros). El original queda como referencia.')
    print(f'  -> {len(frames)} cuadros de {w}x{h}; temblor medio |tx| {np.abs(temblor[:, 0]).mean():.1f} px, '
          f'|ty| {np.abs(temblor[:, 1]).mean():.1f} px, |ang| {np.abs(temblor[:, 2]).mean():.2f}°')

    print('\nPASO 2 — Movimiento de la cámara entre cuadros consecutivos')
    print('  1. GFTT: hasta 300 esquinas en el cuadro anterior')
    print('  2. Lucas-Kanade: dónde quedó cada una en el cuadro actual')
    print(f'  3. estimateAffinePartial2D + RANSAC (umbral {UMBRAL_RANSAC} px): la similaridad que explica')
    print('     a la mayoría de los puntos. Como casi todas las esquinas son del fondo, esa es la')
    print('     cámara; las de las personas que caminan quedan como outliers.')
    print('  Calculando...')
    movs = movimientos_camara(temblorosos, UMBRAL_RANSAC)
    print(f'  -> {len(movs)} movimientos estimados (dx, dy, ángulo), uno por par de cuadros')

    print('\nPASO 3 — Trayectoria y suavizado')
    print('  La trayectoria de la cámara es la suma acumulada de los movimientos. Se suaviza con una')
    print(f'  media móvil de radio {RADIO} (promedio de {2 * RADIO + 1} cuadros): queda el movimiento lento')
    print('  (el "deseado") y se van las vibraciones rápidas.')
    estabilizados, correcciones, tray, tray_suave = estabilizar(temblorosos, movs, RADIO)
    print('  Gráfico: trayectoria con temblor (fina) y suavizada (gruesa). Cerrarlo para seguir.')
    graficar_trayectoria(tray, tray_suave, RADIO)

    print('\nPASO 4 — Corrección de cada cuadro')
    print('  corrección = trayectoria suavizada - trayectoria observada; se aplica con warpAffine.')
    print('  La corrección deja bordes vacíos (rellenados con BORDER_REPLICATE): un zoom del 8 % los esconde.')

    print('\nPASO 5 — Evaluación: dispersión antes y después')
    print('  Como conocemos el temblor, sabemos dónde terminó cada cuadro respecto del original.')
    print('  salto = cuánto se mueve la imagen de un cuadro al siguiente | dispersión = cuánto se aleja')
    print('  de su posición media.')
    print(f'  {"":<14}{"salto [px/cuadro]":>20}{"dispersión [px]":>18}')
    print(f'  {"original":<14}{0:>20.2f}{0:>18.2f}')
    for nombre, corr in (('con temblor', None), ('estabilizado', correcciones)):
        salto, disp = evaluar(temblor, w, h, corr)
        print(f'  {nombre:<14}{salto:>20.2f}{disp:>18.2f}')
    print('  La dispersión no llega a 0: el estabilizado sigue la trayectoria SUAVIZADA, que todavía')
    print('  tiene los movimientos lentos del temblor.')

    comparacion = [lado_a_lado(recortar_bordes(a), recortar_bordes(b), 'Con temblor', 'Estabilizado')
                   for a, b in zip(temblorosos, estabilizados)]
    ruta = guardar_video(comparacion, fps, 'estabilizacion.mp4')
    print(f'\n  -> video guardado en {ruta}')
    print('  Ventana: con temblor | estabilizado. ESPACIO pausa, Q o ESC sale.')
    reproducir(comparacion, fps)
    cv2.destroyAllWindows()
