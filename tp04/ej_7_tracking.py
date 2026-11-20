"""Ejercicio 7 (sección 6 del TP) — Seguimiento de un objeto elegido por el usuario.

El tracker se elige con la constante TRACKER del bloque principal. El usuario elige el objeto con
el mouse sobre el primer cuadro y luego se sigue cuadro a cuadro.
Sobre la caja que da el tracker en cada cuadro se aplica una de las aplicaciones (APLICACION).

Controles:
    selección : arrastrar un rectángulo con el mouse, ENTER/ESPACIO confirma, C cancela
    seguimiento: ESPACIO pausa/reanuda, Q o ESC sale
"""
import time
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
VIDEO = DATOS / 'vtest.avi'
VENTANA = 'Seguimiento'


def abrir_video(ruta):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    return cap


def seleccionar_objeto(frame):
    # selectROI devuelve (x, y, w, h); si se cancela devuelve todo en cero
    bbox = cv2.selectROI(VENTANA, frame, showCrosshair=True, fromCenter=False)
    if bbox[2] == 0 or bbox[3] == 0:
        return None
    return bbox


# Todos comparten la misma interfaz (init / update), así que el resto del código no cambia.
# Los de cv2.legacy vienen del paquete contrib.
TRACKERS = {
    'CSRT': cv2.TrackerCSRT_create,               # preciso, tolera cambios de escala; lento
    'KCF': cv2.TrackerKCF_create,                 # rápido, escala fija
    'MIL': cv2.TrackerMIL_create,                 # robusto a oclusiones parciales; lento
    'MOSSE': cv2.legacy.TrackerMOSSE_create,      # el más rápido, el menos preciso
    'TLD': cv2.legacy.TrackerTLD_create,          # puede re-detectar tras una oclusión; muchos saltos
    'MedianFlow': cv2.legacy.TrackerMedianFlow_create,  # bueno con movimiento suave, avisa cuando falla
    'Boosting': cv2.legacy.TrackerBoosting_create,      # el más antiguo, de referencia
}


def crear_tracker(nombre):
    if nombre not in TRACKERS:
        raise ValueError(f'Tracker desconocido: {nombre}. Opciones: {", ".join(TRACKERS)}')
    return TRACKERS[nombre]()


def centro(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def dibujar_caja(frame, bbox):
    vis = frame.copy()
    x, y, w, h = (int(v) for v in bbox)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def dibujar_info(vis, nombre, fps, ok):
    # Texto común a todas las aplicaciones: tracker, fps de procesamiento y aviso de pérdida
    cv2.putText(vis, f'{nombre}  {fps:.0f} fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    if not ok:
        cv2.putText(vis, 'Objeto perdido', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Aplicación 1: REFLECTOR (resaltado + estela + velocidad)
# ---------------------------------------------------------------------------
def velocidad(tray, fps_video, ventana=5):
    #Calcula la velocidad que se mueve el objeto en pixeles por segundo respecto a 5 ventanas anteriores
    #tray lista de centros de la ventana 
    if len(tray) < 2:
        return 0.0
    k = min(ventana, len(tray) - 1) 
    (x0, y0), (x1, y1) = tray[-1 - k], tray[-1] #posicion actual y k frames antes
    return np.hypot(x1 - x0, y1 - y0) / k * fps_video #hipotenusa entre puntos, pixeles y fps


def aplicar_reflector(frame, bbox, tray, fps_video, oscuridad=0.3, largo_estela=60):
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = centro(bbox)

    # Máscara elíptica alrededor del objeto con borde difuso: 1 = iluminado, 0 = en sombra
    mascara = np.zeros((h, w), np.float32)
    cv2.ellipse(mascara, (int(cx), int(cy)), (int(bw * 0.8), int(bh * 0.7)), 0, 0, 360, 1.0, -1)
    mascara = cv2.GaussianBlur(mascara, (0, 0), sigmaX=15)[..., None]

    # Fuera del reflector la imagen se oscurece hasta `oscuridad`; adentro queda intacta
    vis = (frame * (oscuridad + (1 - oscuridad) * mascara)).astype(np.uint8)

    # Estela: los puntos más viejos se dibujan más finos y oscuros
    pts = tray[-largo_estela:]
    for i in range(1, len(pts)):
        a = i / len(pts)
        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p1 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(vis, p0, p1, (0, int(255 * a), 255), max(1, int(4 * a)), cv2.LINE_AA)

    v = velocidad(tray, fps_video)
    cv2.putText(vis, f'{v:.0f} px/s', (int(x), max(int(y) - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Aplicación 2: ANONIMIZACIÓN (pixelado o desenfoque sólo sobre el objeto)
# ---------------------------------------------------------------------------
def agrandar(bbox, factor, ancho_img, alto_img):
    # Agranda la caja alrededor de su centro y la recorta a los bordes de la imagen.
    # El margen extra cubre las partes del objeto que la caja del tracker deja afuera.
    cx, cy = centro(bbox)
    w, h = bbox[2] * factor, bbox[3] * factor
    x0, y0 = max(int(cx - w / 2), 0), max(int(cy - h / 2), 0)
    x1, y1 = min(int(cx + w / 2), ancho_img), min(int(cy + h / 2), alto_img)
    return x0, y0, x1, y1


def anonimizar(frame, bbox, modo='pixelado', margen=1.2, bloques=8):
    vis = frame.copy()
    x0, y0, x1, y1 = agrandar(bbox, margen, frame.shape[1], frame.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:      # la caja quedó fuera de la imagen
        return vis

    region = vis[y0:y1, x0:x1]
    if modo == 'pixelado':
        # Se achica la región a pocos píxeles y se vuelve a agrandar sin interpolar:
        # cada píxel chico se convierte en un bloque de color de `bloques` x `bloques`
        chica = cv2.resize(region, (max(1, (x1 - x0) // bloques), max(1, (y1 - y0) // bloques)),
                           interpolation=cv2.INTER_AREA)
        vis[y0:y1, x0:x1] = cv2.resize(chica, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    elif modo == 'desenfoque':
        # Gaussiano con kernel proporcional al tamaño de la región (debe ser impar)
        k = max(3, (min(x1 - x0, y1 - y0) // 3) | 1)
        vis[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)
    else:
        raise ValueError(f'Modo de anonimización desconocido: {modo}')
    return vis


# ---------------------------------------------------------------------------
# Aplicación 3: CÁMARA PTZ DIGITAL (recorte que sigue al objeto, sin mover la cámara)
# ---------------------------------------------------------------------------
def ptz_digital(frame, bbox, camara, zoom=2.5, alfa=0.12, salida=(640, 480)):
    # `camara` es el centro actual de la ventana de recorte (None en el primer cuadro).
    # Se devuelve actualizado para usarlo en el cuadro siguiente.
    h, w = frame.shape[:2]

    # Pan/tilt suave: la cámara no salta al centro del objeto, se acerca un `alfa` por cuadro.
    # alfa chico = movimiento más suave pero con más retraso; alfa = 1 = sin suavizado
    objetivo = centro(bbox)
    if camara is None:
        camara = objetivo
    camara = (camara[0] + alfa * (objetivo[0] - camara[0]),
              camara[1] + alfa * (objetivo[1] - camara[1]))

    # Zoom: la ventana mide 1/zoom del cuadro y tiene la misma proporción que la salida
    ancho_rec = int(w / zoom)
    alto_rec = int(ancho_rec * salida[1] / salida[0])

    # Esquina de la ventana, corrida para que no se salga de la imagen
    x0 = int(np.clip(camara[0] - ancho_rec / 2, 0, w - ancho_rec))
    y0 = int(np.clip(camara[1] - alto_rec / 2, 0, h - alto_rec))

    recorte = frame[y0:y0 + alto_rec, x0:x0 + ancho_rec]
    vis = cv2.resize(recorte, salida, interpolation=cv2.INTER_LINEAR)
    return vis, camara


def seguir(cap, tracker, nombre, aplicacion):
    #le mando el video y el tracker
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    tray = []           # centros del objeto en cada cuadro donde el tracker lo encontró
    ultima = None       # última caja válida: si el tracker pierde el objeto, se sigue usando esta
    camara = None       # centro de la cámara virtual de la PTZ digital
    tiempos, perdidos = [], 0
    pausa = False
    while True: #recorro todo el video
        if not pausa:
            ok_lectura, frame = cap.read() #obtengo el frame
            if not ok_lectura:
                break
            t0 = time.perf_counter()
            ok, bbox = tracker.update(frame)
            tiempos.append(time.perf_counter() - t0)
            fps = 1 / max(tiempos[-1], 1e-6)
            perdidos += not ok

            if ok:
                ultima = bbox #guarda la ultima caja optima
                tray.append(centro(bbox)) #se guarda los centros de las cajas del tracking

            # elijo qué aplicación dibujar sobre el cuadro
            if ultima is None:
                vis = frame.copy()
            elif aplicacion == 'reflector':
                vis = aplicar_reflector(frame, ultima, tray, fps_video)
            elif aplicacion == 'anonimizacion':
                vis = anonimizar(frame, ultima, modo='pixelado')  # o modo='desenfoque'
            elif aplicacion == 'ptz':
                vis, camara = ptz_digital(frame, ultima, camara)
            else:
                vis = dibujar_caja(frame, ultima)

            cv2.imshow(VENTANA, dibujar_info(vis, nombre, fps, ok))

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa

    return {'cuadros': len(tiempos), 'perdidos': perdidos,
            'ms': 1000 * float(np.mean(tiempos)) if tiempos else 0.0, 'tray': tray}


if __name__ == '__main__':
    # Tracker a usar: CSRT, KCF, MIL, MOSSE, TLD, MedianFlow o Boosting
    TRACKER = 'CSRT'
    # Aplicación: 'caja' (solo seguimiento), 'reflector', 'anonimizacion' o 'ptz'
    APLICACION = 'ptz'

    print('PASO 1 — Selección del objeto')
    print('  En el primer cuadro, arrastrá un rectángulo alrededor de la persona a seguir.')
    print('  ENTER o ESPACIO confirma, C cancela.')
    cap = abrir_video(VIDEO) #obtenemos el primer frame del video
    ok, primer_frame = cap.read()
    if not ok:
        raise RuntimeError('El video no tiene cuadros')
    bbox = seleccionar_objeto(primer_frame)# seleccionamos objeto

    # tracking
    if bbox is None:
        print('Selección cancelada')
    else:
        print(f'  -> caja inicial (x, y, ancho, alto) = {tuple(int(v) for v in bbox)}')

        print(f'\nPASO 2 — Seguimiento con {TRACKER}')
        print('  El tracker aprende cómo se ve el objeto en la caja inicial y en cada cuadro busca')
        print('  dónde está ahora (update). Si lo pierde, se sigue usando la última caja válida.')
        print('  Opciones: CSRT (preciso, lento), KCF (rápido), MOSSE (el más rápido), TLD (re-detecta')
        print('  tras una oclusión pero salta), MIL, MedianFlow, Boosting. Se cambia con TRACKER.')

        print(f'\nPASO 3 — Aplicación: {APLICACION}')
        print({
            'caja': '  sólo la caja del tracker.',
            'reflector': '  se oscurece todo menos el objeto, se dibuja su estela y su velocidad (px/s).',
            'anonimizacion': '  se pixela sólo la caja del objeto (agrandada 20 %); el resto queda intacto.',
            'ptz': '  una ventana de tamaño fijo (zoom 2.5x) sigue al objeto con movimiento suave\n'
                   '  y se amplía a 640x480: una cámara PTZ simulada sin mover la cámara.',
        }[APLICACION])
        print('  Se cambia con APLICACION: caja, reflector, anonimizacion o ptz.')
        print('\n  Ventana: arriba a la izquierda el tracker y sus fps; "Objeto perdido" cuando falla.')
        print('  ESPACIO pausa, Q o ESC sale.')

        tracker = crear_tracker(TRACKER) #creamos el tracking
        tracker.init(primer_frame, bbox) # lo inicializamos
        r = seguir(cap, tracker, TRACKER, APLICACION) #seguimos el objeto y aplicamos la aplicación elegida

        print('\nPASO 4 — Resultados')
        if r['cuadros']:
            print(f'  -> {r["cuadros"]} cuadros seguidos con {TRACKER}: {r["ms"]:.1f} ms por cuadro '
                  f'(~{1000 / max(r["ms"], 1e-6):.0f} fps)')
            print(f'  -> objeto perdido en {r["perdidos"]} cuadros ({100 * r["perdidos"] / r["cuadros"]:.0f} %)')
            print('  Para comparar trackers: repetir con otro TRACKER eligiendo la misma persona,')
            print('  idealmente una que pase detrás del poste (oclusión).')

    cap.release()
    cv2.destroyAllWindows()
