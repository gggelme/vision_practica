"""Ejercicio 4 — Seguimiento multi-objeto (estilo SORT): conteo, mapa de calor y velocidad.

Arquitectura:
  1. Detección: MOG2 + limpieza morfológica + findContours -> cajas.
  2. Predicción: cada objeto tiene un filtro de Kalman [x, y, vx, vy] (velocidad constante).
  3. Asociación: distancias entre predicciones y detecciones + algoritmo húngaro.
  4. Ciclo de vida: detección sin pista -> pista nueva; pista sin detección -> sobrevive
     unos cuadros con la predicción y después se cierra.
Sobre las trayectorias: conteo por línea, mapa de calor, tiempo de permanencia y velocidad.

Controles: ESPACIO pausa/reanuda, Q o ESC sale.
"""
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DATOS = Path(__file__).parent / 'datos'
SALIDAS = Path(__file__).parent / 'salidas'
VIDEO = DATOS / 'vtest.avi'
VENTANA = 'Multiobjeto'
ESCALA = 0.6            # los cuadros se reducen al 60 % para procesar más rápido


# ---------------------------------------------------------------------------
# 1. Detección (MOG2 + post-proceso, como en el ejercicio 2)
# ---------------------------------------------------------------------------
K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def crear_sustractor():
    # history = cuántos cuadros recuerda el modelo de fondo
    # varThreshold = qué tan distinto tiene que ser un píxel para ser primer plano
    # detectShadows = marca las sombras con 127 (en vez de 255) para poder descartarlas
    return cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30, detectShadows=True)


def limpiar_mascara(m):
    # 1. Umbral > 200: se queda sólo con el primer plano (255) y descarta las sombras (127)
    _, b = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
    # 2. Apertura: elimina puntos sueltos de ruido
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, K_OPEN)
    # 3. Cierre: rellena huecos y une partes de una misma persona
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, K_CLOSE)


def detectar_blobs(mascara, area_min=90):
    # Cada región blanca conectada es un "blob"; se descartan las más chicas que area_min (ruido).
    # Devuelve una lista de cajas (x, y, w, h)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= area_min]


def detectar(sustractor, frame):
    # Detección completa de un cuadro: suavizado -> MOG2 -> limpieza -> cajas
    m = sustractor.apply(cv2.GaussianBlur(frame, (5, 5), 0))
    limpia = limpiar_mascara(m)
    return detectar_blobs(limpia), limpia


# ---------------------------------------------------------------------------
# 2-4. Seguimiento: Kalman + húngaro + ciclo de vida
# ---------------------------------------------------------------------------
def centro(caja):
    x, y, w, h = caja
    return x + w / 2, y + h / 2


class Pista:
    # Un objeto seguido: su identidad (id), su última caja, el recorrido de su centro
    # y su propio filtro de Kalman.
    # Estado = [x, y, vx, vy] (posición y velocidad), medición = [x, y] (el centro de la caja).
    _n = 0      # contador global para dar ids únicos

    def __init__(self, caja, cuadro):
        Pista._n += 1
        self.id = Pista._n #el id generado
        self.caja = caja #la caja que tiene
        self.tray = [centro(caja)] #ultima trayectoria
        self.inicio = cuadro
        self.hits = 1           # en cuántos cuadros se la asoció con una detección
        self.perdidos = 0       # cuántos cuadros seguidos lleva sin detección
        self.lado = 0           # de qué lado de la línea de conteo está (-1, +1; 0 = todavía no se sabe)

        cx, cy = centro(caja)
        kf = cv2.KalmanFilter(4, 2)
        # Modelo de velocidad constante: en cada cuadro x' = x + vx, y' = y + vy (la velocidad no cambia)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        # Sólo se mide la posición; la velocidad el filtro la deduce de cómo cambian las posiciones
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        # Ruido del modelo: cuánto puede apartarse la realidad de "velocidad constante"
        # (más en la velocidad que en la posición: la gente acelera, frena y dobla)
        kf.processNoiseCov = np.diag([1, 1, 4, 4]).astype(np.float32) * 0.05
        # Ruido de la medición: cuánto tiembla el centro de la caja de MOG2 (~2 px)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        # Estado inicial: donde se la detectó, con velocidad 0 (todavía no se sabe hacia dónde va)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf = kf

    def predecir(self):
        # PREDICCIÓN: dónde va a estar en este cuadro según su posición y velocidad.
        # Ojo: avanza el filtro un cuadro, así que se llama UNA vez por cuadro (lo hace el Seguidor).
        # Si la pista está perdida, la predicción sigue avanzando cuadro a cuadro
        p = self.kf.predict()
        return float(p[0, 0]), float(p[1, 0])

    def asociar(self, caja):
        # CORRECCIÓN: el filtro combina lo que predijo con lo que se midió (el centro de la caja),
        # pesando cada uno según su ruido, y de paso ajusta la velocidad estimada
        cx, cy = centro(caja)
        e = self.kf.correct(np.array([[cx], [cy]], np.float32))
        self.caja = caja
        self.tray.append((float(e[0, 0]), float(e[1, 0])))     # posición filtrada (más suave)
        self.hits += 1
        self.perdidos = 0

    def posicion(self):
        # Posición actual estimada por el filtro (sólo lectura, no avanza nada)
        return float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])

    def velocidad(self):
        # Velocidad estimada por el filtro, en píxeles por cuadro: sqrt(vx² + vy²)
        return float(np.hypot(self.kf.statePost[2, 0], self.kf.statePost[3, 0]))


class Seguidor:
    # clase que maneja todas las personas. Tiene lista de pistas y decide quien es nuevo y quien se fue
    def __init__(self, dist_max=28, max_perdidos=6, min_hits=5):
        # dist_max     -> distancia máxima (px) para aceptar que una detección es de una pista
        # max_perdidos -> cuadros que una pista sobrevive sin detección antes de cerrarse
        # min_hits     -> detecciones necesarias para confirmar una pista (y mostrarla / contarla).
        #                 Filtra fantasmas: ruido que aparece uno o dos cuadros y desaparece
        self.dist_max, self.max_perdidos, self.min_hits = dist_max, max_perdidos, min_hits
        self.pistas, self.cerradas = [], []

    def actualizar(self, cajas, cuadro): #recibe las cajas detectadas, cuadro actual
        # 1. Dónde predice el Kalman cada pista y dónde están las detecciones nuevas
        esperadas = np.array([p.predecir() for p in self.pistas]).reshape(-1, 2) #predicción de kalman de cada pista
        detectadas = np.array([centro(c) for c in cajas]).reshape(-1, 2) #calcula los centros de las cajas detectadas

        # 2. Asociación: matriz de distancias pista x detección; el algoritmo húngaro elige
        #    la combinación de pares que minimiza la distancia TOTAL (no la mejor para cada uno)
        asignadas, usadas = set(), set()
        if len(esperadas) and len(detectadas): #hay pistas y detecciones!
            C = cdist(esperadas, detectadas) #calculamos la distancia entre la prediccion de kalman y la nueva
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] <= self.dist_max:     # un par demasiado lejos no es el mismo objeto
                    self.pistas[i].asociar(cajas[j])
                    asignadas.add(i)
                    usadas.add(j)

        # 3. Pistas sin detección: suman un cuadro perdido
        for i, p in enumerate(self.pistas):
            if i not in asignadas:
                p.perdidos += 1

        # 4. Detecciones sin pista: objeto nuevo
        for j, c in enumerate(cajas):
            if j not in usadas:
                self.pistas.append(Pista(c, cuadro))

        # 5. Las pistas perdidas demasiado tiempo se cierran
        vivas = []
        for p in self.pistas:
            (self.cerradas if p.perdidos > self.max_perdidos else vivas).append(p)
        self.pistas = vivas

        # Se devuelven las pistas confirmadas y vistas en este cuadro
        return [p for p in self.pistas if p.hits >= self.min_hits and p.perdidos == 0]


# ---------------------------------------------------------------------------
# Aplicaciones: conteo por línea, permanencia, velocidad, mapa de calor
# ---------------------------------------------------------------------------
def dist_con_signo(p, a, b):
    # Distancia de p a la recta AB, con signo: el signo dice de qué lado de la línea está
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / np.hypot(b[0] - a[0], b[1] - a[1])


def en_zona(p, z):
    return z[0] <= p[0] <= z[0] + z[2] and z[1] <= p[1] <= z[1] + z[3]


class Analisis:
    # Acumula, cuadro a cuadro, lo que se mide sobre las pistas confirmadas
    def __init__(self, forma, linea, zona, fps, metros_por_pixel=None, hist=4.0):
        # hist = histéresis: zona de ±hist px alrededor de la línea donde no se decide el lado.
        # Evita contar dos veces a alguien que oscila justo encima de la línea
        self.linea, self.zona, self.fps = linea, zona, fps
        self.mpp, self.hist = metros_por_pixel, hist
        self.conteo = {'A→B': 0, 'B→A': 0}
        self.eventos = []                           # (cuadro, id, sentido)
        self.calor = np.zeros(forma, np.float32)    # cuántas veces pasó un centro por cada píxel
        self.dwell = defaultdict(float)             # segundos dentro de la zona, por id
        self.vel = defaultdict(list)                # velocidad en cada cuadro, por id
        self.serie = []                             # cuántos objetos hay en cada cuadro

    def actualizar(self, activas, cuadro):
        h, w = self.calor.shape
        for p in activas:
            c = p.posicion()
            # mapa de calor: se suma 1 en el píxel donde está el centro
            self.calor[int(np.clip(c[1], 0, h - 1)), int(np.clip(c[0], 0, w - 1))] += 1
            # conteo: si cambió de lado de la línea (y está lejos de ella), cruzó
            d = dist_con_signo(c, *self.linea)
            if abs(d) > self.hist:
                lado = 1 if d > 0 else -1
                if p.lado != 0 and lado != p.lado:
                    sentido = 'A→B' if lado < 0 else 'B→A'
                    self.conteo[sentido] += 1
                    self.eventos.append((cuadro, p.id, sentido))
                p.lado = lado
            # permanencia: cada cuadro dentro de la zona suma 1/fps segundos
            if en_zona(c, self.zona):
                self.dwell[p.id] += 1 / self.fps
            # velocidad: el Kalman la da en px/cuadro; x fps = px/s; x metros/px = m/s
            self.vel[p.id].append(p.velocidad() * self.fps * (self.mpp or 1))
        self.serie.append(len(activas))


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def color_de(id_):
    # Color fijo para cada id (el mismo id siempre se dibuja igual)
    rng = np.random.default_rng(id_)
    return tuple(int(v) for v in rng.integers(60, 256, 3))


def dibujar(frame, activas, perdidas, an, largo_estela=40):
    vis = frame.copy()
    # Línea de conteo (amarilla) y zona de permanencia (naranja): referencias que definimos nosotros
    a, b = an.linea
    cv2.line(vis, a, b, (0, 255, 255), 2)
    cv2.putText(vis, 'A', (a[0] - 18, a[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(vis, 'B', (a[0] + 8, a[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    zx, zy, zw, zh = an.zona
    cv2.rectangle(vis, (zx, zy), (zx + zw, zy + zh), (0, 165, 255), 2)
    for p in activas:
        col = color_de(p.id)
        x, y, w, h = p.caja
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cv2.putText(vis, f'#{p.id}', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        pts = np.array(p.tray[-largo_estela:], np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], False, col, 1)
    # Pistas perdidas (sin detección en este cuadro): un círculo donde las predice el Kalman
    for p in perdidas:
        cx, cy = (int(v) for v in p.posicion())
        col = color_de(p.id)
        cv2.circle(vis, (cx, cy), 6, col, 1)
        cv2.putText(vis, f'#{p.id}?', (cx + 7, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    cv2.putText(vis, f"A->B: {an.conteo['A→B']}   B->A: {an.conteo['B→A']}   activos: {len(activas)}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Recorrido del video
# ---------------------------------------------------------------------------
def procesar(ruta, linea, zona, metros_por_pixel=None, calentamiento=30, nombre_video='multiobjeto.mp4'):
    # Recorre el video en vivo. calentamiento: los primeros cuadros MOG2 todavía está
    # aprendiendo el fondo y marca casi todo como primer plano, así que no se detecta nada
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}') #nos paramos en el frame inicial
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    sustractor = crear_sustractor() #Mog
    Pista._n = 0 #una pista es una persona que estamos siguiendo
    seguidor = Seguidor()
    an, vw, ruta_video = None, None, None
    para_fondo = []         # un cuadro cada 8, para la imagen de fondo del análisis
    i, pausa = 0, False

    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
            if an is None:
                an = Analisis(frame.shape[:2], linea, zona, fps, metros_por_pixel)
                SALIDAS.mkdir(exist_ok=True)
                ruta_video = SALIDAS / nombre_video
                vw = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (frame.shape[1], frame.shape[0]))

            cajas, mascara = detectar(sustractor, frame) #retorna las cajas y las mascaras
            if i < calentamiento: #esta calentando
                cajas = []

            activas = seguidor.actualizar(cajas, i)
            an.actualizar(activas, i)
            perdidas = [p for p in seguidor.pistas if p.perdidos > 0 and p.hits >= seguidor.min_hits]
            vis = dibujar(frame, activas, perdidas, an)
            vw.write(vis)
            if i % 8 == 0:
                para_fondo.append(frame)

            # A la derecha, la máscara limpia: lo que MOG2 considera primer plano
            cv2.imshow(VENTANA, np.hstack([vis, cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)]))
            i += 1

        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa

    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    # Fondo de referencia: mediana temporal (las personas, que se mueven, desaparecen)
    fondo = np.median(np.stack(para_fondo), axis=0).astype(np.uint8)
    todas = seguidor.cerradas + seguidor.pistas
    return {'cuadros': i, 'fps': fps, 'an': an, 'fondo': fondo, 'ruta_video': ruta_video,
            'pistas': [p for p in todas if p.hits >= seguidor.min_hits], 'ids_usados': Pista._n}


# ---------------------------------------------------------------------------
# Análisis final: mapa de calor, trayectorias, ocupación, permanencia y velocidad
# ---------------------------------------------------------------------------
def mapa_de_calor(fondo, calor):
    # Se suaviza (cada centro aporta a una mancha, no a un solo píxel), se normaliza 0-1
    # y se superpone en colores sólo donde hay suficiente circulación
    cal = cv2.GaussianBlur(calor, (0, 0), 9)
    cal = cal / (cal.max() + 1e-9)
    cal_color = cv2.applyColorMap((cal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return np.where(cal[..., None] > 0.05, cv2.addWeighted(fondo, 0.45, cal_color, 0.55, 0), fondo)


def imagen_trayectorias(fondo, pistas, linea):
    img = fondo.copy()
    for p in pistas:
        pts = np.array(p.tray, np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color_de(p.id), 2)
        cv2.circle(img, tuple(int(v) for v in pts[-1, 0]), 4, (0, 0, 255), -1)    # última posición
    cv2.line(img, *linea, (0, 255, 255), 2)
    return img


def graficar_analisis(r, metros_por_pixel):
    an, fps = r['an'], r['fps']
    calor = mapa_de_calor(r['fondo'], an.calor)
    tray = imagen_trayectorias(r['fondo'], r['pistas'], an.linea)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].imshow(cv2.cvtColor(calor, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Mapa de calor de circulación')
    ax[1].imshow(cv2.cvtColor(tray, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Trayectorias + línea de conteo')
    for a in ax:
        a.axis('off')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    ax[0].plot(np.arange(len(an.serie)) / fps, an.serie)
    ax[0].set(xlabel='tiempo [s]', ylabel='# objetos', title='Ocupación de la escena')
    ids = sorted(an.dwell, key=an.dwell.get, reverse=True)[:10]
    ax[1].bar([f'#{i}' for i in ids], [an.dwell[i] for i in ids], color='#ff8c00')
    ax[1].set(ylabel='s en la zona', title='Tiempo de permanencia (zona naranja)')
    vm = velocidades_medianas(an)
    ax[2].hist(list(vm.values()), bins=12, color='#4285f4')
    ax[2].set(xlabel='m/s' if metros_por_pixel else 'px/s', title='Velocidad mediana por objeto')
    plt.tight_layout()
    plt.show()


def velocidades_medianas(an, min_cuadros=15):
    # Una velocidad representativa por objeto: la mediana (robusta a los cuadros donde el
    # Kalman todavía no estimó bien la velocidad); sólo objetos seguidos al menos 1.5 s
    return {i: float(np.median(v)) for i, v in an.vel.items() if len(v) > min_cuadros}


if __name__ == '__main__':
    # Geometría de la escena, en el cuadro reducido (461x346); ajustar para otro video
    LINEA = ((285, 96), (285, 249))          # línea de conteo, de arriba (P1) hacia abajo (P2)
    ZONA = (253, 155, 138, 103)              # x, y, ancho, alto: zona de permanencia
    METROS_POR_PIXEL = None                  # con calibración (p. ej. 0.05) las velocidades salen en m/s

    print('PASO 1 — Detección con MOG2')
    print('  MOG2 marca lo que cambia (primer plano = 255, sombra = 127); se descartan las sombras,')
    print('  apertura (fuera ruido) + cierre (rellenar huecos); cada blob se convierte en una caja.')
    print('  Los primeros 30 cuadros MOG2 está aprendiendo el fondo: no se detecta nada.')

    print('\nPASO 2 — Predicción con Kalman')
    print('  Cada pista tiene un Kalman con estado [x, y, vx, vy] y modelo de velocidad constante:')
    print('  antes de cada cuadro predice dónde va a estar el objeto según hacia dónde iba.')

    print('\nPASO 3 — Asociación con el algoritmo húngaro')
    print('  Matriz de distancias predicción x detección; el húngaro elige los pares que minimizan')
    print('  la distancia TOTAL. Un par a más de 28 px se rechaza: no es el mismo objeto.')
    print('  Si se asocia, el Kalman corrige su estimación con la medición (y ajusta la velocidad).')

    print('\nPASO 4 — Ciclo de vida de las pistas')
    print('  detección sin pista -> pista nueva | pista sin detección -> sobrevive 6 cuadros con la')
    print('  predicción, después se cierra | se muestran y cuentan sólo las confirmadas (5 detecciones).')

    print('\nPASO 5 — Aplicaciones sobre las trayectorias')
    print(f'  conteo: línea amarilla {LINEA[0]}-{LINEA[1]}; cruzar de izquierda a derecha = A->B (histéresis 4 px)')
    print(f'  permanencia: segundos de cada pista dentro de la zona naranja {ZONA}')
    print('  velocidad: la del Kalman, px/cuadro x fps' + (' x metros/px = m/s' if METROS_POR_PIXEL else ' = px/s'))
    print('  mapa de calor: cuántas veces pasó un centro por cada lugar')

    print('\n  Ventana: izquierda = seguimiento | derecha = máscara limpia de MOG2')
    print('  Cada id tiene su color y su estela; "#id?" = pista perdida, dibujada donde la predice el Kalman.')
    print('  Mirá: gente que camina junta (UNA caja), el poste (la parte en dos), cruces (¿cambian los id?).')
    print('  ESPACIO pausa, Q o ESC sale.')

    r = procesar(VIDEO, LINEA, ZONA, METROS_POR_PIXEL)
    an, fps = r['an'], r['fps']
    unidad = 'm/s' if METROS_POR_PIXEL else 'px/s'

    print('\nPASO 6 — Resultados')
    print(f'  -> {r["cuadros"]} cuadros ({r["cuadros"] / fps:.0f} s); video guardado en {r["ruta_video"]}')
    print(f'  -> conteo: A->B = {an.conteo["A→B"]}, B->A = {an.conteo["B→A"]}')
    for cuadro, id_, sentido in an.eventos[:10]:
        print(f'       cuadro {cuadro:4d} ({cuadro / fps:5.1f} s): #{id_} cruzó {sentido}')
    if len(an.eventos) > 10:
        print(f'       ... y {len(an.eventos) - 10} cruces más')
    print(f'  -> ids usados: {r["ids_usados"]}; pistas confirmadas: {len(r["pistas"])}')
    print(f'     (muchos más ids que personas = la identidad se rompe: gente que camina junta y se')
    print('      separa, el poste que parte a alguien en dos, cruces)')
    if an.dwell:
        top = sorted(an.dwell.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print('  -> permanencia en la zona (las 3 mayores): '
              + ', '.join(f'#{i} {s:.1f} s' for i, s in top))
    vm = velocidades_medianas(an)
    if vm:
        v = np.array(list(vm.values()))
        print(f'  -> velocidad mediana por objeto: media {v.mean():.1f} {unidad}, '
              f'rango {v.min():.1f} a {v.max():.1f} {unidad}')

    print('\nPASO 7 — Análisis: mapa de calor, trayectorias, ocupación, permanencia y velocidad')
    print('  El mapa de calor muestra por dónde pasan más centros (no cuánto tiempo se quedan).')
    print('  Cerrar las ventanas de los gráficos para terminar.')
    graficar_analisis(r, METROS_POR_PIXEL)
