"""Ejercicio 3 — Alarma de objeto abandonado con dos modelos de fondo (MOG2 lento y rápido).

Un objeto que queda quieto termina absorbido por el modelo de fondo. Con dos MOG2 que aprenden
a distinta velocidad, un objeto quieto es primer plano para el LENTO pero ya es fondo para el
RÁPIDO:   estático = primer_plano(lento) Y NO primer_plano(rápido)
Si una región estática se mantiene más de T segundos, se da la alarma.

El video de muestra no tiene ningún objeto abandonado: a partir del cuadro DESDE se pega un
"bolso" sintético. Como sabemos dónde y cuándo aparece, se puede medir si la alarma acierta.
Con un video propio se pone SIMULAR = False.

Controles: ESPACIO pausa/reanuda, Q o ESC sale.
"""
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
VIDEO = DATOS / 'vtest.avi'
VENTANA = 'Objeto abandonado'
ESCALA = 0.6            # los cuadros se reducen al 60 % (como en el notebook)

K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def reducir(frame):
    return cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)


def gris(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f


# ---------------------------------------------------------------------------
# Simulación del objeto abandonado
# ---------------------------------------------------------------------------
def crear_bolso(ancho=26, alto=18, seed=0):
    # Textura aleatoria oscura y suavizada (para que no sea un rectángulo liso) con un borde
    # rojizo. Semilla fija: siempre el mismo bolso
    rng = np.random.default_rng(seed)
    bolso = rng.integers(10, 110, (alto, ancho, 3), dtype=np.uint8)
    bolso = cv2.GaussianBlur(bolso, (3, 3), 0)
    cv2.rectangle(bolso, (0, 0), (ancho - 1, alto - 1), (30, 30, 200), 1)
    return bolso


def pegar_bolso(frame, bolso, pos):
    # Reemplaza una región del cuadro por el bolso (sin modificar el cuadro original)
    x, y = pos
    alto, ancho = bolso.shape[:2]
    salida = frame.copy()
    salida[y:y + alto, x:x + ancho] = bolso
    return salida


# ---------------------------------------------------------------------------
# Dos modelos de fondo: lento y rápido
# ---------------------------------------------------------------------------
def fondo_inicial(ruta, n_init=100, paso=4):
    # Fondo de arranque: mediana, píxel a píxel, de un cuadro cada `paso` de los primeros n_init.
    # Las personas pasan y en cada píxel están poco tiempo, así que la mediana se queda con el
    # fondo. Con n_init = 1 se usaría sólo el primer cuadro, con la gente incluida, y donde
    # estaban esas personas aparecerían "fantasmas" cuando se van
    cap = cv2.VideoCapture(str(ruta))
    cuadros = []
    for k in range(n_init):
        ok, f = cap.read()
        if not ok:
            break
        if k % paso == 0:
            cuadros.append(reducir(f))
    cap.release()
    fondo = np.median(np.stack(cuadros), axis=0).astype(np.uint8)
    return cv2.GaussianBlur(fondo, (5, 5), 0)


def crear_modelos(fondo0):
    # Mismo algoritmo (MOG2), distinta memoria:
    #   lento  -> history 1000: tarda mucho en aceptar algo nuevo como parte del fondo
    #   rápido -> history 100:  en pocos segundos incorpora lo que se queda quieto
    # Sin sombras (detectShadows=False): la máscara es sólo 0 (fondo) o 255 (primer plano)
    lento = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=False)
    rapido = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
    # Los dos arrancan sabiendo cómo es el fondo: se les muestra el fondo inicial varias veces
    # con una tasa de aprendizaje alta. Si no, el lento tardaría minutos en aprenderlo
    for m in (lento, rapido):
        for _ in range(30):
            m.apply(fondo0, learningRate=0.1)
    return lento, rapido


def aplicar_modelos(lento, rapido, frame, lr_lento=0.0008, lr_rapido=0.02):
    # Los dos ven EXACTAMENTE el mismo cuadro; sólo cambia cuánto aprenden de él.
    # learningRate = cuánto sube, por cuadro, el peso del color que se está viendo.
    # MOG2 pasa un color al fondo cuando su peso llega a 1 - backgroundRatio = 0.1, así que
    # un objeto quieto se vuelve fondo en aproximadamente 0.1 / learningRate cuadros:
    #   0.0008 -> ~125 cuadros (~12.5 s a 10 fps)
    #   0.02   -> ~5 cuadros   (~0.5 s)
    f = cv2.GaussianBlur(frame, (5, 5), 0)
    return lento.apply(f, learningRate=lr_lento), rapido.apply(f, learningRate=lr_rapido)


def mascara_estatica(fg_lento, fg_rapido):
    # Lo nuevo (primer plano del lento) que NO se mueve (no es primer plano del rápido).
    # Una persona caminando es primer plano en los dos -> queda afuera.
    est = cv2.bitwise_and(fg_lento, cv2.bitwise_not(fg_rapido))
    # Apertura: borra puntitos sueltos (bordes de personas, ruido) antes de acumular
    return cv2.morphologyEx(est, cv2.MORPH_OPEN, K_OPEN)


# ---------------------------------------------------------------------------
# Persistencia y alarma
# ---------------------------------------------------------------------------
def actualizar_persistencia(persist, estatico):
    # Cuántos cuadros SEGUIDOS lleva cada píxel siendo estático.
    # Si deja de serlo (alguien pasa por encima, o el lento lo absorbió), vuelve a 0
    return np.where(estatico > 0, persist + 1, 0).astype(np.uint16)


def mascara_alarma(persist, T):
    # Píxeles que llevan T cuadros o más siendo estáticos; el cierre une la región del objeto
    alarma = ((persist >= T) * 255).astype(np.uint8)
    return cv2.morphologyEx(alarma, cv2.MORPH_CLOSE, K_CLOSE)


def detectar_blobs(mascara, area_min=80):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contornos if cv2.contourArea(c) >= area_min]


# ---------------------------------------------------------------------------
# Filtros contra falsas alarmas
# ---------------------------------------------------------------------------
def recortar_a_zona(mascara, zona):
    # Sólo interesa lo que pasa dentro de la zona vigilada (andén, hall, acceso...)
    zx, zy, zw, zh = zona
    salida = np.zeros_like(mascara)
    salida[zy:zy + zh, zx:zx + zw] = mascara[zy:zy + zh, zx:zx + zw]
    return salida


def forma_de_objeto_dejado(caja, area_min=200, area_max=900):
    # Un bolso o paquete es compacto y de tamaño acotado: se descartan regiones muy grandes
    # o muy alargadas (cambios de iluminación en paredes, sombras largas, etc.)
    x, y, w, h = caja
    return area_min <= w * h <= area_max and 0.3 <= w / h <= 3.5


def borde_medio(g, caja, grosor=2):
    # Intensidad media del gradiente (Sobel) sobre el contorno de la caja.
    # Un objeto real tiene bordes marcados contra lo que lo rodea
    x, y, w, h = caja
    m = np.zeros(g.shape, np.uint8)
    cv2.rectangle(m, (x, y), (x + w - 1, y + h - 1), 255, grosor)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(cv2.magnitude(gx, gy)[m > 0].mean())


def esta_presente(frame, fondo_lento, caja):
    # ¿El objeto está AHORA (dejado) o estaba ANTES y se fue (retirado = fantasma)?
    # En los dos casos el lento ve una diferencia, pero los bordes del objeto aparecen
    # en la imagen donde el objeto realmente está:
    #   dejado   -> bordes fuertes en el cuadro actual, débiles en el fondo del lento
    #   retirado -> bordes fuertes en el fondo del lento (el objeto viejo), débiles ahora
    return borde_medio(gris(frame), caja) > borde_medio(gris(fondo_lento), caja)


class Detector:
    # Guarda el estado entre cuadros (los dos modelos y la persistencia de cada píxel)
    def __init__(self, fondo0, fps, lr_lento=0.0008, lr_rapido=0.02, t_alarma_s=5, zona=None):
        self.lento, self.rapido = crear_modelos(fondo0)
        self.lr_lento, self.lr_rapido = lr_lento, lr_rapido
        self.T = int(t_alarma_s * fps)                  # tiempo de alarma en cuadros
        self.zona = zona
        self.persist = np.zeros(fondo0.shape[:2], np.uint16)

    def procesar(self, frame):
        r = {}
        # 1. los dos modelos ven el cuadro
        r['fg_lento'], r['fg_rapido'] = aplicar_modelos(self.lento, self.rapido, frame,
                                                         self.lr_lento, self.lr_rapido)
        # 2. lo nuevo que no se mueve
        r['estatico'] = mascara_estatica(r['fg_lento'], r['fg_rapido'])
        # 3. cuánto tiempo lleva quieto cada píxel; los que superan T son alarma
        self.persist = actualizar_persistencia(self.persist, r['estatico'])
        alarma = mascara_alarma(self.persist, self.T)
        if self.zona is not None:
            alarma = recortar_a_zona(alarma, self.zona)
        cajas = detectar_blobs(alarma)
        # 4. filtro de forma: sólo regiones del tamaño y proporción de un bolso
        r['por_forma'] = [c for c in cajas if not forma_de_objeto_dejado(c)]
        cajas = [c for c in cajas if forma_de_objeto_dejado(c)]
        # 5. test de bordes: descarta los fantasmas de objetos retirados
        fondo = self.lento.getBackgroundImage()
        r['fantasmas'] = [c for c in cajas if not esta_presente(frame, fondo, c)]
        r['alarmas'] = [c for c in cajas if esta_presente(frame, fondo, c)]
        r['persist'] = self.persist
        r['fondo_lento'] = fondo
        return r


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def con_titulo(img, texto):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def dibujar_cajas(img, cajas, color, texto=None):
    for x, y, w, h in cajas:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if texto:
            cv2.putText(img, texto, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return img


def mapa_persistencia(persist, T):
    # Persistencia en colores: azul = recién se quedó quieto, rojo = llegó a T (alarma)
    p = np.clip(persist.astype(np.float32) / max(T, 1), 0, 1)
    color = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color[persist == 0] = 0
    return color


def armar_vista(frame, r, det, i, estado, fps):
    # Arriba: cuadro con alarmas | primer plano LENTO | primer plano RÁPIDO
    # Abajo:  máscara ESTÁTICA   | PERSISTENCIA      | fondo que recuerda el lento
    vis = frame.copy()
    if det.zona is not None:
        zx, zy, zw, zh = det.zona
        cv2.rectangle(vis, (zx, zy), (zx + zw, zy + zh), (255, 200, 0), 1)
    dibujar_cajas(vis, r['por_forma'], (0, 165, 255), 'forma')      # descartadas por forma
    dibujar_cajas(vis, r['fantasmas'], (200, 200, 200), 'fantasma')  # descartadas por bordes
    dibujar_cajas(vis, r['alarmas'], (0, 0, 255), 'ABANDONADO')
    if r['alarmas']:
        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 0, 255), 6)
    arriba = np.hstack([con_titulo(vis, f'cuadro {i}  {estado}'),
                        con_titulo(r['fg_lento'], 'primer plano LENTO'),
                        con_titulo(r['fg_rapido'], 'primer plano RAPIDO')])
    abajo = np.hstack([con_titulo(r['estatico'], 'ESTATICO = lento Y NO rapido'),
                       con_titulo(mapa_persistencia(r['persist'], det.T), f'PERSISTENCIA (rojo = {det.T / fps:.0f} s)'),
                       con_titulo(r['fondo_lento'], 'fondo que recuerda el LENTO')])
    return np.vstack([arriba, abajo])


# ---------------------------------------------------------------------------
# Recorrido del video
# ---------------------------------------------------------------------------
def procesar(ruta, bolso, pos_bolso, desde, simular=True, n_init=100, **params):
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir el video: {ruta}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    det = Detector(fondo_inicial(ruta, n_init), fps, **params)
    eventos = []                # alarmas de cada cuadro
    n_forma = n_fantasma = 0    # cuadros con regiones descartadas por cada filtro
    i, pausa = 0, False

    while True:
        if not pausa:
            ok, frame = cap.read()
            if not ok:
                break
            frame = reducir(frame)

            # a partir del cuadro `desde` el bolso está en la escena
            hay_bolso = simular and i >= desde
            if hay_bolso:
                frame = pegar_bolso(frame, bolso, pos_bolso)

            r = det.procesar(frame)
            eventos.append(r['alarmas'])
            n_forma += bool(r['por_forma'])
            n_fantasma += bool(r['fantasmas'])

            estado = f'bolso desde hace {(i - desde) / fps:.1f} s' if hay_bolso else 'sin bolso'
            cv2.imshow(VENTANA, armar_vista(frame, r, det, i, estado, fps))
            i += 1

        tecla = cv2.waitKey(int(1000 / fps)) & 0xFF
        if tecla in (ord('q'), 27):
            break
        if tecla == ord(' '):
            pausa = not pausa

    cap.release()
    cv2.destroyAllWindows()
    return {'cuadros': i, 'fps': fps, 'eventos': eventos,
            'cuadros_forma': n_forma, 'cuadros_fantasma': n_fantasma}


# ---------------------------------------------------------------------------
# Evaluación (sólo con el bolso simulado: sabemos dónde está)
# ---------------------------------------------------------------------------
def sobre_bolso(caja, pos, tam, margen=12):
    # ¿El centro de la caja cae sobre el bolso simulado?
    x, y, w, h = caja
    cx, cy = x + w / 2, y + h / 2
    return (pos[0] - margen <= cx <= pos[0] + tam[0] + margen
            and pos[1] - margen <= cy <= pos[1] + tam[1] + margen)


def evaluar(r, pos, tam, desde):
    # Separa las alarmas en verdaderas (sobre el bolso) y falsas (en cualquier otro lado)
    fps = r['fps']
    sobre = [i for i, cs in enumerate(r['eventos']) if any(sobre_bolso(c, pos, tam) for c in cs)]
    falsas = [i for i, cs in enumerate(r['eventos']) if any(not sobre_bolso(c, pos, tam) for c in cs)]
    return {
        'demora_s': (sobre[0] - desde) / fps if sobre else None,
        'duracion_s': len(sobre) / fps,
        'falsas': falsas,
    }


if __name__ == '__main__':
    SIMULAR = True                       # False para usar un video propio con un objeto real
    POS_BOLSO = (193, 173)               # (x, y) en el cuadro reducido (461x346); misma posición que el notebook
    DESDE = 120                          # cuadro en que aparece el bolso
    T_ALARMA_S = 5                       # segundos quieto para dar la alarma
    LR_LENTO, LR_RAPIDO = 0.0008, 0.02
    ZONA_VIGILADA = (55, 86, 285, 207)   # x, y, ancho, alto
    N_INIT = 100                         # cuadros para el fondo inicial (1 = sólo el primero)

    print('PASO 1 — Simulación del objeto abandonado')
    print(f'  Se pega un "bolso" sintético de 26x18 px en {POS_BOLSO}, a partir del cuadro {DESDE}.')
    print('  Como sabemos dónde y cuándo aparece, se puede medir si la alarma acierta y cuánto tarda.')

    print('\nPASO 2 — Fondo inicial y dos modelos MOG2')
    print(f'  Fondo inicial: mediana de los primeros {N_INIT} cuadros (la gente pasa, el fondo queda).')
    print('  Un objeto quieto se vuelve fondo en ~0.1 / learningRate cuadros:')
    print(f'    LENTO  (lr {LR_LENTO}): ~{0.1 / LR_LENTO / 10:.1f} s -> lo nuevo, se mueva o no')
    print(f'    RÁPIDO (lr {LR_RAPIDO}): ~{0.1 / LR_RAPIDO / 10:.1f} s -> sólo lo que se mueve')

    print('\nPASO 3 — Máscara estática = lento Y NO rápido')
    print('  persona caminando: primer plano en los dos -> 0 | objeto quieto: sólo en el lento -> 1')

    print('\nPASO 4 — Persistencia')
    print(f'  Cada píxel cuenta cuántos cuadros seguidos es estático; a los {T_ALARMA_S} s ({T_ALARMA_S * 10} cuadros)')
    print('  es alarma. Si alguien pasa por encima, ese contador vuelve a 0.')

    print('\nPASO 5 — Filtros')
    print(f'  zona vigilada {ZONA_VIGILADA} | forma de bolso: área 200-900 px y proporción 0.3-3.5')
    print('  test de bordes: un objeto presente tiene bordes en el cuadro ACTUAL; un fantasma (objeto')
    print('  retirado) los tiene en el FONDO que recuerda el lento.')

    print('\n  Ventana: arriba = cuadro | primer plano LENTO | primer plano RÁPIDO')
    print('           abajo  = ESTÁTICO | PERSISTENCIA (azul -> rojo) | fondo que recuerda el LENTO')
    print('  rojo = ABANDONADO | naranja = descartado por forma | gris = fantasma | celeste = zona vigilada')
    print('  Mirá el bolso: blanco en el LENTO, negro en el RÁPIDO, y en la persistencia pasa de azul a rojo.')
    print('  ESPACIO pausa, Q o ESC sale.')

    bolso = crear_bolso()
    tam = (bolso.shape[1], bolso.shape[0])
    r = procesar(VIDEO, bolso, POS_BOLSO, DESDE, SIMULAR, N_INIT, lr_lento=LR_LENTO,
                 lr_rapido=LR_RAPIDO, t_alarma_s=T_ALARMA_S, zona=ZONA_VIGILADA)
    fps = r['fps']

    print('\nPASO 6 — Resultados')
    print(f'  -> {r["cuadros"]} cuadros a {fps:.0f} fps')
    if SIMULAR and r['cuadros'] > DESDE:
        ev = evaluar(r, POS_BOLSO, tam, DESDE)
        if ev['demora_s'] is not None:
            print(f'  -> ALARMA sobre el bolso {ev["demora_s"]:.1f} s después de que aparece '
                  f'(se pidieron {T_ALARMA_S} s); sonó durante {ev["duracion_s"]:.1f} s')
            print('     Se apaga cuando el LENTO también absorbe el bolso: la alarma tiene que sonar')
            print(f'     antes de ~{0.1 / LR_LENTO / 10:.0f} s, si no, nunca suena.')
        else:
            print('  -> la alarma nunca sonó sobre el bolso')
        falsas = ev['falsas']
        print(f'  -> cuadros con alarmas en OTRO lugar: {len(falsas)}'
              + (f' (entre el cuadro {falsas[0]} y el {falsas[-1]})' if falsas else ''))
        if falsas:
            print('     En este video salen del cartel junto al poste: dos personas estuvieron paradas ahí')
            print('     casi todo el arranque, así que la mediana las metió en el fondo inicial. Cuando se')
            print('     van, el LENTO ve "algo nuevo" (el cartel que tapaban) y el RÁPIDO ya lo aprendió:')
            print('     es un fantasma de objeto retirado. El test de bordes no lo descarta porque lo que')
            print('     quedó a la vista, el cartel, tiene bordes fuertes (ver "fondo que recuerda el LENTO").')
    print(f'  -> cuadros con regiones descartadas por forma: {r["cuadros_forma"]}, '
          f'como fantasma: {r["cuadros_fantasma"]}')
