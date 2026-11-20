"""Ejercicio 1 — De la disparidad a la profundidad métrica (con ground truth real).

Se calcula un mapa de disparidad con StereoSGBM, se lo pasa a profundidad en metros con
Z = f * B / (d + doffs) y se valida contra la disparidad de referencia (ground truth) medida
con luz estructurada: no alcanza con que "se vea bien".

Datos: escena Adirondack del Middlebury Stereo Dataset 2014. El .zip (~72 MB) se baja a mano
desde el navegador y se deja en datos/ (el script lo descomprime). Trae:
  im0.png, im1.png  par estéreo (izquierda, derecha)
  disp0.pfm         disparidad ground truth de la imagen izquierda (inf = sin dato)
  calib.txt         calibración: foco, baseline, doffs, ndisp...

Controles: cualquier tecla cierra la ventana.
"""
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

DATOS = Path(__file__).parent / 'datos'
ESCENA = 'Adirondack'   # otras: 'Motorcycle', 'Piano', 'Playtable'...
URL_MIDDLEBURY = 'https://vision.middlebury.edu/stereo/data/scenes2014/zip/{}-imperfect.zip'
ESCALA_VISTA = 0.2      # las imágenes son de ~2900x2000: se achican sólo para mostrarlas


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def buscar_escena(escena):
    # Usa la carpeta de la escena; si sólo está el .zip, lo descomprime.
    # El servidor de Middlebury no acepta descargas desde scripts: el .zip se baja a mano
    carpeta = DATOS / f'{escena}-imperfect'
    zip_path = DATOS / f'{escena}-imperfect.zip'
    if carpeta.exists():
        return carpeta
    if zip_path.exists():
        print(f'  Descomprimiendo {zip_path.name}...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATOS)
        return carpeta
    raise SystemExit(f'Falta la escena {escena}. Bajala desde el navegador:\n'
                     f'  {URL_MIDDLEBURY.format(escena)}\n'
                     f'y guardá el .zip (sin descomprimir) en:\n  {DATOS}')


def leer_pfm(ruta):
    # PFM = imagen de floats. Cabecera de texto: 'Pf' (1 canal), 'ancho alto', y una escala
    # cuyo signo indica el orden de bytes (negativa = little-endian). Después vienen los
    # floats crudos, con las filas de ABAJO hacia ARRIBA (por eso el flipud)
    with open(ruta, 'rb') as f:
        assert f.readline().strip() == b'Pf', 'sólo se soportan PFM de 1 canal'
        ancho, alto = map(int, f.readline().split())
        escala = float(f.readline())
        orden = '<' if escala < 0 else '>'
        datos = np.frombuffer(f.read(), dtype=orden + 'f4')
    return np.flipud(datos.reshape(alto, ancho)).astype(np.float32)


def leer_calibracion(ruta):
    # calib.txt tiene líneas 'clave=valor'. cam0 es la matriz de la cámara izquierda:
    #   [f 0 cx; 0 f cy; 0 0 1]  -> el foco f (en píxeles) es el primer número
    cal = {}
    for linea in Path(ruta).read_text().splitlines():
        clave, valor = linea.split('=')
        cal[clave.strip()] = valor.strip()
    return {
        'f': float(cal['cam0'].strip('[]').split()[0]),     # foco (px)
        'baseline': float(cal['baseline']),                 # distancia entre cámaras (mm)
        'doffs': float(cal['doffs']),                       # diferencia de centros ópticos (px)
        'ndisp': int(cal['ndisp']),                         # cota de la disparidad máxima
    }


def cargar_escena(escena):
    carpeta = buscar_escena(escena)
    return {
        'izq': cv2.imread(str(carpeta / 'im0.png')),
        'der': cv2.imread(str(carpeta / 'im1.png')),
        'gt': leer_pfm(carpeta / 'disp0.pfm'),
        'cal': leer_calibracion(carpeta / 'calib.txt'),
    }


# ---------------------------------------------------------------------------
# Correspondencia: StereoSGBM
# ---------------------------------------------------------------------------
def calcular_disparidad(izq, der, num_disparities=192, block_size=5):
    # Para cada píxel de la izquierda, busca el mismo bloque (block_size x block_size) en la
    # MISMA FILA de la derecha, corriéndose de 0 a num_disparities píxeles. Se queda con el
    # corrimiento más parecido: esa es la disparidad.
    # Sólo se busca en la misma fila porque el par está rectificado (las filas coinciden).
    # "SG" (semi-global): además del parecido de cada bloque, penaliza que la disparidad
    # salte entre píxeles vecinos: P1 para saltos de 1 px, P2 (mayor) para saltos grandes.
    # Así el mapa queda suave en superficies y sólo cambia de golpe en los bordes
    gris_izq = cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY)
    gris_der = cv2.cvtColor(der, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,     # tiene que ser múltiplo de 16
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,                    # chequeo izq->der vs der->izq: descarta oclusiones
        uniquenessRatio=10,                 # el mejor tiene que ganarle al 2do por un 10 %
        speckleWindowSize=100,              # borra manchitas sueltas de disparidad...
        speckleRange=32,                    # ...que difieren mucho de lo que las rodea
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    # OpenCV devuelve la disparidad x16 en enteros (4 bits de decimales): se divide por 16.
    # Los píxeles donde no encontró correspondencia quedan en -1
    return stereo.compute(gris_izq, gris_der).astype(np.float32) / 16.0


# ---------------------------------------------------------------------------
# Evaluación contra el ground truth
# ---------------------------------------------------------------------------
def evaluar(disp, gt, umbral=2.0):
    # Dos preguntas distintas:
    #   densidad:  ¿en qué % de los píxeles con GT nos animamos a dar una disparidad?
    #   precisión: donde dimos una, ¿cuánto le erramos? (sólo donde AMBOS tienen dato)
    con_gt = np.isfinite(gt)
    ambos = con_gt & (disp >= 0)
    err = np.abs(disp[ambos] - gt[ambos])
    return {
        'densidad': ambos.sum() / con_gt.sum() * 100,
        'rmse': float(np.sqrt(np.mean(err ** 2))),      # error típico, castiga los errores grandes
        'mediana': float(np.median(err)),               # error del píxel "del medio"
        'malos': (err > umbral).mean() * 100,           # % con error > umbral (bad-2)
        # Como en el notebook: los píxeles sin dato cuentan como malos
        'malos_con_faltantes': ((err > umbral).sum() + (con_gt & ~ambos).sum()) / con_gt.sum() * 100,
    }


# ---------------------------------------------------------------------------
# Triangulación: de disparidad a profundidad
# ---------------------------------------------------------------------------
def disparidad_a_profundidad(disp, cal):
    # Z = f * B / (d + doffs)
    #   f [px] * B [mm] / d [px] = mm  -> /1000 para metros
    # doffs: los centros ópticos de las dos cámaras no están en la misma columna (cx de cam1
    # menos cx de cam0), así que a la disparidad medida hay que sumarle ese corrimiento.
    # Donde no hay disparidad (inf en el GT, -1 en SGBM) la profundidad queda en NaN
    Z = np.full(disp.shape, np.nan, np.float32)
    ok = np.isfinite(disp) & (disp >= 0)
    Z[ok] = cal['f'] * cal['baseline'] / (disp[ok] + cal['doffs']) / 1000
    return Z


def color_profundidad(Z, z_max=3.0):
    # Profundidad en colores (VIRIDIS): violeta = cerca, amarillo = z_max metros o más lejos
    ok = np.isfinite(Z)
    norm = np.zeros(Z.shape, np.uint8)
    norm[ok] = (np.clip(Z[ok] / z_max, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    color[~ok] = 0
    return color


def error_por_pixel_de_disparidad(Z, cal):
    # ¿Cuánto cambia Z si la disparidad se equivoca en 1 px? Derivando Z = fB/(d+doffs):
    #   dZ/dd = -fB/(d+doffs)^2 = -Z^2 / (fB)
    # El error crece con el CUADRADO de la distancia: lejos, 1 px de error son muchos cm
    return Z ** 2 / (cal['f'] * cal['baseline'] / 1000)


def mapa_error(disp, gt, err_max=10):
    # |nuestra - GT| en colores: azul = acierto, rojo = error de err_max px o más.
    # Negro = no se puede comparar (falta nuestro dato o el del GT)
    ambos = np.isfinite(gt) & (disp >= 0)
    err = np.zeros(gt.shape, np.float32)
    err[ambos] = np.abs(disp[ambos] - gt[ambos])
    norm = (np.clip(err / err_max, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~ambos] = 0
    return color


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def color_disparidad(disp, d_max):
    # Disparidad en colores (JET): rojo = disparidad alta = CERCA, azul = lejos.
    # Los píxeles sin dato (inf o negativos) quedan en negro
    valido = np.isfinite(disp) & (disp > 0)
    norm = np.zeros(disp.shape, np.uint8)
    norm[valido] = np.clip(disp[valido] / d_max * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[~valido] = 0
    return color


def con_titulo(img, texto):
    img = cv2.resize(img, None, fx=ESCALA_VISTA, fy=ESCALA_VISTA, interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == '__main__':
    print(f'PASO 1 — Escena {ESCENA}: par estéreo, ground truth y calibración')
    esc = cargar_escena(ESCENA)
    cal, gt = esc['cal'], esc['gt']
    alto, ancho = esc['izq'].shape[:2]
    print(f'  Imágenes de {ancho}x{alto} px')
    print(f'  Calibración: foco {cal["f"]:.1f} px | baseline {cal["baseline"]:.1f} mm | '
          f'doffs {cal["doffs"]:.2f} px | ndisp {cal["ndisp"]}')
    valido = np.isfinite(gt)
    print(f'  Ground truth: disparidad de {gt[valido].min():.0f} a {gt[valido].max():.0f} px; '
          f'{(~valido).mean() * 100:.1f} % de píxeles sin dato (oclusiones)')

    print('\nPASO 2 — Nuestra disparidad con StereoSGBM')
    # numDisparities tiene que ser múltiplo de 16 y cubrir el rango que indica la escena (ndisp)
    NUM_DISP = int(np.ceil(cal['ndisp'] / 16)) * 16
    BLOCK = 5
    print(f'  numDisparities = {NUM_DISP} (ndisp {cal["ndisp"]} redondeado a múltiplo de 16) | '
          f'blockSize = {BLOCK}')
    print('  Calculando (con imágenes de este tamaño tarda unos segundos)...')
    t0 = time.perf_counter()
    disp = calcular_disparidad(esc['izq'], esc['der'], NUM_DISP, BLOCK)
    print(f'  -> {time.perf_counter() - t0:.1f} s')
    print(f'  -> sin correspondencia (disparidad < 0): {(disp < 0).mean() * 100:.1f} % de los píxeles')

    print('\nPASO 3 — Comparación con el ground truth')
    ev = evaluar(disp, gt)
    print(f'  Densidad: damos disparidad en el {ev["densidad"]:.1f} % de los píxeles que tienen GT')
    print('  Precisión (sólo donde los dos tienen dato):')
    print(f'    RMSE           {ev["rmse"]:6.2f} px')
    print(f'    error mediano  {ev["mediana"]:6.2f} px')
    print(f'    malos (> 2 px) {ev["malos"]:6.1f} %')
    print(f'  Si los píxeles sin dato se cuentan como malos (como el notebook): '
          f'{ev["malos_con_faltantes"]:.1f} %')
    print('  Mediana chica y RMSE grande: la mitad de los píxeles acierta casi exacto, pero los que')
    print('  fallan le erran por MUCHO (decenas de px, emparejaron con el bloque equivocado) y el')
    print('  cuadrado del RMSE los hace pesar más. El mapa de error muestra dónde están.')

    # Arriba: lo que entra y la respuesta correcta | Abajo: nuestra respuesta y dónde le erramos
    cv2.imshow('Disparidad', np.vstack([
        np.hstack([con_titulo(esc['izq'], 'imagen izquierda'),
                   con_titulo(color_disparidad(gt, cal['ndisp']), 'Ground truth (rojo = cerca)')]),
        np.hstack([con_titulo(color_disparidad(disp, cal['ndisp']), 'SGBM (negro = sin dato)'),
                   con_titulo(mapa_error(disp, gt), 'error |SGBM - GT| (azul = 0, rojo >= 10 px)')]),
    ]))
    print('  (cualquier tecla cierra la ventana y sigue con el paso 4)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 4 — De disparidad a profundidad en metros: Z = f * B / (d + doffs)')
    Z_gt = disparidad_a_profundidad(gt, cal)
    Z_mia = disparidad_a_profundidad(disp, cal)
    print(f'  La escena está entre {np.nanmin(Z_gt):.2f} y {np.nanmax(Z_gt):.2f} m de la cámara')

    # Se compara sólo donde los dos tienen profundidad (y a menos de 5 m, como el notebook)
    comparar = np.isfinite(Z_gt) & np.isfinite(Z_mia) & (Z_gt < 5.0)
    err_cm = np.abs(Z_mia - Z_gt)[comparar] * 100
    print(f'  Error de profundidad: medio {err_cm.mean():.1f} cm | mediano {np.median(err_cm):.1f} cm')
    print(f'  -> la mitad de los píxeles le erra por menos de {np.median(err_cm):.1f} cm; '
          f'el {(err_cm > 10).mean() * 100:.1f} % por más de 10 cm')

    print('  Sensibilidad: cuánto cambia Z si la disparidad le erra por 1 px')
    for z in (1.0, 2.0, 5.0, 10.0):
        print(f'    a {z:4.1f} m -> {error_por_pixel_de_disparidad(z, cal) * 100:6.2f} cm')
    print('  Crece con Z al cuadrado: con estéreo, lo lejano siempre se mide peor. Más baseline')
    print('  o más foco (f * B más grande) achican ese error.')

    err_vis = np.full(Z_gt.shape, np.nan, np.float32)
    err_vis[comparar] = np.abs(Z_mia - Z_gt)[comparar]
    cv2.imshow('Profundidad', np.vstack([
        np.hstack([con_titulo(esc['izq'], 'imagen izquierda'),
                   con_titulo(color_profundidad(Z_gt), 'Ground truth: profundidad (violeta = cerca)')]),
        np.hstack([con_titulo(color_profundidad(Z_mia), 'SGBM: profundidad (amarillo >= 3 m)'),
                   con_titulo(color_profundidad(err_vis, z_max=0.10), 'error de profundidad (amarillo >= 10 cm)')]),
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
