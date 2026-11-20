"""Ejercicio 3 — Registración multimodal: CT <-> MRI con Información Mutua.

Se alinean un CT y una resonancia (MRI-T1) de la cabeza del MISMO paciente. Entre modalidades
distintas la intensidad no se relaciona linealmente (el hueso es brillante en CT y oscuro en
MRI), así que SSD/NCC no sirven: se usa Información Mutua, con SimpleITK.

Datos: el par CT/MRI del tutorial oficial de registración de SimpleITK. Se descarga solo la
primera vez en datos/.

Controles: cualquier tecla cierra la ventana.
"""
import hashlib
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

DATOS = Path(__file__).parent / 'datos'

# Los archivos de datos de SimpleITK se publican por su hash SHA-512 (sale del manifest.json
# del repositorio SimpleITK-Notebooks). El hash también sirve para verificar que bajó entero
URL_SITK = 'https://s3.amazonaws.com/simpleitk/public/notebooks/SHA512/{}'
ARCHIVOS = {
    'training_001_ct.mha': '1b950bc42fddfcefc76b9203d5dd6c45960c4fa8dcb69b839d3d083270d3d4c9'
                           'a9d378de3d3f914e432dc18fb44c9b9770d4db5580a70265f3e24e6cdb83015d',
    'training_001_mr_T1.mha': '3d15477962fef5851207964c381ffe77c586a6f70f2a373ecd3b6b4dc50d51dc'
                              '6cd893eb1bedabcd382a96f0dafac893ae9e5a7c2b7333f9ff3f0c6b7016c7bc',
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def descargar(nombre):
    # Baja el archivo si todavía no está en datos/ y verifica su hash
    ruta = DATOS / nombre
    if ruta.exists():
        return ruta
    DATOS.mkdir(exist_ok=True)
    sha = ARCHIVOS[nombre]
    print(f'  Descargando {nombre}...')
    urllib.request.urlretrieve(URL_SITK.format(sha), ruta)
    if hashlib.sha512(ruta.read_bytes()).hexdigest() != sha:
        ruta.unlink()
        raise IOError(f'{nombre} bajó corrupto, volver a correr')
    return ruta


def leer(nombre):
    # float32: la registración trabaja con intensidades continuas
    return sitk.ReadImage(str(descargar(nombre)), sitk.sitkFloat32)


def describir(nombre, img):
    # Tamaño en vóxeles y espaciado en mm. tamaño x espaciado = cuánto mide en el mundo real
    tam, esp = img.GetSize(), img.GetSpacing()
    mm = [t * e for t, e in zip(tam, esp)]
    print(f'  {nombre}: {tam[0]}x{tam[1]}x{tam[2]} vóxeles | '
          f'vóxel de {esp[0]:.2f}x{esp[1]:.2f}x{esp[2]:.2f} mm | '
          f'cubre {mm[0]:.0f}x{mm[1]:.0f}x{mm[2]:.0f} mm')


# ---------------------------------------------------------------------------
# Remuestreo: llevar una imagen a la grilla de otra
# ---------------------------------------------------------------------------
def remuestrear(movil, fija, transformacion=None):
    # Arma una imagen con la MISMA grilla que `fija` (mismo tamaño, espaciado y origen) y
    # para cada vóxel busca qué valor tiene `movil` en ese punto del espacio (en mm),
    # interpolando entre sus vóxeles. Así las dos quedan comparables vóxel a vóxel.
    # La transformación dice dónde buscar: sin ninguna (identidad) se asume que el mismo
    # punto en mm es el mismo punto de la cabeza. Lo que cae fuera de `movil` queda en 0
    if transformacion is None:
        transformacion = sitk.Transform()      # identidad: no mueve nada
    return sitk.Resample(movil, fija, transformacion, sitk.sitkLinear, 0.0, movil.GetPixelID())


# ---------------------------------------------------------------------------
# Registración
# ---------------------------------------------------------------------------
def registrar(fija, movil, seed=42):
    # Busca la transformación rígida (3 rotaciones + 3 traslaciones) que mejor alinea `movil`
    # con `fija`. Es una optimización: se prueba una transformación, se mide qué tan bien
    # alinea (métrica) y se la corrige en la dirección que mejora la métrica, hasta converger
    #
    # Punto de partida: hacer coincidir los centros geométricos de los dos volúmenes
    inicial = sitk.CenteredTransformInitializer(
        fija, movil, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    # Métrica: Información Mutua (Mattes). No compara intensidades directamente: mide cuánto
    # "predice" la intensidad de una imagen a la de la otra (con un histograma conjunto de 50
    # bins). Hueso blanco en CT <-> oscuro en MRI no es problema mientras sea CONSISTENTE.
    # SimpleITK la devuelve negativa: más negativa = mejor alineación
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Se evalúa en un 20 % de vóxeles al azar (mucho más rápido, casi igual de bueno).
    # Semilla fija: la misma corrida da siempre el mismo resultado
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed)
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizador: descenso por gradiente. Escalas por desplazamiento físico: un cambio de
    # 1 radián de rotación mueve los puntos muchos más mm que 1 mm de traslación; las escalas
    # equilibran los 6 parámetros para que avancen parejo
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolución: primero con las imágenes reducidas x4 y muy suavizadas (se ve sólo la
    # forma general de la cabeza: es fácil acercarse sin trabarse en detalles), después x2 y
    # al final a resolución completa para el ajuste fino
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(inicial, inPlace=False)

    # Se anota la métrica en cada iteración para ver cómo avanza en cada nivel
    historia = []
    reg.AddCommand(sitk.sitkIterationEvent,
                   lambda: historia.append((reg.GetCurrentLevel(), reg.GetMetricValue())))
    final = reg.Execute(fija, movil)
    return final, {
        'inicial': inicial,
        'metrica': reg.GetMetricValue(),
        'parada': reg.GetOptimizerStopConditionDescription(),
        'historia': historia,
    }


def evaluar_mi(fija, movil, transformacion):
    # Información Mutua con la transformación dada, SIN optimizar nada: sólo se mide.
    # Mismos bins que en la registración, pero sobre TODOS los vóxeles y a resolución
    # completa, para que los valores de distintas transformaciones sean comparables
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(transformacion)
    return reg.MetricEvaluate(fija, movil)


def parametros_rigidos(transformacion):
    # Execute devuelve una transformación compuesta con una sola Euler3D adentro.
    # Sus parámetros: (ángulo x, ángulo y, ángulo z) en radianes y (tx, ty, tz) en mm
    euler = sitk.CompositeTransform(transformacion).GetNthTransform(0)
    p = euler.GetParameters()
    return np.degrees(p[:3]), np.array(p[3:])


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def checkerboard(a, b, lado=32):
    # Tablero de ajedrez: cuadrados alternados de `a` y `b` (los dos cortes del mismo tamaño).
    # Si están alineados, los contornos (cráneo, ventrículos) siguen continuos de un cuadrado
    # al vecino; si no, aparecen "escalones"
    filas, cols = np.indices(a.shape)
    de_b = (filas // lado + cols // lado) % 2 == 1
    salida = a.copy()
    salida[de_b] = b[de_b]
    return salida


def slice_central(img):
    # SimpleITK indexa (x, y, z) pero el array de numpy queda (z, y, x):
    # arr[k] es el corte axial k (un plano horizontal de la cabeza)
    arr = sitk.GetArrayFromImage(img)
    return arr[arr.shape[0] // 2]


def normalizar_uint8(corte):
    # CT va de ~-1000 (aire) a ~3000 (hueso); MRI tiene otra escala. Se lleva todo a 0-255
    return cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def con_titulo(img, texto, alto=400):
    # Lleva el corte a `alto` píxeles (para poder ponerlos lado a lado) y le agrega un título
    img = cv2.resize(img, (int(img.shape[1] * alto / img.shape[0]), alto))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(vis, texto, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


if __name__ == '__main__':
    print('PASO 1 — Lectura del CT y la MRI')
    ct = leer('training_001_ct.mha')         # fija: la referencia
    mr = leer('training_001_mr_T1.mha')      # móvil: la que se va a alinear
    describir('CT ', ct)
    describir('MRI', mr)
    print('  Distinta cantidad de vóxeles y distinto tamaño de vóxel: para compararlas hay que')
    print('  trabajar en milímetros (espacio físico), no en píxeles.')

    corte_ct = normalizar_uint8(slice_central(ct))
    corte_mr = normalizar_uint8(slice_central(mr))
    cv2.imshow('CT | MRI', np.hstack([
        con_titulo(corte_ct, f'CT - corte central ({corte_ct.shape[1]}x{corte_ct.shape[0]} px)'),
        con_titulo(corte_mr, f'MRI - corte central ({corte_mr.shape[1]}x{corte_mr.shape[0]} px)'),
    ]))
    print('  (cualquier tecla cierra la ventana y sigue con el paso 2)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 2 — Antes de registrar: la MRI en la grilla del CT')
    print('  Se remuestrea la MRI sobre la grilla del CT (512x512x29, vóxel de 0.65 mm) SIN mover')
    print('  nada (transformación identidad). Ahora las dos tienen el mismo tamaño y se pueden')
    print('  comparar corte a corte. Lo que no cubría la MRI queda negro.')
    mr_antes = remuestrear(mr, ct)
    print(f'  MRI remuestreada: {"x".join(map(str, mr_antes.GetSize()))} vóxeles')

    corte_mr_antes = normalizar_uint8(slice_central(mr_antes))
    print('  Tablero: cuadrados alternados de CT y MRI. Mirá el contorno del cráneo: al pasar')
    print('  de un cuadrado de CT a uno de MRI se "corta" -> están desalineadas.')
    cv2.imshow('Antes de registrar', np.hstack([
        con_titulo(corte_ct, 'CT (fija)'),
        con_titulo(corte_mr_antes, 'MRI en la grilla del CT, sin mover'),
        con_titulo(checkerboard(corte_ct, corte_mr_antes), 'tablero CT / MRI - ANTES'),
    ]))
    print('  (cualquier tecla cierra la ventana y sigue con el paso 3)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nPASO 3 — Registración rígida con Información Mutua')
    print('  Se buscan 3 rotaciones y 3 traslaciones (la cabeza es rígida: no se deforma)')
    print('  Optimizando (tarda unos segundos)...')
    t0 = time.perf_counter()
    final, info = registrar(ct, mr)
    print(f'  -> {time.perf_counter() - t0:.1f} s')

    print('  Iteraciones por nivel de resolución:')
    for nivel, factor in enumerate((4, 2, 1)):
        valores = [m for lv, m in info['historia'] if lv == nivel]
        if valores:
            print(f'    nivel {nivel} (imagen /{factor}): {len(valores):3d} iteraciones, '
                  f'métrica {valores[0]:.4f} -> {valores[-1]:.4f}')
    print(f'  Métrica final (más negativa = mejor): {info["metrica"]:.4f}')
    print(f'  Motivo de parada: {info["parada"]}')

    angulos, traslacion = parametros_rigidos(final)
    # Ojo con el sentido: en SimpleITK la transformación lleva cada punto de la FIJA (CT) al
    # punto que le corresponde en la MÓVIL (MRI). Es lo que necesita Resample: para cada
    # vóxel del CT, dónde ir a buscar su valor en la MRI
    print('  Transformación encontrada (lleva cada punto del CT a su punto en la MRI):')
    print(f'    rotación   x {angulos[0]:6.2f}°   y {angulos[1]:6.2f}°   z {angulos[2]:6.2f}°')
    print(f'    traslación x {traslacion[0]:6.1f} mm  y {traslacion[1]:6.1f} mm  z {traslacion[2]:6.1f} mm')

    print('\nPASO 4 — Resultado: la MRI registrada')
    mr_despues = remuestrear(mr, ct, final)
    corte_mr_despues = normalizar_uint8(slice_central(mr_despues))

    print('  Información Mutua (más negativa = mejor), medida igual en los tres casos:')
    mi = {
        'sin mover (identidad)': evaluar_mi(ct, mr, sitk.Euler3DTransform()),
        'centros alineados (inicio)': evaluar_mi(ct, mr, info['inicial']),
        'registrada (final)': evaluar_mi(ct, mr, final),
    }
    for nombre, valor in mi.items():
        print(f'    {nombre:<28} {valor:8.4f}')
    print('  La mayor parte de la mejora la hace la optimización, no el centrado inicial.')

    print('  Ventana: arriba = CT | MRI registrada')
    print('           abajo  = tablero ANTES | tablero DESPUES')
    print('  Después, el contorno del cráneo sigue continuo de un cuadrado de CT a uno de MRI.')
    cv2.imshow('Despues de registrar', np.vstack([
        np.hstack([con_titulo(corte_ct, 'CT (fija)'),
                   con_titulo(corte_mr_despues, 'MRI registrada (mismo corte que el CT)')]),
        np.hstack([con_titulo(checkerboard(corte_ct, corte_mr_antes), 'tablero ANTES'),
                   con_titulo(checkerboard(corte_ct, corte_mr_despues), 'tablero DESPUES')]),
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
