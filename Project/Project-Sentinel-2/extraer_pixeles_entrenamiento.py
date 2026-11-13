"""
Proyecto Final ML101 - Sección 3.1: Extracción de datos y construcción de la tabla
Sentinel-2 L2A, tesela T18NWL, 10-08-2026 (Bogotá), todas las bandas a 20 m.

Objetivo:
  Convertir la imagen multiespectral (10 bandas) en una tabla NO espacial donde
  cada fila es un píxel que cae dentro de un polígono de entrenamiento, con sus
  10 valores de banda y la clase del polígono.

Entradas (en la misma carpeta):
  - 10 archivos JP2, uno por banda:  *_B01_20m.jp2, *_B02_20m.jp2, ..., *_B12_20m.jp2
  - 1 archivo TSV con los polígonos: *PolygonsClasses*.tsv  (columnas WKT, fid, class)

Salida: training_pixels.tsv con las columnas
  fid, Latitude, Longitude, B01..B12 (10 bandas), class

Uso:
  python extraer_pixeles_entrenamiento.py <carpeta_con_jp2_y_tsv> [salida.tsv]
"""
# --- Librerías estándar de Python ---
import glob   # buscar archivos con comodines (p. ej. "*_B04_20m.jp2")
import os     # construir rutas de archivos
import sys    # leer los argumentos de la línea de comandos

# --- Librerías científicas y geoespaciales ---
import numpy as np            # arreglos numéricos (la imagen se maneja como matriz)
import pandas as pd           # tablas (DataFrame) y exportación a TSV
import geopandas as gpd       # tablas con geometrías (los polígonos)
import rasterio               # lectura de imágenes georreferenciadas (JP2, GeoTIFF)
import shapely                # operaciones geométricas (punto dentro de polígono)
from pyproj import Transformer  # conversión entre sistemas de coordenadas

# Orden de las 10 bandas: este será también el orden de las columnas en la tabla.
BANDAS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]

# Convención de clases de la Figura 2 del enunciado (etiqueta entera -> nombre).
# Se usa solo para mostrar el resumen final; la etiqueta que se guarda es el número.
CLASES = {
    0: "SIN_DATOS", 1: "SATURADO_O_DEFECTUOSO", 2: "PIXELES_AREA_OSCURA",
    3: "SOMBRAS_DE_NUBES", 4: "VEGETACION", 5: "NO_VEGETADO", 6: "AGUA",
    7: "SIN_CLASIFICAR", 8: "NUBE_PROBABILIDAD_MEDIA", 9: "NUBE_PROBABILIDAD_ALTA",
    10: "CIRROS_DELGADOS", 11: "NIEVE",
}


def ruta_banda(carpeta, banda):
    """Devuelve la ruta del archivo JP2 de una banda (p. ej. *_B04_20m.jp2).

    Se busca por el sufijo y no por el nombre completo, para que el script funcione
    aunque los archivos tengan prefijos distintos. Si no hay exactamente un archivo
    por banda, se detiene con un error claro en lugar de usar un archivo equivocado.
    """
    hits = glob.glob(os.path.join(carpeta, f"*_{banda}_20m.jp2"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Se esperaba un archivo para {banda}, se encontraron: {hits}")
    return hits[0]


def extraer(carpeta, salida="training_pixels.tsv"):
    # =============================================================================
    # Paso 1: apilar las 10 bandas en un único arreglo (10, filas, columnas)
    # =============================================================================
    rutas = [ruta_banda(carpeta, b) for b in BANDAS]

    # Se toma la primera banda (B01) como referencia de la georreferenciación:
    #  - transform: transformación afín que convierte (fila, columna) <-> (X, Y) en metros
    #  - crs: sistema de referencia de coordenadas (aquí WGS 84 / UTM 18N, EPSG:32618)
    #  - alto, ancho: tamaño de la imagen en píxeles (5490 x 5490)
    with rasterio.open(rutas[0]) as ref:
        transform, crs = ref.transform, ref.crs
        alto, ancho = ref.height, ref.width

    # Arreglo vacío para las 10 bandas. uint16 = entero sin signo de 16 bits,
    # el mismo tipo de dato de los JP2 (~600 MB en memoria).
    pila = np.empty((len(BANDAS), alto, ancho), dtype=np.uint16)
    for i, r in enumerate(rutas):
        with rasterio.open(r) as src:
            # Todas las bandas deben estar alineadas (mismo CRS y misma malla); si no,
            # la fila/columna (i, j) no sería el mismo punto del terreno en todas ellas.
            assert src.crs == crs and src.transform == transform, f"{r} no está alineada"
            pila[i] = src.read(1)   # read(1) lee la banda 1 del archivo como matriz 2D
        print(f"  banda {BANDAS[i]} cargada")

    # =============================================================================
    # Paso 2: cargar los polígonos de entrenamiento
    # =============================================================================
    # El TSV trae la geometría como texto WKT, p. ej. "POLYGON ((516558.9 529752.4, ...))".
    archivo_poli = glob.glob(os.path.join(carpeta, "*PolygonsClasses*.tsv"))[0]
    df = pd.read_csv(archivo_poli, sep="\t")

    # Se convierte el texto WKT en geometrías reales. Las coordenadas ya están en
    # UTM 18N (metros), igual que la imagen, así que se les asigna el mismo CRS
    # y no hace falta reproyectar.
    gdf = gpd.GeoDataFrame(df.drop(columns="WKT"),
                           geometry=gpd.GeoSeries.from_wkt(df["WKT"]), crs=crs)
    gdf["class"] = gdf["class"].astype(int)   # en el TSV la clase viene como texto ("5")
    print(f"  {len(gdf)} polígonos; clases: {sorted(gdf['class'].unique())}")

    # =============================================================================
    # Paso 3: para cada polígono, encontrar los píxeles cuyo CENTRO cae dentro
    # =============================================================================
    bloques = []   # aquí se acumula un bloque de filas por cada polígono
    vacios = 0     # polígonos que no contienen ningún centro de píxel
    inv = ~transform   # transformación inversa: (X, Y) en metros -> (columna, fila)

    for fid, clase, geom in zip(gdf["fid"], gdf["class"], gdf.geometry):
        # 3a. Rectángulo envolvente del polígono, en metros.
        minx, miny, maxx, maxy = geom.bounds

        # 3b. Se pasa ese rectángulo a filas/columnas de la imagen. En la imagen la
        #     fila 0 está arriba (norte), por eso la esquina superior izquierda usa
        #     (minx, maxy) y la inferior derecha (maxx, miny).
        c0, r0 = inv * (minx, maxy)          # esquina superior izquierda -> columna, fila
        c1, r1 = inv * (maxx, miny)          # esquina inferior derecha

        # 3c. Se redondea hacia afuera (floor/ceil, +1 de margen) para no perder
        #     píxeles del borde, y se recorta a los límites de la imagen.
        r0, c0 = max(int(np.floor(r0)), 0), max(int(np.floor(c0)), 0)
        r1, c1 = min(int(np.ceil(r1)) + 1, alto), min(int(np.ceil(c1)) + 1, ancho)
        if r0 >= r1 or c0 >= c1:   # el polígono quedó fuera de la imagen
            vacios += 1
            continue

        # 3d. Malla con todas las filas (rr) y columnas (cc) dentro del rectángulo.
        rr, cc = np.mgrid[r0:r1, c0:c1]

        # 3e. Coordenadas del CENTRO de cada píxel (de ahí el +0,5).
        #     transform.c / transform.f = X / Y de la esquina superior izquierda de la imagen
        #     transform.a = ancho del píxel (+20 m); transform.e = alto del píxel (-20 m)
        xs = transform.c + (cc + 0.5) * transform.a   # X del centro del píxel (UTM)
        ys = transform.f + (rr + 0.5) * transform.e   # Y del centro del píxel (UTM)

        # 3f. Se conservan solo los píxeles cuyo centro está dentro del polígono.
        #     Así no se cuentan píxeles mixtos que solo tocan el borde
        #     (equivale a rasterizar con all_touched=False).
        dentro = shapely.contains_xy(geom, xs, ys)
        if not dentro.any():
            vacios += 1
            continue

        # =========================================================================
        # Paso 4: copiar los 10 valores de banda, el fid y la clase
        # =========================================================================
        # pila[:, filas, columnas] devuelve (10, n); .T lo transpone a (n, 10):
        # una fila por píxel y una columna por banda.
        valores = pila[:, rr[dentro], cc[dentro]].T
        n = dentro.sum()   # número de píxeles de este polígono
        # Bloque de n filas: fid | X | Y | 10 bandas | clase
        bloques.append(np.column_stack([np.full(n, fid), xs[dentro], ys[dentro],
                                        valores, np.full(n, clase)]))

    # Se unen los bloques de todos los polígonos en una sola tabla.
    tabla = pd.DataFrame(np.vstack(bloques),
                         columns=["fid", "X_utm", "Y_utm"] + BANDAS + ["class"])
    # np.column_stack convirtió todo a decimal; se devuelven a entero las columnas
    # que lo son (fid, clase y valores de banda).
    for c in ["fid", "class"] + BANDAS:
        tabla[c] = tabla[c].astype(int)
    print(f"  polígonos sin ningún centro de píxel dentro: {vacios}")

    # =============================================================================
    # Paso 5: limpieza de la tabla
    # =============================================================================
    antes = len(tabla)
    # 5a. Mismo píxel con la misma clase repetido (polígonos superpuestos): se deja uno.
    tabla = tabla.drop_duplicates(subset=["X_utm", "Y_utm", "class"])
    # 5b. Mismo píxel reclamado por dos clases distintas: la etiqueta es ambigua,
    #     así que se eliminan todas sus copias (keep=False).
    conflicto = tabla.duplicated(subset=["X_utm", "Y_utm"], keep=False)
    tabla = tabla[~conflicto]
    # 5c. Píxeles sin datos: las 10 bandas valen 0.
    sin_datos = (tabla[BANDAS] == 0).all(axis=1)
    tabla = tabla[~sin_datos]
    print(f"  filas eliminadas: {antes - len(tabla)} "
          f"({conflicto.sum()} en conflicto, {sin_datos.sum()} sin datos)")

    # =============================================================================
    # Paso 6: convertir las coordenadas UTM 18N a latitud / longitud (WGS84)
    # =============================================================================
    # EPSG:4326 = coordenadas geográficas en grados. always_xy=True fija el orden
    # (X=longitud, Y=latitud) para evitar que se inviertan.
    tf = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(tabla["X_utm"].values, tabla["Y_utm"].values)
    # 7 decimales de grado ~ 1 cm: precisión más que suficiente para píxeles de 20 m.
    tabla.insert(1, "Latitude", np.round(lat, 7))
    tabla.insert(2, "Longitude", np.round(lon, 7))
    tabla = tabla.drop(columns=["X_utm", "Y_utm"])   # ya no se necesitan las UTM

    # =============================================================================
    # Paso 7: exportar la tabla como TSV (separada por tabulaciones)
    # =============================================================================
    tabla.to_csv(salida, sep="\t", index=False)   # index=False: sin columna de índice

    # Resumen por clase: número de píxeles y número de polígonos distintos.
    resumen = tabla.groupby("class").agg(pixeles=("fid", "size"),
                                         poligonos=("fid", "nunique"))
    resumen.insert(0, "nombre", [CLASES[c] for c in resumen.index])
    print(f"\n  guardados {len(tabla):,} píxeles en {salida}")
    print(resumen.to_string())
    return tabla


if __name__ == "__main__":
    # Argumentos opcionales: carpeta de entrada (por defecto la actual)
    # y nombre del archivo de salida (por defecto training_pixels.tsv).
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "."
    salida = sys.argv[2] if len(sys.argv) > 2 else "training_pixels.tsv"
    extraer(carpeta, salida)
