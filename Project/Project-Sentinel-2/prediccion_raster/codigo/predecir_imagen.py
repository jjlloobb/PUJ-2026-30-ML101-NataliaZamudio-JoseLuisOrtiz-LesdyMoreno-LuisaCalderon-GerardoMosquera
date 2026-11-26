"""
Proyecto Final ML101 - Predicción de la imagen georreferenciada (inferencia)

Aplica un modelo entrenado (archivo .joblib) a TODOS los píxeles de la tesela
T18NWL y genera:
  1. <salida>.tif            mapa de clases (uint8), mismo CRS y malla que las bandas
  2. <salida>_confianza.tif  probabilidad de la clase elegida, en % (uint8, 0-100)
  3. <salida>_areas.tsv      píxeles, hectáreas, km² y % por clase
  4. <salida>.qml            estilo de QGIS (paleta discreta con nombres de clase)
  5. <salida>_vista.png      vista previa del mapa con la leyenda de la Figura 3

Uso (los archivos de salida se SOBRESCRIBEN si ya existen):
  python predecir_imagen.py <carpeta_jp2> [modelo.joblib] [salida] [--filas 512]

  carpeta_jp2   carpeta con las 10 bandas *_20m.jp2 (obligatorio)
  modelo        por defecto: modelo_ANN.joblib, en la misma carpeta que este script
  salida        prefijo de los archivos de salida, sin extensión. Por defecto:
                ../mapa/mapa_coberturas_T18NWL_20260810_ANN (relativo a este script)

Ejemplo, desde la carpeta codigo/:
  python predecir_imagen.py C:/datos/T18NWL

IMPORTANTE: el modelo debe recibir los datos EXACTAMENTE como en el entrenamiento.
En este proyecto los modelos se entrenaron con los niveles digitales (DN) de las 10
bandas en el orden BANDAS, como float32, y el escalado (StandardScaler) va dentro
del Pipeline. Por eso aquí NO se convierte a reflectancia ni se escala a mano.
"""
# --- Librerías estándar de Python ---
import argparse   # lectura de los argumentos de la línea de comandos
import glob       # búsqueda de archivos con comodines (p. ej. "*_B04_20m.jp2")
import os         # rutas de archivos y carpetas
import time       # medición del tiempo de procesamiento

# --- Librerías científicas y geoespaciales ---
import joblib                        # carga del modelo entrenado (.joblib)
import numpy as np                   # arreglos numéricos (cada bloque de imagen es una matriz)
import pandas as pd                  # tabla de áreas y exportación a TSV
import rasterio                      # lectura y escritura de rásteres georreferenciados
from rasterio.windows import Window  # ventanas: leer/escribir solo un bloque de filas

# Orden de las bandas: DEBE ser el mismo orden de las columnas con que se entrenó el modelo.
# Si se cambia, la red recibiría, por ejemplo, el rojo donde espera el azul.
BANDAS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]

# Nombre de cada código de clase (convención de la Figura 2 del enunciado).
# Se usa en la tabla de áreas, en el estilo de QGIS, en los metadatos y en la leyenda.
NOMBRES = {0: "Sin datos", 1: "Saturado o defectuoso", 2: "Área oscura", 3: "Sombra de nube",
           4: "Vegetación", 5: "No vegetado", 6: "Agua", 7: "Sin clasificar",
           8: "Nube prob. media", 9: "Nube prob. alta", 10: "Cirros delgados", 11: "Nieve"}
# Colores de la leyenda de la Figura 3 del enunciado (R, G, B)
COLORES = {0: (0, 0, 0), 1: (255, 0, 0), 2: (51, 51, 51), 3: (110, 70, 40), 4: (0, 255, 0),
           5: (255, 255, 0), 6: (0, 51, 204), 7: (128, 128, 128), 8: (200, 200, 200),
           9: (255, 255, 255), 10: (0, 255, 255), 11: (255, 130, 255)}


def ruta_banda(carpeta, banda):
    """Devuelve la ruta del archivo JP2 de una banda (p. ej. *_B04_20m.jp2).

    Se busca por el sufijo para que funcione aunque los archivos tengan otro prefijo.
    Si no hay exactamente un archivo por banda, se detiene con un error claro en
    lugar de usar un archivo equivocado.
    """
    hits = glob.glob(os.path.join(carpeta, f"*_{banda}_20m.jp2"))
    if len(hits) != 1:
        raise FileNotFoundError(f"Se esperaba un archivo para {banda}, se encontraron: {hits}")
    return hits[0]


def predecir(carpeta, ruta_modelo, salida, filas_bloque=512):
    """Clasifica todos los píxeles de la tesela y escribe el mapa, la confianza y las áreas.

    carpeta       carpeta con las 10 bandas JP2
    ruta_modelo   archivo .joblib con el Pipeline entrenado (escalado + red)
    salida        prefijo de los archivos de salida, sin extensión
    filas_bloque  número de filas de la imagen que se procesan a la vez
    """
    # ---- 0. Cargar el modelo ------------------------------------------------------
    modelo = joblib.load(ruta_modelo)
    # classes_ son las etiquetas reales que aprendió el modelo, en el orden de sus salidas.
    # predict_proba devuelve una columna por clase en ESE orden; por eso se usa para
    # traducir "columna con mayor probabilidad" -> "código de clase".
    clases_modelo = modelo.classes_                    # p. ej. [2, 3, ..., 10]
    # La red (y la mayoría de modelos de scikit-learn) da probabilidades; si el modelo
    # no las tiene, el script igual funciona, pero sin ráster de confianza.
    tiene_proba = hasattr(modelo, "predict_proba")

    # ---- 1. Abrir las 10 bandas y comprobar que están alineadas --------------------
    # Se abren los archivos sin leer todavía los datos (se leerán por bloques en el paso 3).
    fuentes = [rasterio.open(ruta_banda(carpeta, b)) for b in BANDAS]
    ref = fuentes[0]                       # B01 es la referencia de la georreferenciación
    alto, ancho = ref.height, ref.width    # 5.490 x 5.490 píxeles
    for s in fuentes[1:]:
        # Mismo CRS y misma transformación afín: la fila/columna (i, j) es el mismo
        # punto del terreno en todas las bandas.
        assert s.crs == ref.crs and s.transform == ref.transform, f"{s.name} no está alineada"

    # ---- 2. Perfil de salida: mismo CRS, transformación y tamaño que las bandas ----
    #      Esto es lo que hace que el mapa quede georreferenciado y se superponga
    #      exactamente a la imagen original en QGIS.
    #      uint8 alcanza para los códigos 0-11; nodata=0 marca los píxeles sin datos;
    #      DEFLATE comprime sin pérdida y el teselado (512 x 512) agiliza la visualización.
    perfil = dict(driver="GTiff", height=alto, width=ancho, count=1, dtype="uint8",
                  crs=ref.crs, transform=ref.transform, nodata=0,
                  compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    # Crea la carpeta de salida si no existe (p. ej. mapa/).
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    try:
        # Modo "w": si el archivo ya existe, se SOBRESCRIBE.
        # photometric="palette" indica que los valores son índices de una paleta de colores.
        mapa = rasterio.open(f"{salida}.tif", "w", **perfil, photometric="palette")
    except rasterio.errors.RasterioIOError as e:
        # En Windows un archivo abierto en QGIS queda bloqueado y no se puede sobrescribir.
        raise SystemExit(f"No se pudo escribir {salida}.tif: si está abierto en QGIS u otro programa, "
                         f"ciérrelo y vuelva a ejecutar.\n({e})")
    # La paleta (colores de la Figura 3) debe escribirse ANTES que los datos
    mapa.write_colormap(1, {k: v + (255,) for k, v in COLORES.items()})
    # Ráster de confianza: mismo perfil, pero 255 marca "sin datos", porque 0 es una
    # confianza válida (0 %).
    conf = rasterio.open(f"{salida}_confianza.tif", "w", **{**perfil, "nodata": 255}) if tiene_proba else None
    # Contador de píxeles por código de clase (0-255), acumulado bloque a bloque.
    conteo = np.zeros(256, dtype=np.int64)

    # ---- 3. Recorrer la imagen por bloques de filas (controla la memoria) ----------
    # Leer las 10 bandas completas ocuparía ~600 MB; por bloques de 512 filas se usa
    # una fracción de esa memoria y el resultado es el mismo.
    t0 = time.time()
    for r0 in range(0, alto, filas_bloque):
        n_filas = min(filas_bloque, alto - r0)     # el último bloque puede ser más corto
        ventana = Window(0, r0, ancho, n_filas)    # (columna inicial, fila inicial, ancho, alto)
        # (10, n_filas, ancho) -> (n_filas*ancho, 10): una fila por píxel, una columna por banda
        bloque = np.stack([s.read(1, window=ventana) for s in fuentes])
        X = bloque.reshape(len(BANDAS), -1).T

        validos = X.any(axis=1)                       # 10 bandas en 0 -> sin datos (clase 0)
        etiquetas = np.zeros(len(X), dtype=np.uint8)              # por defecto: clase 0
        confianza = np.full(len(X), 255, dtype=np.uint8)          # por defecto: sin datos
        if validos.any():
            Xv = X[validos].astype("float32")         # mismo formato que en el entrenamiento
            if tiene_proba:
                # p tiene una fila por píxel y una columna por clase (suman 1 por fila).
                p = modelo.predict_proba(Xv)          # una sola pasada: clase y confianza
                k = p.argmax(axis=1)                  # columna de la clase más probable
                etiquetas[validos] = clases_modelo[k].astype(np.uint8)   # columna -> código
                # Confianza = probabilidad de la clase elegida, en % entero (0-100).
                confianza[validos] = np.round(100 * p[np.arange(len(k)), k]).astype(np.uint8)
            else:
                etiquetas[validos] = modelo.predict(Xv).astype(np.uint8)

        # Se vuelve a dar forma de imagen (n_filas x ancho) y se escribe en la misma
        # ventana de la que se leyó: así cada píxel queda en su posición exacta.
        mapa.write(etiquetas.reshape(n_filas, ancho), 1, window=ventana)
        if conf:
            conf.write(confianza.reshape(n_filas, ancho), 1, window=ventana)
        conteo += np.bincount(etiquetas, minlength=256)   # suma los píxeles de cada clase
        print(f"  filas {r0:>4}-{r0 + n_filas:<4} listas ({time.time() - t0:5.0f} s)", flush=True)

    # ---- 4. Nombres de clase como metadatos del GeoTIFF ----------------------------
    # Quedan dentro del archivo (clase_4 = Vegetación, ...) y se ven en QGIS o con gdalinfo.
    mapa.update_tags(1, **{f"clase_{k}": v for k, v in NOMBRES.items()})
    mapa.close()          # cerrar es lo que termina de escribir el archivo en el disco
    if conf:
        conf.close()

    # ---- 5. Áreas por clase: cada píxel mide 20 m x 20 m = 400 m² -----------------
    # El área del píxel se toma de la transformación afín (a = ancho, e = alto, negativo).
    area_px = abs(ref.transform.a * ref.transform.e)
    presentes = np.nonzero(conteo)[0]                 # solo las clases que aparecen en el mapa
    areas = pd.DataFrame({"clase": presentes, "nombre": [NOMBRES[c] for c in presentes],
                          "pixeles": conteo[presentes]})
    areas["area_ha"] = areas["pixeles"] * area_px / 1e4    # 1 ha = 10.000 m²
    areas["area_km2"] = areas["pixeles"] * area_px / 1e6   # 1 km² = 1.000.000 m²
    # El porcentaje se calcula sobre el área válida: la clase 0 (sin datos) no cuenta.
    validas = areas["clase"] != 0
    areas["porcentaje"] = np.where(validas, 100 * areas["pixeles"] / areas.loc[validas, "pixeles"].sum(), np.nan)
    areas.to_csv(f"{salida}_areas.tsv", sep="\t", index=False)
    escribir_qml(salida, presentes)
    print(areas.round(2).to_string(index=False))
    return areas


def escribir_qml(salida, clases_presentes):
    """Estilo de QGIS: al cargar el GeoTIFF, QGIS lo lee automáticamente si el .qml tiene
    el mismo nombre. Simbología 'Valores en paleta/únicos' con nombre y color por clase."""
    # Una entrada <paletteEntry> por clase presente: valor, color en hexadecimal (#RRGGBB)
    # y la etiqueta que se verá en la leyenda de QGIS. La clase 0 no se incluye, para
    # que los píxeles sin datos queden transparentes.
    filas = "\n".join(
        f'        <paletteEntry value="{c}" color="#{r:02x}{g:02x}{b:02x}" alpha="255" label="{c}: {NOMBRES[c]}"/>'
        for c in clases_presentes if c != 0 for r, g, b in [COLORES[c]])
    # Plantilla mínima de un estilo de QGIS para un ráster "paletizado" (valores únicos).
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <rasterTransparency/>
      <colorPalette>
{filas}
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    with open(f"{salida}.qml", "w", encoding="utf-8") as fh:
        fh.write(qml)


def vista_previa(salida, titulo):
    """Guarda <salida>_vista.png: una imagen rápida del mapa con su leyenda, para revisar
    el resultado sin abrir QGIS (no es una de las figuras del informe)."""
    # matplotlib se importa aquí para que la predicción no dependa de él.
    import matplotlib
    matplotlib.use("Agg")                  # guarda la figura en archivo, sin abrir ventanas
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    with rasterio.open(f"{salida}.tif") as s:
        f = 5                                          # se reduce 5x solo para la figura
        # Resampling.mode: cada píxel reducido toma la clase más frecuente del bloque;
        # promediar códigos de clase crearía clases que no existen.
        img = s.read(1, out_shape=(s.height // f, s.width // f), resampling=rasterio.enums.Resampling.mode)
        b = s.bounds                                   # extensión en metros (UTM 18N)
    areas = pd.read_csv(f"{salida}_areas.tsv", sep="\t")
    # Un color por código 0-11; BoundaryNorm hace que cada entero caiga en su color exacto.
    cmap = ListedColormap([np.array(COLORES[i]) / 255 for i in range(12)])
    fig, ax = plt.subplots(figsize=(9, 6.6))
    ax.imshow(img, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 12.5), 12), interpolation="nearest",
              extent=[b.left / 1000, b.right / 1000, b.bottom / 1000, b.top / 1000])
    # Leyenda con el porcentaje de cada clase, con coma decimal.
    leyenda = [Patch(facecolor=np.array(COLORES[c]) / 255, edgecolor="#555555", lw=.6,
                     label=f"{c} {NOMBRES[c]} ({p:.1f} %)".replace(f"{p:.1f}", f"{p:.1f}".replace(".", ",")))
               for c, p in zip(areas["clase"], areas["porcentaje"]) if c != 0]
    ax.legend(handles=leyenda, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8, frameon=False)
    ax.set_xlabel("Este UTM 18N (km)"); ax.set_ylabel("Norte UTM 18N (km)"); ax.set_title(titulo)
    fig.savefig(f"{salida}_vista.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    # Rutas por defecto relativas a la carpeta de este script (funciona desde cualquier carpeta)
    AQUI = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Predicción de la imagen georreferenciada (punto 3.4)")
    # Único argumento obligatorio: dónde están las bandas.
    ap.add_argument("carpeta", help="carpeta con las 10 bandas *_20m.jp2")
    # nargs="?" hace que el argumento sea opcional y use el valor por defecto si se omite.
    ap.add_argument("modelo", nargs="?", default=os.path.join(AQUI, "modelo_ANN.joblib"),
                    help="modelo entrenado (.joblib); por defecto modelo_ANN.joblib junto a este script")
    ap.add_argument("salida", nargs="?",
                    default=os.path.join(AQUI, "..", "mapa", "mapa_coberturas_T18NWL_20260810_ANN"),
                    help="prefijo de salida sin extensión; por defecto ../mapa/mapa_coberturas_T18NWL_20260810_ANN")
    ap.add_argument("--filas", type=int, default=512, help="filas por bloque")
    a = ap.parse_args()
    salida = os.path.normpath(a.salida)    # limpia la ruta (resuelve "..", separadores)
    predecir(a.carpeta, a.modelo, salida, a.filas)
    vista_previa(salida, f"Coberturas T18NWL 10-08-2026 ({os.path.basename(a.modelo)})")
    # Resumen final: qué archivos quedaron escritos (o sobrescritos).
    print("\nArchivos escritos (se sobrescriben si existían):")
    for suf in [".tif", "_confianza.tif", "_areas.tsv", ".qml", "_vista.png"]:
        print(f"  {salida}{suf}")
