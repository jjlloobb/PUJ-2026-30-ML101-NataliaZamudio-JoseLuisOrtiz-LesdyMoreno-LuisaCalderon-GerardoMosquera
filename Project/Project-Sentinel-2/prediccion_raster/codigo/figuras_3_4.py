"""
Figuras del punto 3.4 (Sección 8): mapa completo, detalle de la sabana de Bogotá y mapa de confianza.

Uso (desde la carpeta codigo/, donde también está predecir_imagen.py):
  python figuras_3_4.py [prefijo_del_mapa] [carpeta_de_figuras]

prefijo_del_mapa: ruta de los archivos generados por predecir_imagen.py, sin la extensión.
                  Por defecto: ../mapa/mapa_coberturas_T18NWL_20260810_ANN (relativo a este script)
carpeta_de_figuras: dónde se guardan las figuras. Por defecto: ../figuras (relativo a este script)
                  Los archivos figura15_mapa.png, figura16_detalle_bogota.png y
                  figura17_confianza.png se SOBRESCRIBEN si ya existen.
La tabla de confianza por clase (..._confianza_por_clase.tsv) se guarda junto a los rásteres.

Este script NO vuelve a clasificar la imagen: solo lee los rásteres que produjo
predecir_imagen.py y los dibuja.
"""
# --- Librerías ---
import os     # rutas de archivos y carpetas
import sys    # argumentos de la línea de comandos
import numpy as np, pandas as pd, rasterio, matplotlib
from rasterio.enums import Resampling        # métodos para reducir la resolución al leer
from rasterio.windows import from_bounds     # convierte coordenadas en una ventana del ráster
matplotlib.use("Agg")                        # guarda las figuras en archivo, sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch         # cuadritos de color para la leyenda

# Paleta y nombres de clase compartidos con predecir_imagen.py (debe estar en la misma carpeta)
from predecir_imagen import COLORES, NOMBRES

# ---- Rutas de entrada y salida ------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))   # rutas por defecto relativas a este script
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "..", "mapa", "mapa_coberturas_T18NWL_20260810_ANN")
FIGURAS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(AQUI, "..", "figuras")
BASE, FIGURAS = os.path.normpath(BASE), os.path.normpath(FIGURAS)
os.makedirs(FIGURAS, exist_ok=True)                 # crea figuras/ si no existe
# Nombres fijos: cada ejecución sobrescribe las mismas tres figuras del informe.
FIG15 = os.path.join(FIGURAS, "figura15_mapa.png")
FIG16 = os.path.join(FIGURAS, "figura16_detalle_bogota.png")
FIG17 = os.path.join(FIGURAS, "figura17_confianza.png")

# ---- Estilo y utilidades --------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
# Formato numérico en español: coma decimal y punto de miles (p. ej. 9358.4 -> "9.358,4").
coma = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
# Paleta discreta: un color por código de clase 0-11 (Figura 3 del enunciado).
cmap = ListedColormap([np.array(COLORES[i]) / 255 for i in range(12)])
# Límites en -0,5; 0,5; 1,5; ... para que cada código entero caiga en su color exacto
# (sin esto, matplotlib trataría los códigos como una escala continua y mezclaría colores).
norm = BoundaryNorm(np.arange(-.5, 12.5), 12)
# Tabla de áreas generada por predecir_imagen.py; la clase 0 (sin datos) no va en la leyenda.
areas = pd.read_csv(f"{BASE}_areas.tsv", sep="\t")
areas = areas[areas["clase"] != 0]


def leyenda(ax, con_area=True):
    """Dibuja la leyenda de clases a la derecha del mapa.

    con_area=True agrega el área (km²) y el porcentaje de cada clase (Figura 15);
    con_area=False muestra solo el código y el nombre (Figura 16).
    """
    h = [Patch(facecolor=np.array(COLORES[c]) / 255, edgecolor="#555555", lw=.6,
               label=(f"{c}: {NOMBRES[c]} ({coma(km, 1)} km²; {coma(p, 1)} %)" if con_area else f"{c}: {NOMBRES[c]}"))
         for c, km, p in zip(areas["clase"], areas["area_km2"], areas["porcentaje"])]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8, frameon=False,
              title="Clase predicha", title_fontsize=9, alignment="left")


def ejes_km(ax):
    """Ejes en kilómetros UTM 18N, sin decimales, con sus títulos."""
    f = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.0f}")
    ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)
    ax.set_xlabel("Este UTM 18N (km)"); ax.set_ylabel("Norte UTM 18N (km)")


# ---- Lectura del mapa de coberturas -------------------------------------------------------
with rasterio.open(f"{BASE}.tif") as s:
    b = s.bounds                                    # extensión de la tesela, en metros
    # Mapa completo reducido 4 veces (1 píxel de la figura = 4 x 4 píxeles reales).
    # Resampling.mode toma la clase MÁS FRECUENTE de cada bloque: con códigos de clase
    # no se puede promediar (el promedio de 4 y 6 daría 5, una clase falsa).
    full = s.read(1, out_shape=(s.height // 4, s.width // 4), resampling=Resampling.mode)
    # Detalle de la sabana de Bogotá a resolución completa (20 m)
    # Caja en metros UTM: (oeste, sur, este, norte) -> este 562-609,8 km; norte 490,2-545 km.
    caja = (562000, b.bottom, b.right, 545000)
    det = s.read(1, window=from_bounds(*caja, transform=s.transform))   # solo esa ventana
# Esquinas en km [izquierda, derecha, abajo, arriba]: así los ejes muestran coordenadas
# reales y no números de píxel.
ext = [b.left / 1e3, b.right / 1e3, b.bottom / 1e3, b.top / 1e3]
ext_det = [caja[0] / 1e3, caja[2] / 1e3, caja[1] / 1e3, caja[3] / 1e3]

# --- Figura: mapa completo -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 6.4))
# interpolation="nearest" evita suavizar los bordes entre clases.
ax.imshow(full, cmap=cmap, norm=norm, interpolation="nearest", extent=ext)
# Recuadro discontinuo que marca la zona del detalle (Figura 16).
ax.add_patch(plt.Rectangle((ext_det[0], ext_det[2]), ext_det[1] - ext_det[0], ext_det[3] - ext_det[2],
                           fill=False, ec="black", lw=1.2, ls="--"))
ax.text(ext_det[0] + .8, ext_det[3] - 2.5, "Figura 16", fontsize=8)
ejes_km(ax); leyenda(ax)
ax.set_title("Mapa de coberturas, tesela T18NWL, 10-08-2026 (red neuronal)")
# bbox_inches="tight" evita que la leyenda quede recortada.
fig.savefig(FIG15, dpi=200, bbox_inches="tight"); plt.close(fig)

# --- Figura: detalle sabana de Bogotá ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 6.4))
ax.imshow(det, cmap=cmap, norm=norm, interpolation="nearest", extent=ext_det)
# Anotación con línea que señala el embalse del Muña (coordenadas en km UTM).
ax.annotate("Embalse del Muña", xy=(582.6, 500.5), xytext=(566, 493.5), fontsize=8,
            arrowprops=dict(arrowstyle="-", color="black", lw=.8),
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#555555", lw=.5))
ejes_km(ax); leyenda(ax, con_area=False)
ax.set_title("Detalle: sabana de Bogotá y cordillera Oriental (resolución de 20 m)")
fig.savefig(FIG16, dpi=200, bbox_inches="tight"); plt.close(fig)

# --- Figura: confianza ---------------------------------------------------------------------
# Panel izquierdo: ráster de confianza reducido 4 veces. Aquí SÍ se promedia
# (Resampling.average), porque la confianza es un porcentaje continuo.
with rasterio.open(f"{BASE}_confianza.tif") as s:
    conf = s.read(1, out_shape=(s.height // 4, s.width // 4), resampling=Resampling.average).astype(float)
conf[conf > 100] = np.nan                           # 255 = sin datos -> vacío en la figura
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
# Escala de un solo color: azul más oscuro = más confianza (de 30 % a 100 %).
im = ax.imshow(conf, cmap="Blues", vmin=30, vmax=100, extent=ext, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.set_label("Confianza (%)")
ejes_km(ax); ax.set_title("Probabilidad de la clase asignada")
# Panel derecho: se leen el mapa y la confianza a resolución completa para calcular,
# por clase, la confianza media y el % de píxeles con confianza menor que 60 %.
with rasterio.open(f"{BASE}.tif") as s1, rasterio.open(f"{BASE}_confianza.tif") as s2:
    m = s1.read(1); c = s2.read(1)
filas = []
for k in areas["clase"]:
    v = c[m == k]; filas.append((k, v.mean(), 100 * (v < 60).mean()))
tab = pd.DataFrame(filas, columns=["clase", "media", "pct_bajo"])
# Esta tabla alimenta los números de la Sección 8.5 del informe.
tab.to_csv(f"{BASE}_confianza_por_clase.tsv", sep="\t", index=False)
y = np.arange(len(tab))[::-1]                       # primera clase arriba
ax2.barh(y, tab["media"], color="#2a6f97", height=.6)
for yi, v in zip(y, tab["media"]):                  # valor escrito al lado de cada barra
    ax2.text(v + 1, yi, coma(v, 1), va="center", fontsize=8)
ax2.set_yticks(y, [f"{k}: {NOMBRES[k]}" for k in tab["clase"]]); ax2.set_xlim(0, 108)
ax2.set_xlabel("Confianza media (%)"); ax2.set_title("Confianza media por clase predicha")
ax2.spines[["top", "right"]].set_visible(False); ax2.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(FIG17, dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Resumen en pantalla ------------------------------------------------------------------
print("Figuras guardadas (se sobrescriben si existían):", FIG15, FIG16, FIG17, sep="\n  ")
# Confianza media y mediana de todo el mapa, y % de píxeles en cada rango
# (<50, 50-70, 70-90 y >=90 %): son los números del primer párrafo de la Sección 8.5.
v = c[m > 0]
print("global", v.mean(), np.median(v), [round(100 * ((v >= a) & (v < b2)).mean(), 1) for a, b2 in [(0, 50), (50, 70), (70, 90), (90, 101)]])
print(tab.round(1).to_string(index=False))
