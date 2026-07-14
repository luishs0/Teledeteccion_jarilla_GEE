# -*- coding: utf-8 -*-
"""
================================================================================
ANALISIS TOPOGRAFICO DEL INCENDIO DE JARILLA (2025)
Procesado integro en Python del Modelo Digital del Terreno (MDT05, 5 m)
================================================================================
 
Este script reune todo el procesamiento topografico del trabajo: a partir del
MDT05 del PNOA-LiDAR (Instituto Geografico Nacional, 5 m de resolucion) calcula
la pendiente y la orientacion de las laderas mediante el algoritmo de Horn (1981),
las recorta al perimetro del incendio, obtiene las estadisticas, evalua la
alineacion viento-ladera y exporta las capas georreferenciadas (GeoTIFF en
EPSG:25830), sus estilos para QGIS (.qml) y los mapas en imagen (PNG).
 
Se prescinde de librerias geoespaciales pesadas (GDAL/rasterio): el MDT se lee
directamente por desplazamientos de fila ("strips") con numpy y Pillow, lo que
permite procesar un mosaico de mas de 2 GB sin cargarlo entero en memoria.
 
Sistema de referencia de todas las salidas: ETRS89 / UTM zona 30N (EPSG:25830).
 
Estructura del script:
  0. Configuracion y rutas
  1. Lectura de la cabecera del MDT (geotransformacion)
  2. Lectura del perimetro del incendio (GeoPackage / WKB)
  3. Definicion de la ventana de recorte
  4. Lectura de la ventana de alturas del MDT
  5. Pendiente y orientacion (algoritmo de Horn, 1981)
  6. Mascara del perimetro
  7. Estadisticas dentro del perimetro
  8. Alineacion viento-ladera
  9. Escritura de GeoTIFFs continuos georreferenciados (pendiente, orientacion, MDE)
 10. Capas categoricas (sectores, umbria/solana, alineacion) + estilos .qml
 11. Mapas en imagen (PNG)
 12. Recorte del sombreado (hillshade) al perimetro
================================================================================
"""
 
import os, struct, json, math, gc, sqlite3
import numpy as np
from PIL import Image, ImageDraw
from PIL.TiffImagePlugin import ImageFileDirectory_v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
 
# Sin este limite, Pillow rechaza imagenes muy grandes por seguridad.
Image.MAX_IMAGE_PIXELS = None
 
 
# ============================================================================
# 0. CONFIGURACION Y RUTAS
# ============================================================================
# MDT05 (mosaico de las hojas PNOA-LiDAR que cubren el area de estudio) y
# perimetro oficial del incendio. Carpeta de salida de los productos.
RAS  = "PROYECTO_GIS/datos/raster/mdt05_jarilla_mosaico.tif"     # MDT05, 5 m
GPKG = "PROYECTO_GIS/datos/vector/incendio_jarilla.gpkg"         # perimetro (MULTIPOLYGON)
HILL = "PROYECTO_GIS/datos/raster/hillshade_jarilla.tif"         # sombreado (misma rejilla)
DST  = "PROYECTO_GIS/datos/raster"                               # destino de las capas
os.makedirs(DST, exist_ok=True)
 
# Cadena WKT del sistema de referencia, para los ficheros .prj que acompanan a
# cada GeoTIFF. Incluye explicitamente AUTHORITY EPSG:25830.
PRJWKT = ('PROJCS["ETRS89 / UTM zone 30N",GEOGCS["ETRS89",'
  'DATUM["European_Terrestrial_Reference_System_1989",'
  'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
  'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
  'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-3],'
  'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
  'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","25830"]]')
 
 
# ============================================================================
# 1. LECTURA DE LA CABECERA DEL MDT (GEOTRANSFORMACION)
# ============================================================================
# Un GeoTIFF guarda, en su cabecera (tags), la informacion que situa cada pixel
# en coordenadas reales. Se leen sin cargar los pixeles:
#   33550 -> tamano de pixel (PixelScale)
#   33922 -> punto de anclaje (ModelTiepoint): liga un pixel a una coordenada
#   273   -> lista de offsets de fila (un byte de inicio por fila; RowsPerStrip=1)
im = Image.open(RAS)
W, H = im.size                       # ancho y alto del mosaico, en pixeles
t = im.tag_v2
px = t[33550][0]                     # tamano de pixel = 5.0 m
tie = t[33922]                       # (i, j, k, X, Y, Z)
OX = tie[3] - tie[0] * px            # coordenada X del origen (esquina superior-izquierda)
OY = tie[4] + tie[1] * px            # coordenada Y del origen
strip_offsets = t[273]               # byte donde empieza cada fila de la imagen
 
 
# ============================================================================
# 2. LECTURA DEL PERIMETRO DEL INCENDIO (GEOPACKAGE / WKB)
# ============================================================================
# El perimetro es un MULTIPOLYGON dentro de un GeoPackage (una base de datos
# SQLite). La geometria se almacena en binario (formato GPB: cabecera propia +
# WKB estandar). Se descodifica a mano para obtener los anillos (listas de
# vertices). El anillo exterior es el contorno del incendio; los siguientes son
# huecos interiores (zonas no quemadas).
blob = sqlite3.connect(GPKG).execute("SELECT geom FROM jarilla_2025").fetchone()[0]
flags = blob[3]
env = (flags >> 1) & 7                       # indica el tamano de la envolvente
env_sz = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
wkb = blob[8 + env_sz[env]:]                 # se salta la cabecera GPB -> WKB puro
 
pos = 0
def _u8():
    global pos; v = wkb[pos]; pos += 1; return v
def _u32():
    global pos; v = struct.unpack_from('<I', wkb, pos)[0]; pos += 4; return v
 
_u8(); _u32(); npoly = _u32()                # orden de bytes, tipo de geometria, nº poligonos
rings = []
for _ in range(npoly):
    _u8(); _u32(); nr = _u32()               # cabecera de poligono + nº de anillos
    for _ in range(nr):
        nn = _u32()                          # nº de vertices del anillo
        pts = np.frombuffer(wkb, dtype='<f8', count=2 * nn, offset=pos).reshape(nn, 2)
        pos += 16 * nn
        rings.append(pts.copy())
outer = rings[0]                             # contorno exterior del incendio
 
minx = min(r[:, 0].min() for r in rings); maxx = max(r[:, 0].max() for r in rings)
miny = min(r[:, 1].min() for r in rings); maxy = max(r[:, 1].max() for r in rings)
 
 
# ============================================================================
# 3. DEFINICION DE LA VENTANA DE RECORTE
# ============================================================================
# Para no procesar todo el mosaico, se calcula la franja de filas y columnas que
# cubre el perimetro. Se anade un margen de 3 pixeles porque el algoritmo de Horn
# necesita los vecinos de cada pixel del borde.
M = 3
col0 = max(0, int(math.floor((minx - OX) / px)) - M)
col1 = min(W, int(math.ceil((maxx - OX) / px)) + M)
row0 = max(0, int(math.floor((OY - maxy) / px)) - M)   # fila superior = Y mayor
row1 = min(H, int(math.ceil((OY - miny) / px)) + M)
nx, ny = col1 - col0, row1 - row0
gx0 = OX + col0 * px                          # coordenadas del origen del recorte
gy0 = OY - row0 * px
 
 
# ============================================================================
# 4. LECTURA DE LA VENTANA DE ALTURAS DEL MDT
# ============================================================================
# Se lee fila a fila, saltando con seek() al byte de inicio de cada una y
# leyendo unicamente las columnas de la ventana. El MDT esta en float32 (4 bytes
# por pixel). El resultado, 'dem', es la matriz de altitudes del entorno del
# incendio. Los valores de relleno (sin dato) se marcan como NaN.
dem = np.empty((ny, nx), dtype='<f4')
with open(RAS, 'rb') as f:
    for i, r in enumerate(range(row0, row1)):
        f.seek(strip_offsets[r] + col0 * 4)
        dem[i] = np.frombuffer(f.read(nx * 4), dtype='<f4')
dem = dem.astype(np.float32)
mask_void = (dem < -1000) | (~np.isfinite(dem))
dem[mask_void] = np.nan
 
 
# ============================================================================
# 5. PENDIENTE Y ORIENTACION (ALGORITMO DE HORN, 1981)
# ============================================================================
# El algoritmo de Horn estima los gradientes del terreno en X (Este-Oeste) y en Y
# (Norte-Sur) usando una ventana movil de 3x3 pixeles, ponderando el doble los
# vecinos contiguos respecto a los diagonales. Con numpy se obtienen los ocho
# vecinos desplazando la matriz completa, evitando bucles pixel a pixel.
#
# Disposicion de la ventana 3x3:
#     a b c        a=NO  b=N  c=NE
#     d X e        d=O   X    e=E
#     f g h        f=SO  g=S  h=SE
zf = np.where(np.isnan(dem), np.float32(np.nanmean(dem)), dem).astype(np.float32)
a = zf[:-2, :-2]; b = zf[:-2, 1:-1]; c = zf[:-2, 2:]
d = zf[1:-1, :-2];                  e2 = zf[1:-1, 2:]
f_ = zf[2:, :-2]; g = zf[2:, 1:-1]; h = zf[2:, 2:]
 
# Gradientes de Horn (variacion de altura por metro) en X e Y.
dzdx = ((c + 2 * e2 + h) - (a + 2 * d + f_)) / np.float32(8 * px)
dzdy = ((f_ + 2 * g + h) - (a + 2 * b + c)) / np.float32(8 * px)
del zf, a, b, c, d, e2, f_, g, h; gc.collect()
 
# Pendiente en grados: angulo de la cuesta total (combinacion de ambos gradientes).
slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
 
# Orientacion (aspect): azimut de la direccion de maxima pendiente, medido de 0 a
# 360 grados en sentido horario desde el Norte (0=N, 90=E, 180=S, 270=O). Las
# superficies llanas, que no miran a ningun sitio, se marcan con -1.
aspect = 90.0 - np.degrees(np.arctan2(dzdy, -dzdx))
aspect = np.where(aspect < 0, aspect + 360, aspect)
aspect = np.where(aspect >= 360, aspect - 360, aspect)
flat = np.hypot(dzdx, dzdy) < 1e-9
aspect = np.where(flat, -1.0, aspect).astype(np.float32)
del dzdx, dzdy, flat; gc.collect()
 
# Horn pierde una orla de 1 pixel; se rellena por replica para conservar el tamano.
slope_deg = np.pad(slope_deg, 1, mode='edge')
aspect = np.pad(aspect, 1, mode='edge')
if mask_void.any():
    slope_deg[mask_void] = np.nan
    aspect[mask_void] = np.nan
 
 
# ============================================================================
# 6. MASCARA DEL PERIMETRO
# ============================================================================
# Para limitar el analisis a la superficie quemada se "pinta" el poligono del
# perimetro sobre un lienzo binario: 1 (blanco) dentro del contorno exterior, 0
# (negro) en los huecos interiores. El resultado es una mascara booleana.
def _to_px(ring):
    cx = (ring[:, 0] - gx0) / px              # columna (coordenada -> pixel)
    cy = (gy0 - ring[:, 1]) / px              # fila
    return list(zip(cx.tolist(), cy.tolist()))
 
mimg = Image.new('1', (nx, ny), 0)
dr = ImageDraw.Draw(mimg)
dr.polygon(_to_px(rings[0]), fill=1)          # contorno exterior
for ring in rings[1:]:
    dr.polygon(_to_px(ring), fill=0)          # huecos no quemados
mask = np.array(mimg, dtype=bool)
del mimg, dr
 
inperim = mask & np.isfinite(slope_deg)
npix = int(inperim.sum())
pix_ha = (px * px) / 10000.0                  # superficie de un pixel en hectareas (25 m2)
 
 
# ============================================================================
# 7. ESTADISTICAS DENTRO DEL PERIMETRO
# ============================================================================
sl = slope_deg[inperim]
asp = aspect[inperim]
elev = dem[inperim]
 
# 7.1 Pendiente: reclasificacion en cinco intervalos de interes para el fuego.
slope_bins = [(0, 5), (5, 15), (15, 30), (30, 45), (45, 90)]
slope_lbl = ["0-5 (llano/suave)", "5-15 (moderada)", "15-30 (fuerte)",
             "30-45 (muy fuerte)", ">45 (escarpada)"]
slope_stats = []
for (lo, hi), lab in zip(slope_bins, slope_lbl):
    sel = (sl >= lo) & (sl < hi) if hi < 90 else (sl >= lo)
    n = int(sel.sum())
    slope_stats.append((lab, n, n * pix_ha, 100 * n / npix))
 
# 7.2 Orientacion: cada azimut se asigna a uno de los ocho sectores cardinales.
sect_lbl = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
def _sector(angle):
    if angle < 0:
        return -1                             # llano
    return int(((angle + 22.5) % 360) // 45)  # 0=N,1=NE,2=E,3=SE,4=S,5=SO,6=O,7=NO
sect_idx = np.array([_sector(x) for x in asp])
aspect_stats = [(lab, int((sect_idx == i).sum()),
                 int((sect_idx == i).sum()) * pix_ha,
                 100 * int((sect_idx == i).sum()) / npix)
                for i, lab in enumerate(sect_lbl)]
 
# 7.3 Agrupacion en umbria (NO-N-NE), solana (SE-S-SO) y transito (E, O).
umbria = int(np.isin(sect_idx, [7, 0, 1]).sum())
solana = int(np.isin(sect_idx, [3, 4, 5]).sum())
este = int((sect_idx == 2).sum()); oeste = int((sect_idx == 6).sum())
 
# 7.4 Resumenes de altitud y pendiente.
elev_stats = dict(min=float(np.nanmin(elev)), max=float(np.nanmax(elev)),
                  mean=float(np.nanmean(elev)), median=float(np.nanmedian(elev)))
slope_summary = dict(mean=float(np.nanmean(sl)), median=float(np.nanmedian(sl)),
                     p95=float(np.nanpercentile(sl, 95)), max=float(np.nanmax(sl)))
 
# ============================================================================
# 8. ALINEACIÓN VIENTO-LADERA (coseno, solo flujo NE — igual que los corredores)
# ============================================================================
VIENTO_PROCEDENCIA = 45      # procedencia del viento sinóptico dominante (NE)
UMBRAL_ALINEACION  = 0.5     # coseno >= 0.5  => dentro de ±60° de alineación perfecta

asp_valid = asp[asp >= 0]
sl_valid  = sl[asp >= 0]
cos_align = np.cos(np.radians(asp_valid - VIENTO_PROCEDENCIA))

n_alin = int((cos_align >= UMBRAL_ALINEACION).sum())                        # laderas favorables
n_crit = int(((cos_align >= UMBRAL_ALINEACION) & (sl_valid >= 15)).sum())   # favorables + pend>=15

align = dict(
    NE_alineada_pct = 100 * n_alin / npix,   # % del perímetro
    NE_alineada_ha  = n_alin * pix_ha,
    NE_critica_pct  = 100 * n_crit / npix,   # % del perímetro (alineada + pendiente>=15)
    NE_critica_ha   = n_crit * pix_ha
)
print("--- Alineación viento-ladera (flujo NE, coseno>=0.5) ---")
print("  Favorable:        %.1f %%  (%.0f ha)" % (align['NE_alineada_pct'], align['NE_alineada_ha']))
print("  Favorable + >=15: %.1f %%  (%.0f ha)" % (align['NE_critica_pct'], align['NE_critica_ha']))
 
 
# ============================================================================
# 9. ESCRITURA DE GEOTIFFS CONTINUOS GEORREFERENCIADOS
# ============================================================================
# Cada variable continua (pendiente, orientacion, MDE) se guarda como GeoTIFF
# float32 con el sistema de referencia EMBEBIDO. La clave esta en la clave
# GeoKey 3072 (ProjectedCSTypeGeoKey) = 25830: es el codigo EPSG que el MDT de
# partida no llevaba (solo tenia el nombre como texto), lo que impedia a QGIS
# reproyectarlo. Al estamparlo aqui, las capas se cargan sin errores de CRS.
# Se acompanan de los ficheros laterales .tfw (situacion) y .prj (CRS en texto).
NOD = -9999.0
def write_geotiff_float(path, arr):
    a = arr.copy().astype(np.float32)
    a[~mask] = NOD                            # fuera del perimetro -> sin dato
    img = Image.fromarray(a, mode='F')
    ifd = ImageFileDirectory_v2()
    ifd.tagtype[33550] = 12; ifd[33550] = (px, px, 0.0)                       # PixelScale
    ifd.tagtype[33922] = 12; ifd[33922] = (0.0, 0.0, 0.0, gx0, gy0, 0.0)      # Tiepoint
    ifd.tagtype[34735] = 3
    ifd[34735] = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1,
                  3072, 0, 1, 25830,           # <-- codigo EPSG (CRS proyectado)
                  3076, 0, 1, 9001)            # unidades = metros
    ifd.tagtype[34737] = 2; ifd[34737] = "ETRS89 / UTM zone 30N|"
    ifd.tagtype[42113] = 2; ifd[42113] = str(NOD)                            # valor sin dato
    img.save(path, tiffinfo=ifd)
    with open(path[:-4] + ".tfw", "w") as fw:
        fw.write(f"{px}\n0.0\n0.0\n{-px}\n{gx0 + px/2}\n{gy0 - px/2}\n")
    open(path[:-4] + ".prj", "w").write(PRJWKT)
 
write_geotiff_float(os.path.join(DST, "jarilla_pendiente.tif"), slope_deg)
write_geotiff_float(os.path.join(DST, "jarilla_orientacion.tif"), aspect)
write_geotiff_float(os.path.join(DST, "jarilla_mde.tif"), dem)
 
 
# ============================================================================
# 10. CAPAS CATEGORICAS + ESTILOS .QML PARA QGIS
# ============================================================================
# Ademas de las capas continuas, se generan versiones categoricas (enteros de un
# byte) que facilitan la simbologia en QGIS, y un fichero de estilo .qml para
# cada mapa. Valor 255 = fuera del perimetro (sin dato).
def write_geotiff_byte(path, arr_u8, nodata=255):
    img = Image.fromarray(arr_u8.astype(np.uint8), mode='L')
    ifd = ImageFileDirectory_v2()
    ifd.tagtype[33550] = 12; ifd[33550] = (px, px, 0.0)
    ifd.tagtype[33922] = 12; ifd[33922] = (0.0, 0.0, 0.0, gx0, gy0, 0.0)
    ifd.tagtype[34735] = 3
    ifd[34735] = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 25830, 3076, 0, 1, 9001)
    ifd.tagtype[34737] = 2; ifd[34737] = "ETRS89 / UTM zone 30N|"
    ifd.tagtype[42113] = 2; ifd[42113] = str(nodata)
    img.save(path, tiffinfo=ifd)
    with open(path[:-4] + ".tfw", "w") as fw:
        fw.write(f"{px}\n0.0\n0.0\n{-px}\n{gx0 + px/2}\n{gy0 - px/2}\n")
    open(path[:-4] + ".prj", "w").write(PRJWKT)
 
# 10.1 Orientacion en 8 sectores (0..7), 8 = llano, 255 = fuera.
sec = np.full(aspect.shape, 255, dtype=np.uint8)
v = mask & np.isfinite(aspect)
aa = aspect[v]
sec[v] = np.where(aa < 0, 8, ((aa + 22.5) % 360) // 45).astype(np.uint8)
write_geotiff_byte(os.path.join(DST, "jarilla_orientacion_sectores.tif"), sec)
 
# 10.2 Umbria / solana / este / oeste (0/1/2/3), 255 = fuera.
us = np.full(aspect.shape, 255, dtype=np.uint8)
us[np.isin(sec, [7, 0, 1])] = 0               # umbria (NO-N-NE)
us[np.isin(sec, [3, 4, 5])] = 1               # solana (SE-S-SO)
us[sec == 2] = 2                              # este
us[sec == 6] = 3                              # oeste
write_geotiff_byte(os.path.join(DST, "jarilla_umbria_solana.tif"), us)
 
# 10.3 Alineacion viento-ladera categorica (solo flujo NE): 0 no alineada, 1 alineada, 255 fuera.
def _align_mask(a, wdir):
    return (np.cos(np.radians(a - wdir)) >= UMBRAL_ALINEACION) & (a >= 0)
A_ne = _align_mask(aspect, VIENTO_PROCEDENCIA)
al = np.full(aspect.shape, 255, dtype=np.uint8)
al[mask] = 0
al[mask & A_ne] = 1
write_geotiff_byte(os.path.join(DST, "jarilla_alineacion.tif"), al)
 
# 10.4 Generadores de ficheros de estilo .qml (formato QGIS 3.x).
def qml_paletted(entries):
    rows = "\n".join(
        f'          <paletteEntry value="{val}" color="{col}" label="{lab}" alpha="255"/>'
        for val, col, lab in entries)
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="LayerConfiguration|Symbology">\n'
            '  <pipe>\n'
            '    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">\n'
            '      <colorPalette>\n' + rows + '\n      </colorPalette>\n'
            '    </rasterrenderer>\n'
            '    <resamplingStage>resamplingFilter</resamplingStage>\n'
            '  </pipe>\n</qgis>')
 
def qml_pseudocolor(items, vmin, vmax, ramp_type="DISCRETE"):
    rows = "\n".join(
        f'            <item value="{val}" color="{col}" label="{lab}" alpha="255"/>'
        for val, col, lab in items)
    return ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
            '<qgis version="3.34" styleCategories="LayerConfiguration|Symbology">\n'
            '  <pipe>\n'
            f'    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="1" '
            f'classificationMin="{vmin}" classificationMax="{vmax}" alphaBand="-1" nodataColor="">\n'
            '      <rastershader>\n'
            f'        <colorrampshader colorRampType="{ramp_type}" classificationMode="1" clip="0">\n'
            + rows + '\n        </colorrampshader>\n      </rastershader>\n'
            '    </rasterrenderer>\n'
            '    <resamplingStage>resamplingFilter</resamplingStage>\n'
            '  </pipe>\n</qgis>')
 
def _save(path, txt):
    open(path, "w").write(txt)
 
# Pendiente: rampa discreta verde -> rojo, cortes en el limite superior de clase.
_save(os.path.join(DST, "jarilla_pendiente.qml"), qml_pseudocolor(
    [(5, "#1a9850", "0-5°"), (15, "#a6d96a", "5-15°"), (30, "#fee08b", "15-30°"),
     (45, "#f46d43", "30-45°"), (90, "#a50026", ">45°")], 0, 90, "DISCRETE"))
# MDE: rampa interpolada tipo terreno.
_save(os.path.join(DST, "jarilla_mde.qml"), qml_pseudocolor(
    [(355, "#2c7bb6", "355 m"), (800, "#abd9e9", "800 m"), (1200, "#ffffbf", "1200 m"),
     (1700, "#fdae61", "1700 m"), (2369, "#a50026", "2369 m")], 355, 2369, "INTERPOLATED"))
# Orientacion: 8 sectores + llano.
_save(os.path.join(DST, "jarilla_orientacion_sectores.qml"), qml_paletted(
    [(0, "#2c7bb6", "N"), (1, "#00a6ca", "NE"), (2, "#7fcdbb", "E"), (3, "#c7e9b4", "SE"),
     (4, "#ffffbf", "S"), (5, "#fee08b", "SO"), (6, "#fdae61", "O"), (7, "#f46d43", "NO"),
     (8, "#cccccc", "Llano")]))
# Umbria / solana.
_save(os.path.join(DST, "jarilla_umbria_solana.qml"), qml_paletted(
    [(0, "#4575b4", "Umbría (NO-N-NE)"), (1, "#d73027", "Solana (SE-S-SO)"),
     (2, "#fee090", "Este"), (3, "#fdae61", "Oeste")]))
# Alineacion viento-ladera (solo flujo NE).
_save(os.path.join(DST, "jarilla_alineacion.qml"), qml_paletted(
    [(0, "#dfdfdf", "No alineada"),
     (1, "#d73027", "Alineada flujo NE (ascendente)")]))
 
 
# ============================================================================
# 11. MAPAS EN IMAGEN (PNG)
# ============================================================================
# Representaciones de cada variable para el documento, con contorno del incendio,
# leyenda y ejes en coordenadas UTM. El parametro 'extent' situa la imagen en
# coordenadas reales (left, right, bottom, top).
extent = [gx0, gx0 + nx * px, gy0 - ny * px, gy0]
slope_m = np.where(mask, slope_deg, np.nan)
aspect_m = np.where(mask, aspect, np.nan)
dem_m = np.where(mask, dem, np.nan)
 
def _base_ax(ax, title):
    ax.plot(outer[:, 0], outer[:, 1], color='k', lw=1.0, zorder=5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("X UTM (m) · ETRS89/UTM 30N"); ax.set_ylabel("Y UTM (m)")
    ax.set_aspect('equal'); ax.ticklabel_format(style='plain'); ax.tick_params(labelsize=8)
 
# 11.1 Pendiente clasificada.
cols = ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"]
labs = ["0-5°", "5-15°", "15-30°", "30-45°", ">45°"]
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(slope_m, extent=extent, origin='upper', cmap=ListedColormap(cols),
          norm=BoundaryNorm([0, 5, 15, 30, 45, 90], 5), interpolation='nearest')
_base_ax(ax, "Pendiente del terreno · Incendio de Jarilla (2025)")
ax.legend(handles=[Patch(facecolor=c, edgecolor='0.3', label=l) for c, l in zip(cols, labs)],
          title="Pendiente", loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(); plt.savefig(os.path.join(DST, "mapa_pendiente.png"), dpi=200); plt.close('all')
 
# 11.2 Orientacion en 8 sectores.
pal = ["#2c7bb6", "#00a6ca", "#7fcdbb", "#c7e9b4", "#ffffbf", "#fee08b", "#fdae61", "#f46d43"]
slbl = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
si = np.full(aspect_m.shape, np.nan); vv = ~np.isnan(aspect_m)
si[vv] = np.where(aspect_m[vv] < 0, np.nan, ((aspect_m[vv] + 22.5) % 360) // 45)
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(si, extent=extent, origin='upper', cmap=ListedColormap(pal),
          norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), 8), interpolation='nearest')
_base_ax(ax, "Orientación de laderas · Incendio de Jarilla (2025)")
ax.legend(handles=[Patch(facecolor=c, edgecolor='0.3', label=l) for c, l in zip(pal, slbl)],
          title="Orientación", loc='upper right', ncol=2, fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(); plt.savefig(os.path.join(DST, "mapa_orientacion.png"), dpi=200); plt.close('all')
 
# 11.3 Umbria / solana.
usm = np.full(aspect_m.shape, np.nan)
usm[np.isin(si, [7, 0, 1])] = 0; usm[np.isin(si, [3, 4, 5])] = 1
usm[si == 2] = 2; usm[si == 6] = 3
palus = ["#4575b4", "#d73027", "#fee090", "#fdae61"]
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(usm, extent=extent, origin='upper', cmap=ListedColormap(palus),
          norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), interpolation='nearest')
_base_ax(ax, "Umbría / Solana · Incendio de Jarilla (2025)")
ax.legend(handles=[Patch(facecolor=palus[0], label="Umbría (NO-N-NE)"),
                   Patch(facecolor=palus[1], label="Solana (SE-S-SO)"),
                   Patch(facecolor=palus[2], label="Este"), Patch(facecolor=palus[3], label="Oeste")],
          loc='upper right', fontsize=9, framealpha=0.9)
plt.tight_layout(); plt.savefig(os.path.join(DST, "mapa_umbria_solana.png"), dpi=200); plt.close('all')
 
# 11.4 Alineacion viento-ladera.
cat = np.full(aspect_m.shape, np.nan)
cat[mask & ~A_pred & ~A_ssw] = 0; cat[A_ssw] = 1; cat[A_pred] = 2
palA = ["#dfdfdf", "#fdae61", "#d73027"]
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(cat, extent=extent, origin='upper', cmap=ListedColormap(palA),
          norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation='nearest')
_base_ax(ax, "Alineación viento-ladera · Incendio de Jarilla (2025)")
ax.legend(handles=[Patch(facecolor=palA[2], label="Ladera alineada con flujo NE-N (ascendente)"),
                   Patch(facecolor=palA[1], label="Ladera alineada con flujo SSO (secundario)"),
                   Patch(facecolor=palA[0], label="No alineada")],
          loc='upper right', fontsize=8.5, framealpha=0.9)
plt.tight_layout(); plt.savefig(os.path.join(DST, "mapa_alineacion_viento_ladera.png"), dpi=200); plt.close('all')
 
# 11.5 Modelo de elevaciones con sombreado de relieve.
azr = np.radians(360 - 315 + 90); altr = np.radians(45)   # luz desde el NO, 45 deg de altura
demf = np.nan_to_num(dem_m, nan=float(np.nanmin(dem_m)))
gy, gx = np.gradient(demf, px, px)
slp = np.pi / 2 - np.arctan(np.hypot(gx, gy)); aspp = np.arctan2(-gx, gy)
hs = np.sin(altr) * np.sin(slp) + np.cos(altr) * np.cos(slp) * np.cos(azr - aspp)
hs = np.where(np.isnan(dem_m), np.nan, hs)
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(hs, extent=extent, origin='upper', cmap='gray', interpolation='nearest', alpha=0.6, zorder=1)
im = ax.imshow(dem_m, extent=extent, origin='upper', cmap='terrain', interpolation='nearest', alpha=0.55, zorder=2)
_base_ax(ax, "Modelo digital de elevaciones · Incendio de Jarilla (2025)")
cb = plt.colorbar(im, ax=ax, shrink=0.7); cb.set_label("Altitud (m s.n.m.)")
plt.tight_layout(); plt.savefig(os.path.join(DST, "mapa_mde_sombreado.png"), dpi=200); plt.close('all')
 
 
# ============================================================================
# 12. RECORTE DEL SOMBREADO (HILLSHADE) AL PERIMETRO
# ============================================================================
# El sombreado se distribuye en la misma rejilla que el MDT (5 m, mismo origen),
# por lo que se reaprovecha la mascara. Se lee por filas, se pone a 0 todo lo
# exterior al perimetro y se reescribe con el EPSG:25830. El 0 se declara como
# 'sin dato', de modo que en QGIS la zona externa queda transparente y el
# sombreado solo se ve dentro del incendio.
if os.path.exists(HILL):
    himg = Image.open(HILL); ht = himg.tag_v2
    hoffs = ht[273]
    out = np.zeros((ny, nx), np.uint8)
    with open(HILL, 'rb') as f:
        for i, r in enumerate(range(row0, row1)):
            f.seek(hoffs[r] + col0)            # hillshade en uint8 (1 byte por pixel)
            out[i] = np.frombuffer(f.read(nx), np.uint8)
    out[~mask] = 0                             # fuera del perimetro -> 0 (transparente)
    hpath = os.path.join(DST, "jarilla_hillshade.tif")
    img = Image.fromarray(out, mode='L')
    ifd = ImageFileDirectory_v2()
    ifd.tagtype[33550] = 12; ifd[33550] = (px, px, 0.0)
    ifd.tagtype[33922] = 12; ifd[33922] = (0.0, 0.0, 0.0, gx0, gy0, 0.0)
    ifd.tagtype[34735] = 3
    ifd[34735] = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 25830, 3076, 0, 1, 9001)
    ifd.tagtype[34737] = 2; ifd[34737] = "ETRS89 / UTM zone 30N|"
    ifd.tagtype[42113] = 2; ifd[42113] = "0"   # 0 = sin dato
    img.save(hpath, tiffinfo=ifd)
    with open(hpath[:-4] + ".tfw", "w") as fw:
        fw.write(f"{px}\n0.0\n0.0\n{-px}\n{gx0 + px/2}\n{gy0 - px/2}\n")
    open(hpath[:-4] + ".prj", "w").write(PRJWKT)
 
print("Procesamiento topografico completado.")
print(f"  Superficie analizada: {npix * pix_ha:,.1f} ha")
print(f"  Altitud: {elev_stats['min']:.0f}-{elev_stats['max']:.0f} m "
      f"(media {elev_stats['mean']:.0f} m)")
print(f"  Pendiente media: {slope_summary['mean']:.1f} deg")
print(f"  Productos escritos en: {DST}")

