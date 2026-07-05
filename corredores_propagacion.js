/***********************************************************************************
 * TFG Jarilla 2025 — Corredores de propagacion y validacion con severidad (GEE)
 *
 * Autor: Luis Hernández  |  Caso: Incendio de Jarilla (ignicion 12-13 ago 2025)
 *
 * QUE HACE ESTE SCRIPT (apartado 5.4 del TFG)
 *   1. Recupera el combustible (NDVI) y la severidad (dNBR) como en el script anterior.
 *   2. Calcula pendiente y orientacion del terreno a partir de un MDE (Copernicus GLO-30).
 *   3. Define la ALINEACION viento-ladera con el viento dominante del episodio (NE, 45 deg).
 *   4. Delimita los CORREDORES de propagacion = pendiente >= 15 deg  Y  alineacion alta
 *      Y  combustible continuo (NDVI por encima de su mediana).
 *   5. VALIDA: compara la severidad DENTRO y FUERA de los corredores (dNBR medio y % de
 *      superficie de severidad alta). Si el modelo es bueno, la severidad alta debe
 *      concentrarse dentro de los corredores.
 *   6. Imprime en consola los numeros para redactar resultados y exporta el mapa de corredores.
 *
 * COMO USARLO
 *   - Pegar este script en un editor NUEVO de Earth Engine (deja el otro como esta).
 *   - Tiene que estar subido el perímetro en shapefile.
 *   - Analizar los datos de la consola.
 ***********************************************************************************/
 
// ============================ 0. PARAMETROS ============================
var PERIMETRO_ASSET = 'projects/iberia2025/assets/perimetro_incendio';   // perimetro ya subido
var aoi = ee.FeatureCollection(PERIMETRO_ASSET).geometry();
 
// Viento dominante del episodio: PROCEDENCIA del NE ~ 45 grados.
var VIENTO_PROCEDENCIA = 45;     // grados (de donde viene el viento)
var UMBRAL_PENDIENTE   = 15;     // grados: laderas que aceleran el fuego
var UMBRAL_ALINEACION  = 0.5;    // coseno >= 0.5  => dentro de +-60 deg de alineacion perfecta
 
// Ventanas temporales (iguales al script de indices)
var PRE_INI='2025-07-15', PRE_FIN='2025-08-11';
var POST_INI='2025-08-25', POST_FIN='2025-09-30';
var MAX_NUBES=60, CS_UMBRAL=0.60;
var EPSG='EPSG:25830', ESCALA=20, CARPETA_DRIVE='TFG_Jarilla_Sentinel2';
 
Map.centerObject(aoi, 11);
 
// ====================== 1. SENTINEL-2: NDVI y dNBR ======================
function cargarS2(ini, fin){
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(aoi).filterDate(ini, fin)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_NUBES));
  var csp = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');
  col = col.linkCollection(csp, ['cs']).map(function(img){
    return img.updateMask(img.select('cs').gte(CS_UMBRAL))
              .divide(10000).copyProperties(img, img.propertyNames());
  });
  return col;
}
var pre  = cargarS2(PRE_INI, PRE_FIN).median().clip(aoi);
var post = cargarS2(POST_INI, POST_FIN).median().clip(aoi);
 
var NDVI_pre = pre.normalizedDifference(['B8','B4']).rename('NDVI');
var NBR_pre  = pre.normalizedDifference(['B8','B12']).rename('NBR');
var NBR_post = post.normalizedDifference(['B8','B12']).rename('NBR');
var dNBR = NBR_pre.subtract(NBR_post).multiply(1000).rename('dNBR');
 
var severidad = dNBR
  .where(dNBR.lt(-100),1).where(dNBR.gte(-100).and(dNBR.lt(100)),2)
  .where(dNBR.gte(100).and(dNBR.lt(270)),3).where(dNBR.gte(270).and(dNBR.lt(440)),4)
  .where(dNBR.gte(440).and(dNBR.lt(660)),5).where(dNBR.gte(660),6).rename('severidad');
 
// ====================== 2. TOPOGRAFIA: pendiente y alineacion ======================
// MDE05 subido previamente
var dem   = ee.Image('projects/iberia2025/assets/mdt05_jarilla');
var slope = ee.Terrain.slope(dem).clip(aoi).rename('pendiente');    // grados
var aspect= ee.Terrain.aspect(dem).clip(aoi).rename('orientacion'); // grados (ladera de bajada)
 
// Alineacion viento-ladera: maxima cuando la ladera asciende hacia donde sopla el viento.
// coseno(orientacion - procedencia_viento); = 1 si la ladera mira hacia el NE (asciende al SO).
var alineacion = aspect.subtract(VIENTO_PROCEDENCIA).multiply(Math.PI/180).cos().rename('alineacion');
 
// Comprobacion rapida: la pendiente media debe ser de varios grados (no ~0).
print('Pendiente media del terreno (grados):',
  slope.reduceRegion({reducer:ee.Reducer.mean(), geometry:aoi, scale:ESCALA, maxPixels:1e13, bestEffort:true}).get('pendiente'));
 
// ====================== 3. CORREDORES DE PROPAGACION ======================
var ndviMediana = ee.Number(NDVI_pre.reduceRegion({
  reducer: ee.Reducer.median(), geometry: aoi, scale: ESCALA, maxPixels:1e13, bestEffort:true
}).get('NDVI'));
print('NDVI mediana (umbral de combustible):', ndviMediana);
 
var combustible = NDVI_pre.gte(ee.Image.constant(ndviMediana));
var corredor = slope.gte(UMBRAL_PENDIENTE)
                 .and(alineacion.gte(UMBRAL_ALINEACION))
                 .and(combustible)
                 .rename('corredor');
 
// ====================== 4. VISUALIZACION ======================
Map.addLayer(severidad, {min:1,max:6,palette:['1b9e77','c2c2c2','ffffb2','fecc5c','fd8d3c','e31a1c']}, 'Severidad');
Map.addLayer(alineacion, {min:-1,max:1,palette:['blue','white','red']}, 'Alineacion viento-ladera', false);
Map.addLayer(corredor.selfMask(), {palette:['black']}, 'Corredores');
 
// ====================== 5. VALIDACION: severidad dentro vs fuera ======================
var ha = ee.Image.pixelArea().divide(10000);
function sumaHa(mask){
  return ha.updateMask(mask).reduceRegion({reducer:ee.Reducer.sum(), geometry:aoi, scale:ESCALA, maxPixels:1e13, bestEffort:true}).get('area');
}
function mediaDNBR(mask){
  return dNBR.updateMask(mask).reduceRegion({reducer:ee.Reducer.mean(), geometry:aoi, scale:ESCALA, maxPixels:1e13, bestEffort:true}).get('dNBR');
}
var altaSev = severidad.gte(5);            // clases 5 y 6 (moderada-alta + alta)
var fuera = corredor.not();
 
print('=============== RESULTADOS PARA 5.4 ===============');
print('Superficie de CORREDORES (ha):', sumaHa(corredor));
print('dNBR medio DENTRO de corredores:', mediaDNBR(corredor));
print('dNBR medio FUERA de corredores:', mediaDNBR(fuera));
print('Superficie severidad ALTA (clases 5-6) DENTRO de corredores (ha):', sumaHa(corredor.and(altaSev)));
print('Superficie severidad ALTA (clases 5-6) FUERA de corredores (ha):', sumaHa(fuera.and(altaSev)));
print('% del corredor que es severidad alta:',
      ee.Number(sumaHa(corredor.and(altaSev))).divide(sumaHa(corredor)).multiply(100));
print('% del area NO-corredor que es severidad alta:',
      ee.Number(sumaHa(fuera.and(altaSev))).divide(sumaHa(fuera)).multiply(100));
 
// ====================== 6. EXPORTAR MAPA DE CORREDORES ======================
Export.image.toDrive({
  image: corredor.toByte(), description:'jarilla_corredores', folder:CARPETA_DRIVE,
  fileNamePrefix:'jarilla_corredores', region:aoi, scale:ESCALA, crs:EPSG, maxPixels:1e13
});
print('>>> Revisar los numeros del bloque "RESULTADOS PARA 5.4" y exportar el mapa (pestaña Tasks).');
