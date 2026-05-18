# exp09 — Batch-full + comparacion de bases (chebyshev vs monomial)

## Cambio metodologico respecto a exp08

| Aspecto | exp08 | exp09 |
|---|---|---|
| Entrenamiento | incremental (frame a frame) con replay | **epoch-based "batch full" via gradient accumulation** |
| Bases | solo Chebyshev | **dos sub-experimentos: Chebyshev y monomial** |
| Hardware | MPS (Mac M2) | **CUDA (Windows + RTX 4050)** |
| Metricas compresion | bytes del modelo | **+ comparacion AVIF frame-por-frame** |
| Input | clips sinteticos solo PNG | **mp4 -> PNGs (automatico) o PNGs directos** |

## Idea central

1. **Batch full sin OOM**: en cada step del optimizer, el loss se calcula sobre
   TODOS los frames del clip. Como construir el grafo completo de N frames a la
   vez no entra en una 4050, lo implementamos via **gradient accumulation**:
   un `backward()` por frame, sin retener el grafo entre frames. Matematicamente
   equivalente al batch full (el gradiente es lineal), memoria pico = 1 frame.
2. **Comparacion empirica de bases**: validar que la base de Chebyshev produce
   mejor convergencia que la monomial para grados altos (q=40), tal como
   predice la teoria (mejor condicionamiento, no Runge).
3. **Compresion vs codecs de imagen**: ademas del tamano del modelo, codificar
   cada frame con AVIF (calidades 80 y 95) y reportar bytes por frame. Compara
   "1 video con N modelos GS por frame" vs "1 modelo GS para todo el video".

## Archivos

```
exp09_batchfull_comparacion_bases/
├── README.md
├── requirements.txt
├── configs/
│   ├── config_chebyshev.json
│   └── config_monomial.json
├── bases.py                       # T_k(t), t^k, recurrencia, dispatcher
├── modelo.py                      # GaussianasPolinomial2D (agnostico a la base)
├── rasterizador.py                # vectorizado + comentarios de memoria
├── perdidas.py                    # render_frame, render_batch, smoothness
├── optimizador.py                 # Adam con 12 param_groups
├── trainer.py                     # epoch-based + gradient accumulation
├── pruning_post.py                # lottery ticket por contribucion de opacity
├── preparar_video.py              # mp4 -> PNGs via ffmpeg (auto)
├── visualizar_trayectorias.py     # 5 figuras (trayectorias, heatmap, ...)
├── metricas_calidad.py            # PSNR/SSIM/LPIPS por frame
├── metricas_compresion.py         # tamano modelo + AVIF q80/q95
├── correr.py                      # entry point por config
└── comparar_bases.py              # cruza las dos corridas
```

Y en la raiz de `tesis_2dgs_video/`:

```
video/                  # poner aqui tu mp4 descargado
clips/                  # carpetas con frames (auto-pobladas si usas video_mp4)
resultados/exp09/       # outputs por sub-experimento
```


## Como usar tu propio video

1. **Descarga el video** que quieras procesar y guardalo en
   `tesis_2dgs_video/video/`. Por ejemplo: `tesis_2dgs_video/video/mi_video.mp4`.

2. **Edita el config** (`configs/config_chebyshev.json` o `config_monomial.json`)
   y rellena los campos del video:

   ```json
   {
       ...
       "video_mp4": "video/mi_video.mp4",
       "n_frames_extraer": 90,
       "fps_extraccion": null,
       "resolucion_extraccion": [256, 256],
       "forzar_extraccion": false,

       "clip": "mi_video",
       ...
   }
   ```

   - `video_mp4`: ruta al mp4 (relativa a `tesis_2dgs_video/` o absoluta).
   - `n_frames_extraer`: cuantos frames extraer (por ejemplo `90` = 3 segundos
     a 30 fps). `null` = todos los que el video produzca.
   - `fps_extraccion`: si querés cambiar el frame rate (ej. `15` para
     submuestrear). `null` = mantener el fps nativo.
   - `resolucion_extraccion`: `[H, W]`. El video se recorta cuadrado centrado al
     menor lado y despues se escala a esta resolucion.
   - `forzar_extraccion`: si `true`, re-extrae aunque ya exista la carpeta;
     `false` = usa cache (si `clips/<clip>/` ya tiene frames, no llama a ffmpeg).
   - `clip`: nombre de la carpeta destino dentro de `clips/`. Conviene
     ponerle el mismo nombre que el video (sin extension) para tenerlo prolijo.

3. **Corre normal**:

   ```bash
   python correr.py --config configs/config_chebyshev.json
   ```

   Si `video_mp4` esta definido, el script primero invoca `ffmpeg` para
   poblar `clips/<clip>/` y despues sigue con el entrenamiento como si fuera
   una secuencia PNG cualquiera. **No hay paso manual**: el mp4 -> PNGs ocurre
   automaticamente al inicio.

**Si no querés usar video**: dejá `video_mp4: null`. El script asume que ya
existe `clips/<clip>/` con los PNGs (por ejemplo los que genera
`tools/generar_clips_sinteticos.py`).

**Requisito**: necesitas `ffmpeg` en el PATH.
- macOS: `brew install ffmpeg`
- Windows: descarga de ffmpeg.org y agrega al PATH del sistema
- Linux: `apt install ffmpeg` o `dnf install ffmpeg`


## Como correr todo (Windows + CUDA)

```bash
# instalar deps
pip install -r requirements.txt

# (opcional) preparar mp4 -> el config se encarga, ver seccion anterior

# corrida Chebyshev (sub-experimento 1)
python correr.py --config configs/config_chebyshev.json

# corrida monomial (sub-experimento 2)
python correr.py --config configs/config_monomial.json

# comparacion final entre las dos bases
python comparar_bases.py --clip mi_video
```

Cada corrida produce `resultados/exp09/<nombre_experimento>/<clip>/` con:
- `checkpoint_final.pt`, `modelo_pruneado.pt`
- `metricas_calidad.json`, `metricas_compresion.json`
- `loss_curve.png`, `log_entrenamiento.csv`
- `trayectorias.png`, `heatmap_opacity_temporal.png`,
  `evolucion_parametros.png`, `coeficientes_magnitudes.png`
- `reconstruccion_vs_original.gif`
- `frames_rasterizados/`, `frames_originales_avif_q*/`,
  `frames_rasterizados_avif_q*/`

La comparacion produce `resultados/exp09/comparacion_<clip>/` con:
- `tabla_comparativa.csv`
- `bars_calidad.png`, `curvas_loss_superpuestas.png`, `rd_tamano_vs_ssim.png`
- `comparacion.md`


## Parametros del config — guia completa

**Todos los parametros se editan en `configs/config_chebyshev.json` y
`configs/config_monomial.json`.** Los dos archivos tienen la misma estructura;
la unica diferencia "intrinseca" es `base`.

### 1. Identidad del experimento

| Parametro | Tipo | Descripcion | Cuando cambiarlo |
|---|---|---|---|
| `nombre_experimento` | str | Nombre del sub-experimento. Se usa para nombrar la carpeta de resultados. | Si haces variantes del mismo experimento (ej. `exp09_chebyshev_q20`). |
| `base` | `"chebyshev"` o `"monomial"` | Que base de polinomios usar. | NO cambiar dentro de un config — esa decision define al sub-experimento. |
| `clip` | str | Nombre de la carpeta en `clips/`. | Cambiar cuando uses otro clip. |

### 2. Extraccion de frames desde mp4

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `video_mp4` | str / null | `null` | Ruta al mp4. Si `null`, NO se hace extraccion (se asume que `clips/<clip>/` ya tiene PNGs). |
| `n_frames_extraer` | int / null | `null` | Cuantos frames cortar del video. `null` = todos. **Bajalo para experimentos rapidos** (ej. 30-60). |
| `fps_extraccion` | int / null | `null` | Submuestreo temporal. `null` = fps nativo. Ej. `15` para video largo a la mitad. |
| `resolucion_extraccion` | `[H, W]` | `[256, 256]` | Resolucion de salida. Hace crop cuadrado + scale. `[128, 128]` para iterar rapido; `[256, 256]` para corrida real. |
| `forzar_extraccion` | bool | `false` | Si `true`, sobreescribe la carpeta destino. Si `false`, usa cache (no re-llama a ffmpeg). |

### 3. Modelo (capacidad)

| Parametro | Tipo | Default | Descripcion | Cuando cambiarlo |
|---|---|---|---|---|
| `n_gaussianas_inicial` | int | `1000` | N inicial (antes del pruning post). | Subir si la calidad esta limitada por capacidad. Bajar para experimentos rapidos. |
| `grados.mu` | int | `40` | Grado del polinomio para la posicion. | Bajar si el contenido es estatico; subir si hay movimiento muy rapido. |
| `grados.opacity` | int | `40` | Grado para la opacidad temporal. | Subir si gaussianas aparecen/desaparecen muchas veces en el clip. |
| `grados.color` | int | `20` | Grado para color. | El color cambia menos abruptamente que mu/opacity, por eso 20 alcanza. |
| `grados.scale` | int | `20` | Grado para escala. | Cambiar si las gaussianas crecen/decrecen rapido. |
| `grados.theta` | int | `20` | Grado para rotacion. | Subir si hay rotaciones rapidas. |
| `grados.depth` | int | `20` | Grado para profundidad de orden front-to-back. | Casi siempre lo dejas. |

### 4. Optimizador (learning rates por parametro × orden)

Hay 12 lrs (6 parametros × {a_0, a_high}). El sufijo `_a0` controla el
coeficiente constante (movimiento "promedio"); el sufijo `_high` controla los
coeficientes de orden 1 a q (variacion temporal). Por defecto `_a0` es 10×
mayor que `_high` para que primero converja la "posicion media" antes que la
"trayectoria fina".

| Parametro | Default | Significado |
|---|---|---|
| `lrs.mu_a0`, `lrs.mu_high` | `1e-3`, `1e-4` | Velocidad de aprendizaje de la posicion. |
| `lrs.opacity_a0`, `lrs.opacity_high` | `5e-2`, `5e-3` | Aprendizaje rapido porque la opacidad usa sigmoid. |
| `lrs.color_a0`, `lrs.color_high` | `1e-2`, `1e-3` | Color tambien con sigmoid. |
| `lrs.scale_a0`, `lrs.scale_high` | `5e-3`, `5e-4` | Conservador porque scale usa exp (sensible a explosion). |
| `lrs.theta_a0`, `lrs.theta_high` | `5e-3`, `5e-4` | Rotacion suele necesitar poco ajuste. |
| `lrs.depth_a0`, `lrs.depth_high` | `5e-3`, `5e-4` | Idem. |

**Cuando subir todos los lrs**: el loss baja muy lento al inicio.
**Cuando bajar**: el loss oscila o explota (NaN).

### 5. Training loop

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `n_epochs` | int | `300` | Steps del optimizer. Cada epoch ve todo el clip por gradient accumulation. **Para validar rapido bajalo a 30-50**, para corrida real usa 300+. |
| `lambda_dssim` | float | `0.2` | Peso de DSSIM en el loss render: `(1-lambda)*L1 + lambda*DSSIM`. Subir = mas enfasis en estructura; bajar = mas en colores puros. |
| `sub_batch_frames` | int | `1` | Cuantos frames procesar por backward (gradient accumulation). `1` = memoria minima; subir si tu GPU es grande (≥12 GB) para mejor throughput. |
| `seed` | int | `42` | Semilla para inicializacion reproducible. |

### 6. Regularizacion (smoothness)

Penaliza coeficientes de orden alto para que el modelo prefiera funciones
suaves. Formula: `sum_k (a_k)^2 * k^2`. El factor `k^2` hace que los coefs
de grado alto cuesten cuadraticamente mas.

| Parametro | Default | Cuando cambiarlo |
|---|---|---|
| `beta_smoothness` | `1e-5` | Subir si ves "flicker" entre frames (oscilaciones espurias) o overfitting a frames individuales. Bajar si la reconstruccion se ve "atrasada" (no sigue movimientos rapidos). |
| `pesos_smoothness.mu` | `1.0` | Peso relativo. |
| `pesos_smoothness.opacity` | `0.5` | Menos regularizacion: queremos que la opacidad cambie rapidamente para encender/apagar. |
| `pesos_smoothness.color` | `2.0` | Mas regularizacion: el color del objeto suele cambiar suave. |
| `pesos_smoothness.scale` | `2.0` | Idem. |
| `pesos_smoothness.theta` | `2.0` | Idem. |
| `pesos_smoothness.depth` | `2.0` | Idem. |

### 7. Inicializacion

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `escala_inicial_px` | float | `5.0` | Tamano inicial (en pixeles) de cada gaussiana. `log(esto)` va al `scale_a0`. |

### 8. Pruning post-training (lottery ticket)

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `umbral_pruning_post` | float | `0.05` | Una gaussiana se elimina si `max_t sigma(opacity(t)) < umbral`. Subir = pruning mas agresivo (modelo mas chico, riesgo de perder detalle). |

### 9. Checkpoints, hardware y carga

| Parametro | Tipo | Default | Descripcion |
|---|---|---|---|
| `checkpoint_cada_n_epochs` | int | `50` | Frecuencia de guardado de `checkpoint_epoch*.pt` y de logging detallado (PSNR, render del primer y ultimo frame). |
| `max_frames` | int / null | `null` | Limite duro sobre la longitud del clip cargado (post-extraccion). Util si extraiste 200 frames pero solo querés entrenar con los primeros 90. |
| `calidades_avif` | `[int, ...]` | `[80, 95]` | Calidades AVIF a probar en el reporte de compresion. |
| `device` | `"cuda"` / `"mps"` / `"cpu"` | `"cuda"` | Donde correr. **No cambiar a `"cuda"` si no tenés GPU NVIDIA con CUDA.** |


## Patrones de uso comunes

### "Quiero validar el pipeline lo mas rapido posible"

```json
{
    "video_mp4": "video/mi_video.mp4",
    "n_frames_extraer": 20,
    "resolucion_extraccion": [128, 128],
    "n_gaussianas_inicial": 100,
    "n_epochs": 30,
    "grados": {"mu": 10, "opacity": 10, "color": 6, "scale": 6, "theta": 6, "depth": 6},
    "checkpoint_cada_n_epochs": 10
}
```
Tiempo: 1-2 min en GPU.

### "Quiero corrida real con buena calidad"

Dejar los defaults (`n_epochs: 300`, `n_gaussianas_inicial: 1000`,
`grados.mu: 40`, etc.).

### "El loss diverge / aparece NaN"

Bajar los lrs (especialmente `mu_a0`, `scale_a0`, `opacity_a0`) a la mitad.
Si persiste, subir `beta_smoothness` un orden de magnitud.

### "El loss no baja"

Subir `lrs.*_a0`. Verificar que `n_gaussianas_inicial` sea suficiente para el
contenido. Verificar que `n_epochs` no sea ridiculamente bajo.

### "El video tiene movimiento muy rapido y el modelo no lo sigue"

Subir `grados.mu` a 60 o mas. Considerar bajar `beta_smoothness` y/o el
`pesos_smoothness.mu`.

### "El video se ve casi estatico pero el modelo "tiembla""

Subir `beta_smoothness` (1e-5 -> 1e-4 o 1e-3). El modelo esta usando coefs
de orden alto innecesariamente.


## Decisiones documentadas (marcadas con `# DECISION:` en el codigo)

1. **`a_0` vs `a_high` en `nn.Parameter` separados**
   (`modelo.py:GaussianasPolinomial2D`). Razones:
   - lr distinto por orden via `param_groups` (a_0 base, a_high base*0.1).
   - smoothness ignora a_0 automaticamente (k=0 -> peso k²=0).
   - inicializacion distinta natural (a_0 con valor razonable, a_high en 0).

2. **Una matriz de evaluacion por grado distinto**, no una grande con
   slicing. Solo hay 2 grados (40 y 20) -> 2 matrices.
   `bases.py:construir_matriz`.

3. **Batch full = gradient accumulation con loop sobre frames**
   (`trainer.py`). Memoria pico = 1 frame en lugar de N frames. Equivalente
   matematicamente al batch full porque el gradiente es lineal. Si la GPU
   es grande, se puede subir `sub_batch_frames` en el config.

4. **AVIF a calidades 80 y 95** por defecto. q=80 ya es practicamente
   indistinguible visualmente; q=95 es upper bound de calidad. Cualquier
   calidad de la lista en `config["calidades_avif"]` se procesa.

5. **5 gaussianas "representativas"** en `evolucion_parametros.png`: top-5
   por opacity promedio sobre el clip. Son las que mas contribuyen
   visualmente, las mas relevantes para mostrar.

6. **Recurrencias en CPU float64 + transferencia al device float32**
   (`bases.py`). Mas estable numericamente para grado 40 que hacer la
   recurrencia directo en float32 en GPU. El costo de construir la matriz
   es despreciable (una vez por entrenamiento).

7. **Cache implicito en `preparar_video.py`**: si `clips/<clip>/` ya tiene
   los frames esperados y `forzar_extraccion=false`, NO se invoca ffmpeg.
   Util para iterar rapido cambiando hiperparametros sobre el mismo video.


## Estimacion de tiempo

Para un clip de 90 frames a 256x256 con N=1000, grados q_mu=40/q_otros=20,
`n_epochs=300`:

- Tiempo por epoch en CPU: ~3-5 min por epoch (90 frames con render y
  backward cada uno).
- En CUDA RTX 4050: estimacion 30-60 seg por epoch -> 2.5-5h total.

Al inicio del entrenamiento el trainer imprime `eta=XX.X min` recalculado
cada checkpoint para que puedas decidir si abortar.
