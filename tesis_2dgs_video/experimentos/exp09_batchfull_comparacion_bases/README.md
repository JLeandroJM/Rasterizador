# exp09 — Batch-full + comparacion de bases (chebyshev vs monomial)

## Cambio metodologico respecto a exp08

| Aspecto | exp08 | exp09 |
|---|---|---|
| Entrenamiento | incremental (frame a frame) con replay | **epoch-based "batch full" via gradient accumulation** |
| Bases | solo Chebyshev | **dos sub-experimentos: Chebyshev y monomial** |
| Hardware | MPS (Mac M2) | **CUDA (Windows + RTX 4050)** |
| Metricas compresion | bytes del modelo | **+ comparacion AVIF frame-por-frame** |

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
├── visualizar_trayectorias.py     # 5 figuras (trayectorias, heatmap, ...)
├── metricas_calidad.py            # PSNR/SSIM/LPIPS por frame
├── metricas_compresion.py         # tamano modelo + AVIF q80/q95
├── correr.py                      # entry point por config
└── comparar_bases.py              # cruza las dos corridas
```

## Como correr (Windows + CUDA)

```bash
# instalar deps
pip install -r requirements.txt

# corrida Chebyshev (sub-experimento 1)
python correr.py --config configs/config_chebyshev.json

# corrida monomial (sub-experimento 2)
python correr.py --config configs/config_monomial.json

# comparacion final
python comparar_bases.py --clip drifting_circles
```

Cada corrida produce `resultados/exp09/<nombre_experimento>/<clip>/` con:
- `checkpoint_final.pt`, `modelo_pruneado.pt`
- `metricas_calidad.json`, `metricas_compresion.json`
- `loss_curve.png`, `log_entrenamiento.csv`
- `trayectorias.png`, `heatmap_opacity_temporal.png`,
  `evolucion_parametros.png`, `coeficientes_magnitudes.png`
- `reconstruccion_vs_original.gif`
- `frames_rasterizados/`, `frames_originales_avif_q*/`, `frames_rasterizados_avif_q*/`

La comparacion produce `resultados/exp09/comparacion_<clip>/` con:
- `tabla_comparativa.csv`
- `bars_calidad.png`, `curvas_loss_superpuestas.png`, `rd_tamano_vs_ssim.png`
- `comparacion.md`

## Decisiones documentadas (todas marcadas con `# DECISION:` en el codigo)

1. **`a_0` vs `a_high` en `nn.Parameter` separados**
   (`modelo.py:GaussianasPolinomial2D`). Razones:
   - lr distinto por orden via `param_groups` (a_0 base, a_high base*0.1).
   - smoothness ignora a_0 automaticamente (k=0 -> peso k²=0).
   - inicializacion distinta natural (a_0 con valor razonable, a_high en 0).

2. **Una matriz de evaluacion por grado distinto**, no una grande con slicing.
   Solo hay 2 grados (40 y 20) -> 2 matrices. `bases.py:construir_matriz`.

3. **Batch full = gradient accumulation con loop sobre frames**
   (`trainer.py`). Memoria pico = 1 frame en lugar de N frames. Equivalente
   matematicamente al batch full porque el gradiente es lineal. Si la GPU es
   grande, se puede subir `sub_batch_frames` en el config.

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

## Estimacion de tiempo

Para un clip de 90 frames a 256x256 con N=1000, grados q_mu=40/q_otros=20,
`M_iter_por_frame` no existe en este experimento (batch full), `n_epochs=300`:

- Tiempo por epoch en CPU: ~3-5 min por epoch (90 frames con render y
  backward cada uno).
- En CUDA RTX 4050: estimacion 30-60 seg por epoch -> 2.5-5h total.

Al inicio del entrenamiento el trainer imprime `eta=XX.X min` recalculado
cada checkpoint para que puedas decidir si abortar.

## Validacion local antes de la corrida real

Para validar el pipeline rapidamente (sin gastar la GPU horas):

```bash
# editar configs/config_chebyshev.json con:
#   "clip": "drifting_circles",
#   "max_frames": 15,
#   "n_gaussianas_inicial": 100,
#   "n_epochs": 30,
#   "device": "cuda"        (o "cpu" para test sin GPU)

python correr.py --config configs/config_chebyshev.json
```

Esto deberia correr en pocos minutos y validar que el pipeline esta intacto
antes de lanzar la corrida completa.
