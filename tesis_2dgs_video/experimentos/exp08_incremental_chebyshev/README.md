# exp08 — Incremental + Chebyshev (giro metodologico)

## Cambio respecto a exp01/02

| Aspecto | exp01/02 | exp08 |
|---|---|---|
| Base de funciones | Monomios `t^k` | **Chebyshev `T_k(t)`** |
| Grado max | 3-5 | **40 (mu, opacity), 20 (resto)** |
| Parametros temporales | solo mu (exp01) o mu+opacity (exp02) | **TODOS** los parametros |
| Entrenamiento | sampling random | **incremental con replay completo** |
| Regularizacion | clamp escala | **smoothness `sum a_k^2 * k^2`** |
| Densificacion | clone/split/prune | **lottery ticket: N fijo + prune final** |

## Idea de tesis

**El tamano del modelo es independiente del numero de frames.**
Un video de 100 frames y uno de 10000 frames pesan lo mismo. Solo la
calidad puede variar segun complejidad del contenido.

## Archivos

```
exp08_incremental_chebyshev/
├── chebyshev.py              base T_k(t), recurrencia, evaluacion
├── modelo.py                 GaussianasChebyshev2D (a_0 vs a_high separados)
├── rasterizador.py           vectorizado y diferenciable
├── perdidas.py               loss_render + loss_smoothness
├── optimizador.py            Adam con param_groups por (param, orden)
├── trainer.py                loop incremental con replay (completo/parcial)
├── pruning_post.py           lottery ticket post-training
├── visualizar_trayectorias.py  4 figuras clave
├── metricas.py               PSNR/SSIM/LPIPS/tamano/ratio
├── correr.py                 entry point
├── config.json
└── README.md
```

## Como correr

```bash
cd experimentos/exp08_incremental_chebyshev
python correr.py
```

Lee `config.json`. Para validar primero con un clip corto sin esperar horas:

```json
"clip": "drifting_circles",
"max_frames": 20,
"M_iter_por_frame": 30,
"n_gaussianas_inicial": 200
```

## Decisiones documentadas

- **Base Chebyshev** (no monomial): evita el fenomeno de Runge con grado 40.
  Ver `chebyshev.py` para la matematica completa.
- **a_0 vs a_high en `nn.Parameter` separados**: simplifica los param_groups
  del optimizador (lr distinto por orden) y la penalizacion de smoothness
  (k=0 queda automaticamente fuera). Ver `modelo.py:GaussianasChebyshev2D`.
- **Matrices de Chebyshev por grado distinto** (no una sola grande con slicing):
  dos matrices (grado 40 y grado 20) que se reusan. Ver `chebyshev.py`.
- **Replay completo por defecto**: `modo_replay="completo"` en config. Activar
  `"parcial"` con `K_replay=10` si el frame k=N-1 toma demasiado tiempo.
- **Clamp de scale en evaluacion**: aplicado en `modelo.evaluar_en_frame()`
  con rango log[0.5, max(H,W)]. Evita NaN cuando coefs altos se descontrolan.
- **Pruning post-training, no durante**: lottery ticket con `umbral=0.05`
  sobre `max_t sigma(opacity(t))`.

## Test de no-olvido (auto-diagnostico)

Durante el entrenamiento, el trainer imprime `PSNR(f=0)` despues de cada
intervalo de frames. Si decrece notablemente conforme avanza `k`, hay
**olvido catastrofico** y algo del replay no funciona. La curva se guarda
en `psnr_frame0_no_olvido.png`.

## Outputs

- `loss.png` — loss concatenado sobre todas las iteraciones
- `psnr_frame0_no_olvido.png` — PSNR(f=0) vs k
- `tiempo_por_frame.png` — costo computacional por frame
- `trayectorias.png` — frame 0 + trayectorias mu_i(t) superpuestas
- `heatmap_opacity_temporal.png` — sigma(opacity_i(t_j)) por gaussiana
- `evolucion_parametros.png` — 5 gaussianas representativas en detalle
- `reconstruccion_vs_original.gif` — original | reconstruido | diff x5
- `checkpoint_k*.pt` — checkpoints intermedios
- `checkpoint_final.pt` — modelo final post-pruning
- `metricas.json`, `metricas_resumen.csv`

## Estimacion de tiempo

Para un clip de 90 frames a 256x256 con `M_iter=200`, N=1000 y replay
completo, el tiempo crece linealmente con `k`. El total estimado al
comienzo de la corrida sale impreso (`eta=...min`).

Para iteraciones de desarrollo, usar el config recortado de arriba.
