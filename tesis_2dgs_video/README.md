# Tesis 2DGS Video

Compresion de video con Gaussian Splatting 2D temporal.

## Idea central

Cada parametro de cada gaussiana (centro, escala, theta, opacidad, color) que sea
"tiempo-dependiente" se reemplaza por un polinomio en `t`:

    p(t) = a_0 + a_1*t + a_2*t^2 + ... + a_q*t^q

donde `t = frame_idx / (num_frames - 1)` esta normalizado a `[0, 1]`.
Los coeficientes `[a_0, ..., a_q]` son los parametros aprendidos (`nn.Parameter`).
En el forward pass del frame `t`, se evalua el polinomio y se obtienen los
parametros de cada gaussiana en ese instante, despues se rasteriza como en
2DGS estatico.

## Regimen de gaussianas

Numero fijo `N`. Las gaussianas "aparecen y desaparecen" haciendo que
`opacidad(t)` sea tiempo-dependiente (cuando `opacidad(t) ~ 0` la gaussiana
esta apagada en ese frame). NO hay densificacion durante training temporal.

## Estructura

```
tesis_2dgs_video/
├── clips/                  # 5 clips inmutables como PNG sequences
├── tools/                  # generacion de clips, medicion, comparador
├── experimentos/           # 8 experimentos, cada uno autocontenido
└── resultados/             # outputs de cada exp sobre cada clip
```

## Orden recomendado de ejecucion

1. `tools/generar_clips_sinteticos.py`  -> genera los 5 clips sinteticos
2. `experimentos/exp00_baseline_estatico/train.py`
3. `experimentos/exp01_solo_posicion_temporal/train.py`
4. `experimentos/exp02_posicion_opacidad_temporal/train.py`
5. `experimentos/exp03_full_temporal/train.py`
6. `experimentos/exp04_grado_polinomio/train.py`   (barrido q)
7. `experimentos/exp05_numero_gaussianas/train.py`  (barrido N)
8. `experimentos/exp06_cuantizacion/eval.py`        (reusa modelos de exp05)
9. `experimentos/exp07_codecs_clasicos/run.py`      (independiente)
10. `tools/comparar_experimentos.py`                (tabla final)

## Decisiones de diseno

- **Codigo duplicado a proposito** entre experimentos. La independencia es mas
  valiosa que la no-duplicacion.
- **SSIM propio** (sin pytorch-msssim) para evitar dependencia.
- **LPIPS opcional**: si `lpips` no esta instalado, se reporta `None`.
- **Polinomios en base monomial** `(1, t, t^2, ...)`. Alternativa Chebyshev
  documentada pero no implementada por simplicidad pedagogica.

## Dispositivo

PyTorch usa MPS si esta disponible, sino CPU. `torch.manual_seed(42)` en cada script.
