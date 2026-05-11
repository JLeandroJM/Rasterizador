# exp04 — Barrido del grado del polinomio

## Hipotesis

Usando la configuracion ganadora de exp01/02/03 (definida por `config["base"]`),
barrer el grado `q ∈ {1, 2, 3, 5, 8}` y medir como cambia PSNR/SSIM/LPIPS
vs el tamano del modelo.

Esperamos un "codo" donde el PSNR satura: a partir de cierto `q`, agregar mas
grado solo aumenta tamano sin mejorar calidad.

## Decision

`# DECISION:` por defecto este experimento usa la configuracion de exp02
(`base = "exp02_posicion_opacidad_temporal"`), porque exp02 es el modelo
mas conceptualmente distintivo (opacidad temporal = "encender/apagar"). Cambia
`base` en `config.json` cuando tengas tus resultados.

Las clases del modelo se importan dinamicamente segun `base`:
- `exp01_solo_posicion_temporal`     -> `Modelo2DGSTemporalPos`
- `exp02_posicion_opacidad_temporal` -> `Modelo2DGSTemporalPosOp`
- `exp03_full_temporal`              -> `Modelo2DGSTemporalFull`

## Outputs

Por clip, ademas de las metricas estandar de cada corrida:
- `rd_curve.png` — PSNR/SSIM/LPIPS vs `q` (eje X), tamano modelo en color
- `rd_table.csv` — tabla q, PSNR, SSIM, LPIPS, bytes, ratio

## Como correr

```bash
cd experimentos/exp04_grado_polinomio
python train.py
```
