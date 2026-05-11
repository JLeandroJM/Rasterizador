# exp05 — Barrido del numero de gaussianas

## Hipotesis

Con la configuracion ganadora de exp01/02/03 y el `q` ganador de exp04, barrer
`N ∈ {50, 100, 200, 500, 1000}` y medir como cambia PSNR/SSIM/LPIPS vs el
tamano del modelo.

Este es **el experimento mas importante de tu tesis**: produce la curva
Rate-Distortion de tu metodo. Tamano del modelo en X, SSIM (o PSNR) en Y.
Esa curva define tu propuesta.

## Decision

`# DECISION:` por defecto este experimento usa la configuracion de exp02
(`base = "exp02_posicion_opacidad_temporal"`) con `q = 3`. Cambia `base` y
`q` en `config.json` segun los resultados de exp03 y exp04.

## Outputs

Por clip:
- Subcarpeta `N=K/` por cada valor de N
- `rd_curve.png` — PSNR/SSIM vs bytes_modelo (eje X), con curva paramétrica
- `rd_table.csv`

## Como correr

```bash
cd experimentos/exp05_numero_gaussianas
python train.py
```
