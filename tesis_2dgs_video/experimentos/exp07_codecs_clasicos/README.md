# exp07 — Comparacion con codecs clasicos (H.264, H.265)

## Hipotesis

Comprimir los mismos 5 clips con H.264 y H.265 a varias tasas (CRF) y medir
las mismas metricas (PSNR, SSIM, LPIPS) para situarlos en la misma curva
Rate-Distortion que tu metodo. Este es el grafico central del capitulo de
comparacion de tu tesis.

## Como funciona

Para cada clip:
1. Codifica las PNG-sequence con `ffmpeg` a H.264 / H.265 con cada CRF.
2. Decodifica el video de vuelta a frames.
3. Mide PSNR/SSIM/LPIPS por frame.
4. Reporta tamano del archivo codificado.

CRFs: 18 (alta calidad), 23 (defecto), 28, 35 (baja calidad).
Codecs: `libx264`, `libx265`.

## Outputs

Por clip:
- `videos/<codec>_crf<NN>.mp4` — archivos codificados
- `tabla_rd.csv` — codec, crf, bytes, ratio, PSNR, SSIM, LPIPS
- `rd_curve.png` — curva R-D para H.264, H.265 (en el mismo grafico)

`tools/comparar_experimentos.py` despues junta esto con la curva R-D del
metodo Gaussian Splatting Temporal (exp05).

## Como correr

```bash
cd experimentos/exp07_codecs_clasicos
python run.py
```

Requiere `ffmpeg` en el PATH (ya disponible en macOS via homebrew).
