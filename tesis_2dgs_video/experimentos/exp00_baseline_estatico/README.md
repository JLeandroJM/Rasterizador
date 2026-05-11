# exp00 — Baseline estatico (control)

## Hipotesis

Aplicar 2DGS estatico independientemente a cada frame, sin compartir nada
entre frames. Este es el baseline contra el cual el aporte de la representacion
temporal compartida debe demostrar valor.

Si el modelo temporal no supera a este baseline en ratio de compresion para una
calidad fija, no hay aporte.

## Que esperar

- PSNR/SSIM aceptables porque cada frame se ajusta independiente.
- Tamano de modelo ENORME: N * params_por_gaussiana * 4 bytes * num_frames.
  Con N=200 y 90 frames, es como guardar 90 modelos completos.
- Ratio de compresion vs raw probablemente bajo o negativo.

## Como correr

```bash
cd experimentos/exp00_baseline_estatico
python train.py
```

Procesa todos los clips de `clips/` y guarda resultados en
`resultados/exp00_baseline_estatico/<clip>/`.

## Outputs (por clip)

- `metricas.json` — PSNR, SSIM, LPIPS promedio, tamano, ratio, tiempo
- `loss.csv`, `loss.png` — curva de loss (acumulada sobre todos los frames)
- `metricas_por_frame.csv`, `metricas_por_frame.png` — PSNR/SSIM por frame_idx
- `reconstruido/frame_NNNN.png` — secuencia reconstruida
- `comparativa.gif` — original | reconstruido | diferencia x5
- `reconstruido.gif` — solo el reconstruido (cada 3er frame)
- `checkpoint.pt` — todos los modelos concatenados

## Decisiones

- `# DECISION:` 500 iteraciones por frame. Mas que eso hace inviable correr
  los 5 clips en tiempo razonable. Es el valor estandar para baselines de este
  tipo cuando cada frame se entrena por separado.
- N=200 gaussianas por frame.
- Inicializacion aleatoria con `semilla = frame_idx` para que cada frame
  empiece distinto.
