# exp06 — Cuantizacion (reduccion de precision)

## Hipotesis

Tomar un modelo entrenado de `exp05_numero_gaussianas` con N intermedio (200)
y comprimir adicionalmente reduciendo la precision de los parametros.
Reportar la perdida de SSIM y la reduccion de tamano.

Variantes:
- `float32` (baseline, sin cambio)
- `float16`
- `bfloat16`
- `int8` cuantizado simetrico (DECISION: opcional, se incluye si tiempo lo permite)

Adicional: aplicar `zlib`/`zstd` al archivo final, reportar tamano post
compresion entropica.

## Como correr

```bash
cd experimentos/exp06_cuantizacion
python eval.py
```

Por defecto carga `resultados/exp05_numero_gaussianas/<clip>/N=200/checkpoint.pt`.

## Outputs

Por clip:
- `tabla_cuantizacion.csv` — dtype, bytes, bytes_compresos_zlib, PSNR, SSIM, LPIPS
- `comparativa.png` — barras comparando tamano y SSIM por dtype

## Decisiones

- `# DECISION:` cuantizacion int8 simetrica por-tensor. Cada parametro se
  escala a `[-127, 127]` con un factor `s = max(|x|)/127`. Para
  des-cuantizar: `x ~ q * s`. El factor `s` se guarda en float32. No es
  cuantizacion fina (per-channel) — se podria mejorar.
