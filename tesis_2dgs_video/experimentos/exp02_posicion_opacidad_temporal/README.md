# exp02 — Posicion y opacidad temporales

## Hipotesis

Ademas de `mu(t)` polinomico (como en exp01), ahora `opacidad(t)` tambien es
polinomico. Esto permite que las gaussianas "se prendan y apaguen" suavemente
en el tiempo — el mecanismo central de la tesis para representar contenido
que aparece o desaparece.

Esperamos:
- Mejor calidad que exp01 en clips con contenido emergente (ej. `fade_transition`,
  `swelling_blobs`) donde algunas regiones cambian intensidad.
- Algo mas de parametros que exp01 (N*(q+1) coefs adicionales por opacidad).

## Matematica de la extension temporal

Igual que exp01 para `centro_coefs`, mas:

```
opacidad_coefs : nn.Parameter de shape (N, q+1)

opacidad_raw_t = sum_k opacidad_coefs[:, k] * t^k         # (N,)
opacidad_t     = sigmoid(opacidad_raw_t)
```

La sigmoid se aplica DESPUES de evaluar el polinomio — asi el polinomio puede
moverse libremente en R y la opacidad real queda en (0, 1).

Para que opacidad(t) ~ 0 en algunos frames y ~ alta en otros, el polinomio
necesita pasar por valores muy negativos a positivos. Con grado q=3 ya hay
suficiente expresividad para "encender" y "apagar" una vez por clip.

## Parametros

```
centro_coefs   : (N, 2, q+1)
opacidad_coefs : (N, q+1)        <-- NUEVO
escala         : (N, 2)          estatico
theta          : (N,)            estatico
color          : (N, 3)          estatico
profundidad    : (N,)            sin grad
```

## Output extra: heatmap "vida" de gaussianas

Ademas de las metricas estandar, este experimento genera `heatmap_vida.png`:
matriz de N filas (gaussianas) x T columnas (frames) donde la celda es 1 si
`opacidad(t) > 0.1`, sino 0. Permite ver cuales gaussianas se activan en
distintas regiones temporales del clip. Es uno de los hallazgos mas bonitos
para mostrar en la tesis.

## Como correr

```bash
cd experimentos/exp02_posicion_opacidad_temporal
python train.py
```
