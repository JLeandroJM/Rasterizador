# exp03 — Full temporal

## Hipotesis

TODOS los parametros son tiempo-dependientes (polinomios en t):
`centro(t)`, `escala(t)`, `theta(t)`, `opacidad(t)`, `color(t)`.

Configuracion de maxima expresividad. La pregunta: la ganancia de PSNR/SSIM
justifica el aumento de parametros?

## Matematica

Para cada parametro X que era estatico, su version temporal es:

```
X_coefs : (N, dim_X, q+1)
X_raw(t) = sum_k X_coefs[..., k] * t^k       # (N, dim_X)
```

Activaciones:
- centro:   sin activacion (centro(t) = centro_raw(t))
- escala:   exp(escala_raw(t))               siempre positiva
- theta:    sin activacion
- opacidad: sigmoid(opacidad_raw(t))         en (0, 1)
- color:    sigmoid(color_raw(t))            en (0, 1)

## Parametros

```
centro_coefs   : (N, 2, q+1)
escala_coefs   : (N, 2, q+1)
theta_coefs    : (N, q+1)
opacidad_coefs : (N, q+1)
color_coefs    : (N, 3, q+1)
profundidad    : (N,)            sin grad
```

Total parametros optimizables por gaussiana: `(2 + 2 + 1 + 1 + 3) * (q+1) = 9*(q+1)`
vs `5*(q+1) + 5` en exp01 + estaticos.

## Como correr

```bash
cd experimentos/exp03_full_temporal
python train.py
```
