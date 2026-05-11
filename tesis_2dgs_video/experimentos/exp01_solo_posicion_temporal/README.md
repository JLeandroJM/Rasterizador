# exp01 — Solo posicion temporal

## Hipotesis

La posicion `mu` de cada gaussiana se vuelve un polinomio en `t` de grado `q`.
Todos los demas parametros son estaticos. Es la configuracion temporal minima.

Esperamos:
- Tamano de modelo mucho menor que exp00 (un solo modelo para todo el clip).
- Calidad razonable en clips con movimiento suave (los polinomios de grado bajo
  representan bien trayectorias suaves).
- Calidad pobre en clips con cambios bruscos (rebotes con direccion abrupta).

## Matematica de la extension temporal

Para cada gaussiana `i`, su centro en el frame `t` es:

```
mu_i(t) = a_{i,0} + a_{i,1} * t + a_{i,2} * t^2 + ... + a_{i,q} * t^q
```

donde `t = frame_idx / (num_frames - 1)` esta normalizado a `[0, 1]`.
Los coeficientes `a_{i,k}` para `k = 0..q` son los parametros aprendidos:

```
centro_coefs : nn.Parameter de shape (N, 2, q+1)
```

En el forward para renderizar el frame `t`:

```python
t_powers = torch.stack([t**k for k in range(q+1)])      # (q+1,)
centro_t = (centro_coefs * t_powers).sum(dim=-1)        # (N, 2)
# luego rasterizar normal con centro_t y los demas params estaticos
```

Autograd se encarga del backward sobre los coeficientes — no hay nada exotico:
es una combinacion lineal de los coeficientes con potencias de `t`. Las
potencias de `t` se calculan SIN grad (son escalares constantes para cada frame).

## Parametros

```
centro_coefs : (N, 2, q+1)   — coeficientes del polinomio por gaussiana / dim
escala       : (N, 2)        — estatico
theta        : (N,)          — estatico
opacidad     : (N,)          — estatico
color        : (N, 3)        — estatico
profundidad  : (N,)          — estatico (sin grad)
```

## Como correr

```bash
cd experimentos/exp01_solo_posicion_temporal
python train.py
```

## Hiperparametros principales

- `N = 200`
- `q = 3` (cubico — suficiente para movimientos suaves)
- `iteraciones = 3000` sobre el clip completo (no por frame)
- Cada iteracion samplea un frame aleatorio del clip.

## Decisiones

- `# DECISION:` base monomial `(1, t, t^2, ..., t^q)`. Alternativa: Chebyshev
  T_k(2t-1) (mejor numericamente para q alto) — no implementada por simplicidad.
- `# DECISION:` un frame por iteracion (no batch). Mas legible, suficiente para
  3000 iters.

## Outputs

Los mismos que el resto de experimentos (ver README raiz).
