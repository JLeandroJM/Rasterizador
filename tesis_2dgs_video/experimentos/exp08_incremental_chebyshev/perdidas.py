"""
Funciones de perdida.

loss_render        : (1 - lambda) L1 + lambda DSSIM   (lambda = 0.2)
loss_smoothness    : penalizacion de coeficientes de orden alto

Sobre smoothness:
-----------------
Para cada parametro temporal de cada gaussiana, penalizamos:
    sum_{k=1}^{q} (a_k)^2 * k^2

El factor k^2 hace que los coeficientes de grado alto cuesten cuadraticamente
mas que los de grado bajo. Esto fuerza al modelo a usar grado alto SOLO cuando
realmente lo necesita, evitando que oscile a frecuencias altas para ajustar
ruido del target.

Como en el modelo separamos los coefs en (a_0, a_high), donde a_high tiene
en su indice k_local=0 el valor de a_1, en k_local=1 el valor de a_2, etc.,
el factor k^2 que va dentro de la suma corresponde a k = k_local + 1.

beta global multiplica la suma final. pesos_por_param permite ajustar la
importancia relativa entre tipos de parametros (ej. queremos mas suavidad en
color que en opacity).
"""
import torch
import torch.nn.functional as F



def _kernel_gauss_2d(tamano=11, sigma=1.5, device='cpu', dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def _dssim_diferenciable(x, y):
    """DSSIM con grad. x, y: (H, W, 3) en [0, 1]."""
    x_b = x.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = y.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    C = x_b.shape[1]
    device = x_b.device
    kernel = _kernel_gauss_2d(11, 1.5, device, x_b.dtype).expand(C, 1, 11, 11)

    mu_x = F.conv2d(x_b, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=5, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x_b * x_b, kernel, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(y_b * y_b, kernel, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(x_b * y_b, kernel, padding=5, groups=C) - mu_xy
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2))
    return (1.0 - ssim_map.mean()) / 2.0



def loss_render(render, target, lambda_dssim=0.2):
    """L1 + DSSIM combinada."""
    l1 = torch.mean(torch.abs(render - target))
    dssim = _dssim_diferenciable(render, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim



def loss_smoothness(modelo, pesos_por_param=None):
    """
    Penaliza coeficientes de orden alto.

    pesos_por_param: dict opcional {nombre_param: peso}. Default 1.0 cada uno.

    Para cada parametro temporal:
        loss += peso_param * sum sobre gaussianas, dims, y k de:
                (a_high[..., k])^2 * (k+1)^2
        (k+1 porque a_high empieza en a_1, asi que k_local=0 corresponde a a_1)
    """
    total = None
    for nombre, (a0, a_high, grado, _) in modelo.parametros_temporales().items():
        peso = (pesos_por_param or {}).get(nombre, 1.0)
        if peso == 0.0 or grado == 0:
            continue

        # a_high shape (N, dim_p, grado). El indice k del ultimo eje corresponde
        # al coeficiente a_{k+1}, asi que el factor de penalizacion es (k+1)^2.
        k_idx = torch.arange(1, grado + 1, device=a_high.device, dtype=a_high.dtype)   # (grado,)
        factor = k_idx ** 2                                                            # (grado,)

        # broadcast: a_high^2 * factor, sumar todo
        contrib = (a_high ** 2 * factor).sum() * peso
        total = contrib if total is None else (total + contrib)

    if total is None:
        # ningun parametro temporal con peso > 0
        return torch.tensor(0.0, device=modelo.mu_a0.device)
    return total
