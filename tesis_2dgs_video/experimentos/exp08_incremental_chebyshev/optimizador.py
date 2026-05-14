"""
Adam con param groups por TIPO de parametro Y por ORDEN del coeficiente.

DECISION (separacion a_0 vs a_high):
------------------------------------
Como el modelo ya almacena a_0 y a_high en nn.Parameters distintos
(mu_a0 + mu_high, opacity_a0 + opacity_high, ...), basta con crear un
param group para cada uno con su lr correspondiente. No hace falta usar
mascaras de gradiente ni manipular tensores fusionados.

Esquema de lrs (cada parametro tiene dos lrs):
    <param>_a0:    lr base, el coef constante puede moverse rapido
    <param>_high:  lr base * 0.1, los coefs de orden alto se mueven mas
                   lento para evitar inestabilidades.
"""
import torch



def construir_optimizador(modelo, lrs):
    """
    lrs: dict con claves *_a0 y *_high para cada parametro temporal.
    Si falta alguna clave usamos un default razonable.
    """
    # defaults (por si el config se queda corto)
    DEFAULTS = {
        'mu_a0':      1e-3, 'mu_high':      1e-4,
        'opacity_a0': 5e-2, 'opacity_high': 5e-3,
        'color_a0':   1e-2, 'color_high':   1e-3,
        'scale_a0':   5e-3, 'scale_high':   5e-4,
        'theta_a0':   5e-3, 'theta_high':   5e-4,
        'depth_a0':   5e-3, 'depth_high':   5e-4,
    }
    def lr(clave):
        return lrs.get(clave, DEFAULTS[clave])

    grupos = []
    for nombre in ['mu', 'opacity', 'color', 'scale', 'theta', 'depth']:
        a0 = getattr(modelo, f"{nombre}_a0")
        hi = getattr(modelo, f"{nombre}_high")
        grupos.append({'params': [a0], 'lr': lr(f"{nombre}_a0"),   'name': f"{nombre}_a0"})
        grupos.append({'params': [hi], 'lr': lr(f"{nombre}_high"), 'name': f"{nombre}_high"})

    return torch.optim.Adam(grupos)
