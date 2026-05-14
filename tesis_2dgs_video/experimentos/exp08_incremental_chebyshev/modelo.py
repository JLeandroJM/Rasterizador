"""
GaussianasChebyshev2D: modelo donde TODOS los parametros de cada gaussiana
son polinomios de Chebyshev en el tiempo.

Diseno de parametros - DECISION:
--------------------------------
Cada parametro p tiene grado q_p. Almacenamos sus coeficientes SEPARADOS
en dos tensores:
    p_a0   : nn.Parameter de shape (N, dim_p, 1)         -- coef a_0
    p_high : nn.Parameter de shape (N, dim_p, q_p)       -- coefs a_1..a_{q_p}

Razones:
  1. El optimizer recibe (vai param_groups) lr distinto para a_0 vs a_high:
        a_0 con lr base; a_high con lr base * 0.1 (mas conservador, evita
        que el modelo explote al recibir grado alto desde cero).
  2. La inicializacion es naturalmente diferente: a_0 lleva el valor
     constante "inicial razonable" del parametro; a_high arranca en 0.
  3. La penalizacion de smoothness ignora a_0 (k=0 -> peso k^2 = 0). Tener
     a_0 separado lo deja fuera del calculo de smoothness de forma natural.

Al evaluar, concatenamos:
    coefs = cat([p_a0, p_high], dim=-1)     # (N, dim_p, q_p + 1)
    p(t_j) = coefs @ B[j, :q_p+1]            # (N, dim_p)
"""
import numpy as np
import torch
from torch import nn

from chebyshev import evaluar_polinomio_cheb



class GaussianasChebyshev2D(nn.Module):

    def __init__(self, n_gaussianas, n_frames, grados, H, W, device,
                 escala_inicial_px=5.0, frame_0_imagen=None, semilla=42):
        """
        Args:
            n_gaussianas    : N
            n_frames        : T  (numero total de frames del clip)
            grados          : dict {'mu':40, 'opacity':40, 'color':20, 'scale':20, 'theta':20, 'depth':20}
            H, W            : alto y ancho de la imagen (para inicializacion de mu)
            device          : torch.device
            escala_inicial_px : escala (en pixeles) que va al coef a_0 de scale (log)
            frame_0_imagen  : tensor opcional (H, W, 3) en [0, 1] -- si se da,
                              el a_0 de color de cada gaussiana se inicializa
                              muestreando bilinealmente esa imagen en la pos inicial.
            semilla         : reproducibilidad de la inicializacion.
        """
        super().__init__()

        self.N = n_gaussianas
        self.n_frames = n_frames
        self.H = H
        self.W = W
        self.grados = dict(grados)   # copia
        self.device = device

        g = torch.Generator(device='cpu').manual_seed(semilla)

        # === mu ============================================================
        # mu vive en pixeles, espacio (fila, columna)
        # a_0: posiciones uniformes en la imagen
        mu_a0_init = torch.empty(n_gaussianas, 2, 1)
        mu_a0_init[:, 0, 0] = torch.rand(n_gaussianas, generator=g) * H
        mu_a0_init[:, 1, 0] = torch.rand(n_gaussianas, generator=g) * W
        # a_high arranca en 0 -> sin movimiento inicial
        mu_high_init = torch.zeros(n_gaussianas, 2, grados['mu'])
        self.mu_a0   = nn.Parameter(mu_a0_init.to(device))
        self.mu_high = nn.Parameter(mu_high_init.to(device))

        # === opacity =======================================================
        # a_0 inicial = sigmoid_inversa(0.5) = 0
        opacity_a0_init = torch.zeros(n_gaussianas, 1)
        opacity_high_init = torch.zeros(n_gaussianas, grados['opacity'])
        # NOTA: opacity es escalar por gaussiana, no tiene "dim_p". Lo
        # tratamos como dim_p=1 implicito.
        self.opacity_a0   = nn.Parameter(opacity_a0_init.to(device))
        self.opacity_high = nn.Parameter(opacity_high_init.to(device))

        # === color =========================================================
        # color_a0: si nos pasan frame_0_imagen, muestreamos el color en la pos
        # inicial; si no, random.
        if frame_0_imagen is not None:
            assert frame_0_imagen.shape == (H, W, 3), "frame_0_imagen debe ser (H, W, 3)"
            color_a0_inicial = _muestrear_color_bilineal(
                frame_0_imagen.detach().cpu(),
                mu_a0_init[:, :, 0],                         # (N, 2) en (fila, col)
            )                                                 # (N, 3) en [0, 1]
            # invertir sigmoid: el modelo guarda raw, aplica sigmoid al evaluar
            color_a0_init = _logit(color_a0_inicial.clamp(1e-4, 1 - 1e-4)).unsqueeze(-1)  # (N, 3, 1)
        else:
            # uniforme en (-1, 1) -> sigmoid da rango variado
            color_a0_init = ((torch.rand(n_gaussianas, 3, 1, generator=g) - 0.5) * 2.0)
        color_high_init = torch.zeros(n_gaussianas, 3, grados['color'])
        self.color_a0   = nn.Parameter(color_a0_init.to(device))
        self.color_high = nn.Parameter(color_high_init.to(device))

        # === scale =========================================================
        # scale real = exp(scale_raw). a_0 = log(escala_inicial_px) por canal.
        scale_a0_init = torch.full((n_gaussianas, 2, 1), float(np.log(escala_inicial_px)))
        scale_high_init = torch.zeros(n_gaussianas, 2, grados['scale'])
        self.scale_a0   = nn.Parameter(scale_a0_init.to(device))
        self.scale_high = nn.Parameter(scale_high_init.to(device))

        # === theta =========================================================
        # rotacion inicial = 0
        theta_a0_init = torch.zeros(n_gaussianas, 1)
        theta_high_init = torch.zeros(n_gaussianas, grados['theta'])
        self.theta_a0   = nn.Parameter(theta_a0_init.to(device))
        self.theta_high = nn.Parameter(theta_high_init.to(device))

        # === depth =========================================================
        # uniforme en [-1, 1] para que el orden front-to-back sea variado
        depth_a0_init = ((torch.rand(n_gaussianas, 1, generator=g) - 0.5) * 2.0)
        depth_high_init = torch.zeros(n_gaussianas, grados['depth'])
        self.depth_a0   = nn.Parameter(depth_a0_init.to(device))
        self.depth_high = nn.Parameter(depth_high_init.to(device))


    # ---- helpers --------------------------------------------------------
    def _coefs_completos(self, p_a0, p_high):
        """Concatena los dos buffers en el eje del grado: (..., q+1)."""
        return torch.cat([p_a0, p_high], dim=-1)


    def parametros_temporales(self):
        """
        Devuelve un dict {nombre: (a_0, a_high, grado, dim_p)} usado por la
        regularizacion de smoothness y por el optimizador.

        dim_p: la dimensionalidad "por gaussiana" del parametro
               (mu y scale: 2; color: 3; opacity/theta/depth: 1).
        """
        return {
            'mu':      (self.mu_a0,      self.mu_high,      self.grados['mu'],      2),
            'opacity': (self.opacity_a0, self.opacity_high, self.grados['opacity'], 1),
            'color':   (self.color_a0,   self.color_high,   self.grados['color'],   3),
            'scale':   (self.scale_a0,   self.scale_high,   self.grados['scale'],   2),
            'theta':   (self.theta_a0,   self.theta_high,   self.grados['theta'],   1),
            'depth':   (self.depth_a0,   self.depth_high,   self.grados['depth'],   1),
        }


    def evaluar_en_frame(self, frame_idx, matrices_cheb):
        """
        Evalua todos los polinomios en el frame `frame_idx` y devuelve los
        parametros YA ACTIVADOS listos para el rasterizador.

        matrices_cheb: dict {grado: tensor (n_frames, grado+1)}.

        Returns dict:
            mu:      (N, 2)        -- coordenadas (fila, columna), sin activacion
            scale:   (N, 2)        -- exp(scale_raw)
            theta:   (N,)          -- sin activacion
            opacity: (N,)          -- sigmoid(opacity_raw)
            color:   (N, 3)        -- sigmoid(color_raw)
            depth:   (N,)          -- sin activacion, solo para ordenar
        """
        out = {}

        for nombre, (p_a0, p_high, grado, dim_p) in self.parametros_temporales().items():
            B = matrices_cheb[grado]                        # (n_frames, grado+1)
            coefs = self._coefs_completos(p_a0, p_high)     # (N, dim_p, grado+1)
            valor_raw = evaluar_polinomio_cheb(coefs, B, frame_idx)  # (N, dim_p)
            if dim_p == 1:
                valor_raw = valor_raw.squeeze(-1)           # (N,)
            out[nombre + '_raw'] = valor_raw

        # activaciones
        out['mu']      = out['mu_raw']
        out['theta']   = out['theta_raw']
        out['depth']   = out['depth_raw']
        out['opacity'] = torch.sigmoid(out['opacity_raw'])
        out['color']   = torch.sigmoid(out['color_raw'])

        # scale clampeada para evitar exp() gigante (origen frecuente de NaN
        # cuando los coefs de orden alto se descontrolan). DECISION: usamos
        # un rango log[0.5, max(H,W)] que cubre desde sub-pixel hasta toda
        # la imagen, y conserva gradiente para valores dentro del rango.
        log_min = float(np.log(0.5))
        log_max = float(np.log(max(self.H, self.W)))
        out['scale'] = torch.exp(out['scale_raw'].clamp(min=log_min, max=log_max))

        return out


    def numero_gausianas(self):
        return self.mu_a0.shape[0]


    # ---- serializacion --------------------------------------------------
    def state_dict_coefs(self):
        """Devuelve solo los coefs (sin el resto del nn.Module state), util para tamano."""
        d = {}
        for nombre, (a0, hi, _, _) in self.parametros_temporales().items():
            d[f"{nombre}_a0"]   = a0.detach().cpu()
            d[f"{nombre}_high"] = hi.detach().cpu()
        d['grados'] = self.grados
        return d


# ===========================================================================
# utilidades auxiliares
# ===========================================================================

def _logit(p):
    """Inversa de sigmoid: log(p / (1-p))."""
    return torch.log(p / (1.0 - p))


def _muestrear_color_bilineal(imagen_hw3, posiciones_fila_col):
    """
    imagen_hw3 : (H, W, 3) en [0, 1]
    posiciones : (N, 2)  con (fila, columna), en pixeles, posiblemente fraccionario
    Devuelve   : (N, 3)  color interpolado bilinealmente en cada posicion.
    Posiciones fuera del rango se clampean al borde.
    """
    H, W, C = imagen_hw3.shape
    f = posiciones_fila_col[:, 0].clamp(0, H - 1)
    c = posiciones_fila_col[:, 1].clamp(0, W - 1)

    f0 = f.floor().long().clamp(0, H - 1)
    f1 = (f0 + 1).clamp(0, H - 1)
    c0 = c.floor().long().clamp(0, W - 1)
    c1 = (c0 + 1).clamp(0, W - 1)

    df = (f - f0.float()).unsqueeze(-1)   # (N, 1)
    dc = (c - c0.float()).unsqueeze(-1)

    Ia = imagen_hw3[f0, c0]               # (N, 3)
    Ib = imagen_hw3[f0, c1]
    Ic = imagen_hw3[f1, c0]
    Id = imagen_hw3[f1, c1]

    return (Ia * (1 - df) * (1 - dc) +
            Ib * (1 - df) * dc +
            Ic * df * (1 - dc) +
            Id * df * dc)



# ===========================================================================
# tests
# ===========================================================================

def _tests():
    """Verifica shapes y que evaluar en frame 0 con a_high=0 da el a_0."""
    from chebyshev import construir_matriz_chebyshev

    device = torch.device('cpu')
    n_g, n_f, H, W = 7, 20, 32, 32
    grados = {'mu': 8, 'opacity': 8, 'color': 4, 'scale': 4, 'theta': 4, 'depth': 4}

    modelo = GaussianasChebyshev2D(n_g, n_f, grados, H, W, device, semilla=0)

    matrices_cheb = {
        8: construir_matriz_chebyshev(n_f, 8, device),
        4: construir_matriz_chebyshev(n_f, 4, device),
    }

    # con todos los a_high = 0, el polinomio es p(t) = a_0 * T_0(t) = a_0
    # asi que el valor evaluado deberia ser igual al a_0 (despues de activaciones)
    out = modelo.evaluar_en_frame(0, matrices_cheb)
    assert out['mu'].shape == (n_g, 2), f"mu shape: {out['mu'].shape}"
    assert out['opacity'].shape == (n_g,)
    assert out['color'].shape == (n_g, 3)
    assert out['scale'].shape == (n_g, 2)
    assert out['theta'].shape == (n_g,)
    assert out['depth'].shape == (n_g,)

    # mu en frame 0 con coefs solo en a_0: deberia coincidir con mu_a0 squeezed
    mu_esperado = modelo.mu_a0[:, :, 0]
    assert torch.allclose(out['mu'], mu_esperado, atol=1e-5), "mu evaluado no coincide con a_0"

    # opacity en frame 0 con opacity_a0 = 0  ->  sigmoid(0) = 0.5
    assert torch.allclose(out['opacity'], torch.full((n_g,), 0.5), atol=1e-5)

    # eval en otro frame con a_high=0 da el mismo a_0 (porque solo T_0 contribuye)
    out_mid = modelo.evaluar_en_frame(n_f // 2, matrices_cheb)
    assert torch.allclose(out_mid['mu'], mu_esperado, atol=1e-5)

    print("[modelo] tests OK")


if __name__ == "__main__":
    _tests()
