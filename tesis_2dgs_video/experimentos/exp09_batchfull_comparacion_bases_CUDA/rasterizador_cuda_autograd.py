import os
import sys
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))

# Estamos en:
# D:\TesisProyecto\Rasterizador\tesis_2dgs_video\experimentos\exp09_batchfull_comparacion_bases_CUDA
#
# Queremos llegar a:
# D:\TesisProyecto\Rasterizador\raster_cuda

RAIZ_RASTERIZADOR = os.path.abspath(
    os.path.join(AQUI, "..", "..", "..")
)

RUTA_RASTER_CUDA = os.path.join(RAIZ_RASTERIZADOR, "raster_cuda")

print(f"[rasterizador_cuda_autograd] buscando raster_cuda en: {RUTA_RASTER_CUDA}", flush=True)

if not os.path.isdir(RUTA_RASTER_CUDA):
    raise FileNotFoundError(f"No existe la carpeta raster_cuda: {RUTA_RASTER_CUDA}")

if RUTA_RASTER_CUDA not in sys.path:
    sys.path.insert(0, RUTA_RASTER_CUDA)

try:
    import raster_cuda
except Exception as e:
    print("[rasterizador_cuda_autograd] sys.path usado:", flush=True)
    for p in sys.path[:8]:
        print("  ", p, flush=True)
    raise e

def construir_conic(scale, theta):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    inv_sx2 = 1.0 / (sx * sx + 1e-8)
    inv_sy2 = 1.0 / (sy * sy + 1e-8)

    m00 = c * c * inv_sx2 + s * s * inv_sy2
    m01 = c * s * (inv_sx2 - inv_sy2)
    m11 = s * s * inv_sx2 + c * c * inv_sy2

    return torch.stack([m00, m01, m11], dim=1).contiguous()


def grad_conic_a_scale_theta(scale, theta, grad_conic):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    den_x = sx * sx + 1e-8
    den_y = sy * sy + 1e-8

    A = 1.0 / den_x
    B = 1.0 / den_y

    gm00 = grad_conic[:, 0]
    gm01 = grad_conic[:, 1]
    gm11 = grad_conic[:, 2]

    grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s
    grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c

    dA_dsx = -2.0 * sx / (den_x * den_x)
    dB_dsy = -2.0 * sy / (den_y * den_y)

    grad_sx = grad_A * dA_dsx
    grad_sy = grad_B * dB_dsy

    grad_scale = torch.stack([grad_sx, grad_sy], dim=1)

    dm00_dtheta = 2.0 * c * s * (B - A)
    dm01_dtheta = (c * c - s * s) * (A - B)
    dm11_dtheta = 2.0 * c * s * (A - B)

    grad_theta = (
        gm00 * dm00_dtheta +
        gm01 * dm01_dtheta +
        gm11 * dm11_dtheta
    )

    return grad_scale, grad_theta


class RasterizarConicCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, theta, opacity, color, H, W):
        mu_c = mu.contiguous()
        scale_c = scale.contiguous()
        theta_c = theta.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()

        conic = construir_conic(scale_c, theta_c)

        render = raster_cuda.forward_conic(
            mu_c,
            conic,
            opacity_c,
            color_c,
            H,
            W
        )

        ctx.save_for_backward(
            mu_c,
            scale_c,
            theta_c,
            conic,
            opacity_c,
            color_c
        )
        ctx.H = H
        ctx.W = W

        return render

    @staticmethod
    def backward(ctx, grad_render):
        mu, scale, theta, conic, opacity, color = ctx.saved_tensors

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_conic(
            mu,
            conic,
            opacity,
            color,
            grad_render.contiguous(),
            ctx.H,
            ctx.W
        )

        grad_scale, grad_theta = grad_conic_a_scale_theta(
            scale,
            theta,
            grad_conic
        )

        return (
            grad_mu,
            grad_scale,
            grad_theta,
            grad_opacity,
            grad_color,
            None,
            None
        )


def rasterizar_un_frame_cuda_conic(params_frame, H, W):
    """
    Version diferenciable CUDA.

    Recibe params_frame con:
        mu      : (N, 2)
        scale   : (N, 2)
        theta   : (N,)
        opacity : (N,)
        color   : (N, 3)

    Devuelve:
        render  : (H, W, 3)
    """
    depth = params_frame["depth"]
    idx = torch.argsort(depth)

    mu = params_frame["mu"][idx].contiguous()
    scale = params_frame["scale"][idx].contiguous()
    theta = params_frame["theta"][idx].contiguous()
    opacity = params_frame["opacity"][idx].contiguous()
    color = params_frame["color"][idx].contiguous()

    return RasterizarConicCUDA.apply(
        mu,
        scale,
        theta,
        opacity,
        color,
        H,
        W
    )