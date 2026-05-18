import os
import sys
import time
import argparse

import torch
import numpy as np
from PIL import Image

import raster_cuda


# ============================================================
# Rutas
# ============================================================

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))

EXP09 = os.path.join(
    RAIZ,
    "tesis_2dgs_video",
    "experimentos",
    "exp09_batchfull_comparacion_bases"
)

sys.path.insert(0, EXP09)

from modelo import GaussianasPolinomial2D
from bases import construir_matriz
from rasterizador import rasterizar_un_frame


# ============================================================
# Utilidades CUDA
# ============================================================

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


def calcular_tiles_tocados_gpu(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    device = mu.device
    N = mu.shape[0]

    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    total_tiles = tiles_x * tiles_y

    fila = mu[:, 0]
    col = mu[:, 1]

    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    extent_fila = k_sigma * torch.sqrt((c * sx) ** 2 + (s * sy) ** 2)
    extent_col = k_sigma * torch.sqrt((s * sx) ** 2 + (c * sy) ** 2)

    min_f = torch.floor(fila - extent_fila).clamp(0, H - 1).to(torch.int64)
    max_f = torch.ceil(fila + extent_fila).clamp(0, H - 1).to(torch.int64)
    min_c = torch.floor(col - extent_col).clamp(0, W - 1).to(torch.int64)
    max_c = torch.ceil(col + extent_col).clamp(0, W - 1).to(torch.int64)

    tile_y_min = torch.div(min_f, tile_size, rounding_mode="floor")
    tile_y_max = torch.div(max_f, tile_size, rounding_mode="floor")
    tile_x_min = torch.div(min_c, tile_size, rounding_mode="floor")
    tile_x_max = torch.div(max_c, tile_size, rounding_mode="floor")

    num_y = tile_y_max - tile_y_min + 1
    num_x = tile_x_max - tile_x_min + 1
    counts = num_y * num_x

    total_instancias = int(counts.sum().item())

    gaussian_ids = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64),
        counts
    )

    starts = torch.cumsum(counts, dim=0) - counts
    local_offsets = torch.arange(total_instancias, device=device, dtype=torch.int64) - starts[gaussian_ids]

    nx = num_x[gaussian_ids]

    local_y = torch.div(local_offsets, nx, rounding_mode="floor")
    local_x = local_offsets - local_y * nx

    tx = tile_x_min[gaussian_ids] + local_x
    ty = tile_y_min[gaussian_ids] + local_y

    tile_ids = ty * tiles_x + tx

    # Importante:
    # En exp08/exp09 depth puede ser negativo o mayor que 1.
    # Entonces normalizamos depth solo para ordenar.
    depth_vals = depth[gaussian_ids]
    d_min = depth_vals.min()
    d_max = depth_vals.max()
    depth_norm = (depth_vals - d_min) / (d_max - d_min + 1e-8)
    depth_q = (depth_norm * 1_000_000).to(torch.int64)

    keys = tile_ids * 1_000_000 + depth_q

    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    if tile_ids.numel() > 0:
        is_start = torch.empty_like(tile_ids, dtype=torch.bool)
        is_start[0] = True
        is_start[1:] = tile_ids[1:] != tile_ids[:-1]

        starts_idx = torch.nonzero(is_start, as_tuple=False).flatten()
        unique_tiles = tile_ids[starts_idx]

        ends_idx = torch.empty_like(starts_idx)
        if starts_idx.numel() > 1:
            ends_idx[:-1] = starts_idx[1:]
        ends_idx[-1] = tile_ids.numel()

        ranges[unique_tiles, 0] = starts_idx
        ranges[unique_tiles, 1] = ends_idx

    return gaussian_ids.contiguous(), ranges.contiguous(), tiles_x, tiles_y


def rasterizar_cuda_tiled(params, H, W, tile_size=16, k_sigma=3.5):
    mu = params["mu"].contiguous()
    scale = params["scale"].contiguous()
    theta = params["theta"].contiguous()
    opacity = params["opacity"].contiguous()
    color = params["color"].contiguous()
    depth = params["depth"].contiguous()

    conic = construir_conic(scale, theta)

    gaussian_ids, ranges, _, _ = calcular_tiles_tocados_gpu(
        mu,
        scale,
        theta,
        depth,
        H,
        W,
        tile_size=tile_size,
        k_sigma=k_sigma
    )

    return raster_cuda.forward_tiled(
        mu,
        conic,
        opacity,
        color,
        gaussian_ids,
        ranges,
        H,
        W,
        tile_size
    )


def rasterizar_cuda_conic(params, H, W):
    depth = params["depth"]
    idx = torch.argsort(depth)

    mu = params["mu"][idx].contiguous()
    scale = params["scale"][idx].contiguous()
    theta = params["theta"][idx].contiguous()
    opacity = params["opacity"][idx].contiguous()
    color = params["color"][idx].contiguous()

    conic = construir_conic(scale, theta)

    return raster_cuda.forward_conic(
        mu,
        conic,
        opacity,
        color,
        H,
        W
    )


# ============================================================
# Carga del checkpoint
# ============================================================

def cargar_state_dict_coefs_en_modelo(modelo, state_dict_coefs):
    nombres = [
        "mu_a0", "mu_high",
        "opacity_a0", "opacity_high",
        "color_a0", "color_high",
        "scale_a0", "scale_high",
        "theta_a0", "theta_high",
        "depth_a0", "depth_high",
    ]

    with torch.no_grad():
        for nombre in nombres:
            if nombre not in state_dict_coefs:
                raise KeyError(f"Falta {nombre} en state_dict_coefs")

            destino = getattr(modelo, nombre)
            origen = state_dict_coefs[nombre].to(destino.device, dtype=destino.dtype)

            if destino.shape != origen.shape:
                raise RuntimeError(
                    f"Shape incompatible en {nombre}: modelo {tuple(destino.shape)} vs checkpoint {tuple(origen.shape)}"
                )

            destino.copy_(origen)


def reconstruir_modelo_desde_checkpoint(ruta_checkpoint, H_arg, W_arg, n_frames_arg, device):
    ckpt = torch.load(ruta_checkpoint, map_location="cpu")

    if "state_dict_coefs" not in ckpt:
        raise KeyError("El checkpoint no tiene state_dict_coefs")

    sd = ckpt["state_dict_coefs"]
    config = ckpt.get("config", {})

    grados = sd.get("grados", config.get("grados", None))
    if grados is None:
        raise KeyError("No se encontro grados ni en state_dict_coefs ni en config")

    base = sd.get("base", config.get("base", "chebyshev"))

    N = int(sd["mu_a0"].shape[0])

    H = int(sd.get("H", H_arg))
    W = int(sd.get("W", W_arg))
    n_frames = int(sd.get("n_frames", n_frames_arg))

    if H is None or W is None or n_frames is None:
        raise ValueError("Debes pasar --H --W --n_frames porque el checkpoint no los trae")

    modelo = GaussianasPolinomial2D(
        n_gaussianas=N,
        n_frames=n_frames,
        grados=grados,
        base=base,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=config.get("escala_inicial_px", 5.0),
        frame_0_imagen=None,
        semilla=config.get("seed", 42),
    )

    cargar_state_dict_coefs_en_modelo(modelo, sd)
    modelo.eval()

    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz(base, n_frames, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }

    return modelo, matrices_base, config, H, W, n_frames, base, N


# ============================================================
# Guardado
# ============================================================

def guardar_png(tensor_hw3, ruta):
    arr = tensor_hw3.detach().clamp(0, 1).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr).save(ruta)


def crear_gif(carpeta_frames, ruta_gif, fps=10):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[WARN] imageio no instalado. No se creo GIF.")
        return

    archivos = sorted(
        f for f in os.listdir(carpeta_frames)
        if f.startswith("frame_") and f.endswith(".png")
    )

    if not archivos:
        print("[WARN] No hay frames para crear GIF.")
        return

    imgs = []
    for f in archivos:
        imgs.append(imageio.imread(os.path.join(carpeta_frames, f)))

    imageio.mimsave(ruta_gif, imgs, duration=1.0 / fps)
    print(f"gif guardado: {ruta_gif}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salida", default=None)

    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--n_frames", type=int, default=None)

    parser.add_argument("--modo", choices=["tiled", "conic"], default="tiled")
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--k_sigma", type=float, default=3.5)

    parser.add_argument("--comparar_torch", action="store_true")
    parser.add_argument("--solo_frame", type=int, default=None)

    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--fps", type=int, default=10)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    device = torch.device("cuda")

    if args.salida is None:
        nombre = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.salida = os.path.join(AQUI, f"preview_cuda_{nombre}")

    os.makedirs(args.salida, exist_ok=True)

    modelo, matrices_base, config, H, W, n_frames, base, N = reconstruir_modelo_desde_checkpoint(
        args.checkpoint,
        args.H,
        args.W,
        args.n_frames,
        device
    )

    print("=== Checkpoint cargado ===")
    print(f"checkpoint : {args.checkpoint}")
    print(f"salida     : {args.salida}")
    print(f"base       : {base}")
    print(f"N          : {N}")
    print(f"frames     : {n_frames}")
    print(f"resolucion : {H} x {W}")
    print(f"modo       : {args.modo}")
    print(f"device     : {torch.cuda.get_device_name(0)}")

    if args.solo_frame is not None:
        frames_a_renderizar = [args.solo_frame]
    else:
        frames_a_renderizar = list(range(n_frames))

    tiempos = []

    with torch.no_grad():
        for idx, j in enumerate(frames_a_renderizar):
            params_j = modelo.evaluar_en_frame(j, matrices_base)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            if args.modo == "tiled":
                render = rasterizar_cuda_tiled(
                    params_j,
                    H,
                    W,
                    tile_size=args.tile,
                    k_sigma=args.k_sigma
                )
            else:
                render = rasterizar_cuda_conic(params_j, H, W)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            render = render.clamp(0, 1)
            tiempos.append((t1 - t0) * 1000.0)

            ruta_png = os.path.join(args.salida, f"frame_{j:04d}.png")
            guardar_png(render, ruta_png)

            if args.comparar_torch:
                render_torch = rasterizar_un_frame(params_j, H, W).clamp(0, 1)
                max_diff = (render_torch - render).abs().max().item()
                mean_diff = (render_torch - render).abs().mean().item()
                print(
                    f"frame {j:04d} | cuda {tiempos[-1]:.4f} ms | "
                    f"max_diff {max_diff:.8f} | mean_diff {mean_diff:.8f}"
                )
            else:
                print(f"frame {j:04d} | cuda {tiempos[-1]:.4f} ms")

    if tiempos:
        print("\n=== Tiempo CUDA ===")
        print(f"frames renderizados : {len(tiempos)}")
        print(f"promedio ms/frame   : {sum(tiempos) / len(tiempos):.4f}")
        print(f"min ms/frame        : {min(tiempos):.4f}")
        print(f"max ms/frame        : {max(tiempos):.4f}")

    if args.gif and args.solo_frame is None:
        ruta_gif = os.path.join(args.salida, "preview_cuda.gif")
        crear_gif(args.salida, ruta_gif, fps=args.fps)

    print("\nlisto")


if __name__ == "__main__":
    main()