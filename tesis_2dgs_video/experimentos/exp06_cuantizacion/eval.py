"""
exp06 — cuantizacion: carga checkpoints de exp05 (N=N_objetivo) y reporta
metricas con float32, float16, bfloat16, y int8 cuantizado simetrico.

NO entrena nada. Solo evalua.

Uso:
    cd experimentos/exp06_cuantizacion
    python eval.py
"""
import json
import os
import sys
import zlib
import io

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelos import construir_modelo_por_base
from rasterizador import rasterizar_diferenciable
from metricas import (
    psnr, calcular_ssim, calcular_lpips,
    tamano_bytes_state_dict, ratio_compresion,
    asegurar_carpeta, guardar_csv, escribir_metricas_json,
    BYTES_POR_DTYPE,
)



def cargar_clip(carpeta_clip, device):
    archivos = sorted(f for f in os.listdir(carpeta_clip) if f.startswith("frame_") and f.endswith(".png"))
    frames = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).to(device))
    return torch.stack(frames, dim=0)



def reducir_precision_state_dict(sd, dtype_nombre):
    """
    Convierte los tensores del state_dict al dtype objetivo.
    Devuelve (sd_reducido, sd_para_renderizar_en_float32).

    - float32, float16, bfloat16: simplemente .to(dtype)
    - int8: cuantizacion simetrica por-tensor (factor en float32 aparte)

    Para renderizar siempre volvemos a float32 (los rasterizadores no son
    estables con dtypes reducidos en MPS).
    """
    NOMBRE_A_DTYPE = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_nombre in NOMBRE_A_DTYPE:
        dt = NOMBRE_A_DTYPE[dtype_nombre]
        sd_reducido = {}
        sd_render = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                sd_reducido[k] = v.to(dt)
                sd_render[k]   = sd_reducido[k].to(torch.float32)
            else:
                sd_reducido[k] = v
                sd_render[k]   = v
        return sd_reducido, sd_render

    if dtype_nombre == "int8":
        # cuantizacion simetrica por-tensor:
        #   q = round(x / s),  s = max(|x|) / 127
        # des-cuantizar: x ~ q * s
        sd_reducido = {}
        sd_render = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                abs_max = v.abs().max().item()
                s = abs_max / 127.0 if abs_max > 0 else 1.0
                q = torch.round(v / s).clamp(-127, 127).to(torch.int8)
                sd_reducido[k]              = q
                sd_reducido[k + "__scale"]  = torch.tensor([s], dtype=torch.float32)
                sd_render[k] = q.to(torch.float32) * s
            else:
                sd_reducido[k] = v
                sd_render[k]   = v
        return sd_reducido, sd_render

    raise ValueError(f"dtype desconocido: {dtype_nombre}")



def cargar_y_aplicar_a_modelo(sd_render, modelo):
    """Sobreescribe los nn.Parameter del modelo con los tensores del sd."""
    with torch.no_grad():
        for nombre in ["centro_coefs", "opacidad_coefs", "escala_coefs",
                        "theta_coefs", "color_coefs",
                        "escala", "theta", "opacidad", "color",
                        "centro"]:
            if hasattr(modelo, nombre) and nombre in sd_render:
                attr = getattr(modelo, nombre)
                if isinstance(attr, torch.nn.Parameter):
                    attr.data.copy_(sd_render[nombre].to(attr.device, dtype=attr.dtype))
                else:
                    setattr(modelo, nombre, sd_render[nombre].to(attr.device, dtype=attr.dtype))
        if "profundidad" in sd_render:
            modelo.profundidad = sd_render["profundidad"].to(modelo.centro_coefs.device if hasattr(modelo, 'centro_coefs') else modelo.centro.device)



def tamano_sd_post_zlib(sd):
    """Serializa el state_dict a bytes y aplica zlib. Devuelve tamano comprimido."""
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    return len(zlib.compress(raw, level=9))



def renderizar_clip(modelo, T, H, W, device):
    """Renderiza todos los frames y devuelve lista de tensores (H, W, 3)."""
    rs = []
    with torch.no_grad():
        for fi in range(T):
            t_norm = fi / (T - 1) if T > 1 else 0.0
            c, e, th, o_raw, k, p = modelo.evaluar_en_t(t_norm)
            r = rasterizar_diferenciable(c, e, th, o_raw, k, p, H, W).clamp(0, 1)
            rs.append(r)
    return rs



def procesar_clip(nombre_clip, carpeta_clip, ckpt_ruta, carpeta_salida, config, device):
    print(f"\n===== procesando clip: {nombre_clip} =====")
    frames = cargar_clip(carpeta_clip, device)
    T, H, W, _ = frames.shape

    asegurar_carpeta(carpeta_salida)

    ckpt = torch.load(ckpt_ruta, map_location='cpu', weights_only=False)
    sd_orig = ckpt["state_dict"]
    config_orig = ckpt["config"]
    base = config_orig["base"]
    q = ckpt["N"] if False else config_orig.get("q", 3)   # N esta en ckpt, q en config

    N = config["N_objetivo"]
    print(f"  cargando checkpoint: N={N}, base={base}, q={q}")

    # construimos un modelo "fresco" para sobreescribir sus params al renderizar
    modelo = construir_modelo_por_base(base, N, q, H, W, device,
                                         escala_inicial=1.0, semilla=0)

    filas = []
    for dtype_nombre in config["dtypes"]:
        sd_reducido, sd_render = reducir_precision_state_dict(sd_orig, dtype_nombre)

        bytes_directos = tamano_bytes_state_dict(sd_reducido)
        bytes_zlib = tamano_sd_post_zlib(sd_reducido)
        ratio_directo = ratio_compresion(T, H, W, bytes_directos)
        ratio_zlib = ratio_compresion(T, H, W, bytes_zlib)

        # cargar params al modelo y renderizar
        cargar_y_aplicar_a_modelo(sd_render, modelo)
        renders = renderizar_clip(modelo, T, H, W, device)

        psnrs, ssims, lpipss = [], [], []
        for fi in range(T):
            psnrs.append(psnr(renders[fi], frames[fi]))
            ssims.append(calcular_ssim(renders[fi], frames[fi]))
            lpipss.append(calcular_lpips(renders[fi], frames[fi]))

        psnr_avg = float(np.mean(psnrs))
        ssim_avg = float(np.mean(ssims))
        lpips_validos = [l for l in lpipss if l is not None]
        lpips_avg = float(np.mean(lpips_validos)) if lpips_validos else None

        print(f"  {dtype_nombre:10s}: bytes={bytes_directos:8d}  zlib={bytes_zlib:8d}  "
              f"PSNR={psnr_avg:.2f}  SSIM={ssim_avg:.4f}  ratio_zlib={ratio_zlib:.1f}x")

        filas.append([
            dtype_nombre, bytes_directos, bytes_zlib,
            ratio_directo, ratio_zlib,
            psnr_avg, ssim_avg, lpips_avg if lpips_avg is not None else "",
        ])

    guardar_csv(filas,
                 ["dtype", "bytes", "bytes_zlib", "ratio", "ratio_zlib",
                  "PSNR", "SSIM", "LPIPS"],
                 os.path.join(carpeta_salida, "tabla_cuantizacion.csv"))

    # comparativa grafica: barras
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    nombres = [f[0] for f in filas]
    bytes_zlib_arr = [f[2] for f in filas]
    ssim_arr = [f[6] for f in filas]
    ax[0].bar(nombres, bytes_zlib_arr, color='tab:blue')
    ax[0].set_ylabel("bytes (con zlib)")
    ax[0].set_title("tamano comprimido por dtype")
    ax[1].bar(nombres, ssim_arr, color='tab:orange')
    ax[1].set_ylabel("SSIM")
    ax[1].set_title("calidad por dtype")
    ax[1].set_ylim(min(ssim_arr) * 0.95 if ssim_arr else 0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta_salida, "comparativa.png"))
    plt.close(fig)

    escribir_metricas_json(
        os.path.join(carpeta_salida, "metricas.json"),
        exp="exp06_cuantizacion",
        clip=nombre_clip,
        N=N, base=base, q=q,
        dtypes_probados=config["dtypes"],
        tabla=[dict(zip(["dtype", "bytes", "bytes_zlib", "ratio", "ratio_zlib",
                          "PSNR", "SSIM", "LPIPS"], f)) for f in filas],
    )



def main():
    torch.manual_seed(42)

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    clips_dir = os.path.join(raiz, "clips")
    resultados_dir = os.path.join(raiz, "resultados", "exp06_cuantizacion")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"dispositivo: {device}")

    with open(os.path.join(aqui, "config.json")) as f:
        config = json.load(f)

    ruta_exp05 = os.path.abspath(os.path.join(aqui, config["ruta_exp05"]))
    N_obj = config["N_objetivo"]

    for nombre_clip in config["clips"]:
        carpeta_clip = os.path.join(clips_dir, nombre_clip)
        if not os.path.isdir(carpeta_clip):
            print(f"[saltado] clip no encontrado: {carpeta_clip}")
            continue

        ckpt_ruta = os.path.join(ruta_exp05, nombre_clip, f"N={N_obj}", "checkpoint.pt")
        if not os.path.isfile(ckpt_ruta):
            print(f"[saltado] no encuentro checkpoint: {ckpt_ruta}")
            print(f"  (corre antes exp05_numero_gaussianas con N={N_obj} incluido)")
            continue

        salida = os.path.join(resultados_dir, nombre_clip)
        procesar_clip(nombre_clip, carpeta_clip, ckpt_ruta, salida, config, device)



if __name__ == "__main__":
    main()
