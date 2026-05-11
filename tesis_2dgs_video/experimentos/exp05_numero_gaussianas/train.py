"""
exp05 — barrido del numero N de gaussianas (curva Rate-Distortion).

Para cada clip, entrena un modelo (de la base configurada, con q fijo) con
cada N en config["Ns"] y reporta PSNR/SSIM/LPIPS vs bytes_modelo.

Uso:
    cd experimentos/exp05_numero_gaussianas
    python train.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelos import construir_modelo_por_base
from rasterizador import rasterizar_diferenciable
from metricas import (
    psnr, calcular_ssim, calcular_lpips, calcular_loss,
    tamano_bytes_state_dict, ratio_compresion,
    asegurar_carpeta, guardar_curva, guardar_csv, guardar_frame,
    guardar_gif, guardar_gif_comparativa,
    guardar_metricas_por_frame_png, escribir_metricas_json,
)



def cargar_clip(carpeta_clip, device):
    archivos = sorted(f for f in os.listdir(carpeta_clip) if f.startswith("frame_") and f.endswith(".png"))
    frames = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).to(device))
    return torch.stack(frames, dim=0)



def entrenar_una_combo(frames, base, N, q, iteraciones, lrs, lambda_dssim,
                        escala_inicial, semilla, device):
    T, H, W, _ = frames.shape
    modelo = construir_modelo_por_base(base, N, q, H, W, device,
                                         escala_inicial=escala_inicial,
                                         semilla=semilla)
    opt = torch.optim.Adam(modelo.grupos_optimizer(lrs))

    losses = []
    t_ini = time.time()
    for it in range(iteraciones):
        fi = torch.randint(0, T, ()).item()
        t_norm = fi / (T - 1) if T > 1 else 0.0
        target = frames[fi]

        opt.zero_grad()
        c, e, th, o_raw, k, p = modelo.evaluar_en_t(t_norm)
        render = rasterizar_diferenciable(c, e, th, o_raw, k, p, H, W)
        loss = calcular_loss(render, target, lambda_dssim=lambda_dssim)
        loss.backward()
        opt.step()
        modelo.post_step()
        losses.append(float(loss.item()))

    t_train = time.time() - t_ini

    psnrs, ssims, lpipss = [], [], []
    reconstruidos = []
    with torch.no_grad():
        for fi in range(T):
            t_norm = fi / (T - 1) if T > 1 else 0.0
            c, e, th, o_raw, k, p = modelo.evaluar_en_t(t_norm)
            render = rasterizar_diferenciable(c, e, th, o_raw, k, p, H, W).clamp(0, 1)
            psnrs.append(psnr(render, frames[fi]))
            ssims.append(calcular_ssim(render, frames[fi]))
            lpipss.append(calcular_lpips(render, frames[fi]))
            reconstruidos.append(render.cpu())

    return modelo, losses, psnrs, ssims, lpipss, reconstruidos, t_train



def graficar_rd_N(bytes_modelos, psnrs_avg, ssims_avg, Ns, ruta, titulo):
    """Curva rate-distortion: bytes en X (log), PSNR/SSIM en Y."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    ax[0].plot(bytes_modelos, psnrs_avg, 'o-')
    for x, y, n in zip(bytes_modelos, psnrs_avg, Ns):
        ax[0].annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax[0].set_xlabel("bytes modelo")
    ax[0].set_ylabel("PSNR (dB)")
    ax[0].set_xscale("log")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(bytes_modelos, ssims_avg, 'o-', color='tab:orange')
    for x, y, n in zip(bytes_modelos, ssims_avg, Ns):
        ax[1].annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax[1].set_xlabel("bytes modelo")
    ax[1].set_ylabel("SSIM")
    ax[1].set_xscale("log")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



def procesar_clip(nombre_clip, carpeta_clip, carpeta_salida, config, device):
    print(f"\n===== procesando clip: {nombre_clip} =====")
    frames = cargar_clip(carpeta_clip, device)
    T, H, W, _ = frames.shape

    base = config["base"]
    q = config["q"]
    Ns = config["Ns"]
    lrs = config["lrs"]
    iteraciones = config["iteraciones"]
    lambda_dssim = config["lambda_dssim"]

    asegurar_carpeta(carpeta_salida)

    tabla = []
    psnrs_avg, ssims_avg, lpipss_avg, bytes_modelos = [], [], [], []

    for N in Ns:
        print(f"\n  --- N = {N} ---")
        sub_carpeta = os.path.join(carpeta_salida, f"N={N}")
        asegurar_carpeta(sub_carpeta)
        carpeta_recon = os.path.join(sub_carpeta, "reconstruido")
        asegurar_carpeta(carpeta_recon)

        modelo, losses, psnrs, ssims, lpipss, reconstruidos, t_train = entrenar_una_combo(
            frames, base, N, q, iteraciones, lrs, lambda_dssim,
            config["escala_inicial"], config["semilla"], device
        )

        for fi, f in enumerate(reconstruidos):
            guardar_frame(f, os.path.join(carpeta_recon, f"frame_{fi:04d}.png"))

        sd = modelo.state_dict()
        bytes_modelo = tamano_bytes_state_dict(sd)
        ratio = ratio_compresion(T, H, W, bytes_modelo)

        torch.save({"state_dict": sd, "config": config, "N": N},
                   os.path.join(sub_carpeta, "checkpoint.pt"))

        guardar_curva(losses, f"loss N={N}", "loss", os.path.join(sub_carpeta, "loss.png"))
        guardar_csv([[i, l] for i, l in enumerate(losses)], ["iteracion", "loss"],
                    os.path.join(sub_carpeta, "loss.csv"))
        guardar_metricas_por_frame_png(psnrs, ssims, lpipss,
                                        os.path.join(sub_carpeta, "metricas_por_frame.png"))
        guardar_csv(
            [[i, p_, s_, l_ if l_ is not None else ""] for i, (p_, s_, l_) in enumerate(zip(psnrs, ssims, lpipss))],
            ["frame_idx", "PSNR", "SSIM", "LPIPS"],
            os.path.join(sub_carpeta, "metricas_por_frame.csv"),
        )
        guardar_gif(reconstruidos, os.path.join(sub_carpeta, "reconstruido.gif"), paso=3)

        psnr_avg = float(np.mean(psnrs))
        ssim_avg = float(np.mean(ssims))
        lpips_validos = [l for l in lpipss if l is not None]
        lpips_avg = float(np.mean(lpips_validos)) if lpips_validos else None

        psnrs_avg.append(psnr_avg)
        ssims_avg.append(ssim_avg)
        lpipss_avg.append(lpips_avg)
        bytes_modelos.append(bytes_modelo)
        tabla.append([N, psnr_avg, ssim_avg, lpips_avg if lpips_avg is not None else "", bytes_modelo, ratio, t_train])

        escribir_metricas_json(
            os.path.join(sub_carpeta, "metricas.json"),
            exp="exp05_numero_gaussianas", base=base, clip=nombre_clip,
            num_frames=T, resolucion=[H, W],
            N=N, q=q, iteraciones=iteraciones,
            psnr_promedio=psnr_avg, ssim_promedio=ssim_avg, lpips_promedio=lpips_avg,
            bytes_modelo=bytes_modelo, ratio_compresion=ratio,
            tiempo_entrenamiento_seg=t_train,
        )
        print(f"    N={N} PSNR={psnr_avg:.2f} SSIM={ssim_avg:.4f} bytes={bytes_modelo} ratio={ratio:.1f}x")

    guardar_csv(tabla, ["N", "PSNR", "SSIM", "LPIPS", "bytes", "ratio", "tiempo_seg"],
                 os.path.join(carpeta_salida, "rd_table.csv"))
    graficar_rd_N(bytes_modelos, psnrs_avg, ssims_avg, Ns,
                   os.path.join(carpeta_salida, "rd_curve.png"),
                   f"Rate-Distortion vs N — {nombre_clip} — base={base}, q={q}")

    escribir_metricas_json(
        os.path.join(carpeta_salida, "metricas.json"),
        exp="exp05_numero_gaussianas",
        base=base, clip=nombre_clip,
        Ns=Ns, q=q,
        psnr_promedio_por_N=psnrs_avg,
        ssim_promedio_por_N=ssims_avg,
        lpips_promedio_por_N=lpipss_avg,
        bytes_modelo_por_N=bytes_modelos,
    )



def main():
    torch.manual_seed(42)

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    clips_dir = os.path.join(raiz, "clips")
    resultados_dir = os.path.join(raiz, "resultados", "exp05_numero_gaussianas")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"dispositivo: {device}")

    with open(os.path.join(aqui, "config.json")) as f:
        config = json.load(f)

    for nombre_clip in config["clips"]:
        carpeta_clip = os.path.join(clips_dir, nombre_clip)
        if not os.path.isdir(carpeta_clip):
            print(f"[saltado] clip no encontrado: {carpeta_clip}")
            continue
        salida = os.path.join(resultados_dir, nombre_clip)
        procesar_clip(nombre_clip, carpeta_clip, salida, config, device)



if __name__ == "__main__":
    main()
