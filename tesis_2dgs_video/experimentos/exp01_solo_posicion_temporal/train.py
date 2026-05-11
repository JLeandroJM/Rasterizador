"""
exp01 — entrena UN modelo temporal con centro polinomico sobre cada clip
(no por frame). Reporta metricas por frame y agregadas.

Uso:
    cd experimentos/exp01_solo_posicion_temporal
    python train.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modelo import Modelo2DGSTemporalPos
from rasterizador import rasterizar_diferenciable, clampear_escala
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



def crear_optimizador(modelo, lrs):
    # NOTA: centro_coefs reemplaza a centro; los demas son iguales que en exp00
    return torch.optim.Adam([
        {'params': [modelo.centro_coefs], 'lr': lrs['centro']},
        {'params': [modelo.escala],       'lr': lrs['escala']},
        {'params': [modelo.theta],        'lr': lrs['theta']},
        {'params': [modelo.opacidad],     'lr': lrs['opacidad']},
        {'params': [modelo.color],        'lr': lrs['color']},
    ])



def procesar_clip(nombre_clip, carpeta_clip, carpeta_salida, config, device):
    print(f"\n===== procesando clip: {nombre_clip} =====")

    frames = cargar_clip(carpeta_clip, device)
    T, H, W, _ = frames.shape

    N = config["N"]
    q = config["q"]
    iteraciones = config["iteraciones"]
    lrs = config["lrs"]
    lambda_dssim = config["lambda_dssim"]

    asegurar_carpeta(carpeta_salida)
    carpeta_recon = os.path.join(carpeta_salida, "reconstruido")
    asegurar_carpeta(carpeta_recon)

    modelo = Modelo2DGSTemporalPos(N, q, H, W, device,
                                    escala_inicial=config["escala_inicial"],
                                    semilla=config["semilla"])
    opt = crear_optimizador(modelo, lrs)

    losses = []
    t_ini = time.time()
    for it in range(iteraciones):
        # sampleo un frame aleatorio del clip
        fi = torch.randint(0, T, ()).item()
        t_norm = fi / (T - 1) if T > 1 else 0.0
        target = frames[fi]

        opt.zero_grad()
        c, e, th, o, k, p = modelo.evaluar_en_t(t_norm)
        render = rasterizar_diferenciable(c, e, th, o, k, p, H, W)
        loss = calcular_loss(render, target, lambda_dssim=lambda_dssim)
        loss.backward()
        opt.step()
        clampear_escala(modelo, H, W)

        losses.append(float(loss.item()))

        if (it + 1) % 500 == 0:
            print(f"  it {it+1:4d}/{iteraciones}  loss={loss.item():.4f}  ({time.time()-t_ini:.1f}s)")

    t_train = time.time() - t_ini

    # evaluacion sobre todos los frames (orden temporal)
    print("  evaluando metricas por frame...")
    psnrs, ssims, lpipss = [], [], []
    reconstruidos = []
    with torch.no_grad():
        for fi in range(T):
            t_norm = fi / (T - 1) if T > 1 else 0.0
            c, e, th, o, k, p = modelo.evaluar_en_t(t_norm)
            render = rasterizar_diferenciable(c, e, th, o, k, p, H, W).clamp(0, 1)
            psnrs.append(psnr(render, frames[fi]))
            ssims.append(calcular_ssim(render, frames[fi]))
            lpipss.append(calcular_lpips(render, frames[fi]))
            guardar_frame(render, os.path.join(carpeta_recon, f"frame_{fi:04d}.png"))
            reconstruidos.append(render.cpu())

    t_total = time.time() - t_ini

    # tamano
    sd = modelo.state_dict()
    bytes_modelo = tamano_bytes_state_dict(sd)
    ratio = ratio_compresion(T, H, W, bytes_modelo)

    torch.save({"state_dict": sd, "config": config}, os.path.join(carpeta_salida, "checkpoint.pt"))

    guardar_curva(losses, f"loss vs iteracion — {nombre_clip}", "loss",
                  os.path.join(carpeta_salida, "loss.png"))
    guardar_csv([[i, l] for i, l in enumerate(losses)], ["iteracion", "loss"],
                os.path.join(carpeta_salida, "loss.csv"))
    guardar_metricas_por_frame_png(psnrs, ssims, lpipss,
                                    os.path.join(carpeta_salida, "metricas_por_frame.png"))
    guardar_csv(
        [[i, p_, s_, l_ if l_ is not None else ""] for i, (p_, s_, l_) in enumerate(zip(psnrs, ssims, lpipss))],
        ["frame_idx", "PSNR", "SSIM", "LPIPS"],
        os.path.join(carpeta_salida, "metricas_por_frame.csv"),
    )

    originales_cpu = [frames[i].cpu() for i in range(T)]
    guardar_gif(reconstruidos, os.path.join(carpeta_salida, "reconstruido.gif"), paso=3)
    guardar_gif_comparativa(originales_cpu, reconstruidos,
                             os.path.join(carpeta_salida, "comparativa.gif"), paso=3)

    psnr_avg = float(np.mean(psnrs))
    ssim_avg = float(np.mean(ssims))
    lpips_validos = [l for l in lpipss if l is not None]
    lpips_avg = float(np.mean(lpips_validos)) if lpips_validos else None

    escribir_metricas_json(
        os.path.join(carpeta_salida, "metricas.json"),
        exp="exp01_solo_posicion_temporal",
        clip=nombre_clip,
        num_frames=T,
        resolucion=[H, W],
        N=N, q=q, iteraciones=iteraciones,
        psnr_promedio=psnr_avg,
        ssim_promedio=ssim_avg,
        lpips_promedio=lpips_avg,
        bytes_modelo=bytes_modelo,
        ratio_compresion=ratio,
        tiempo_entrenamiento_seg=t_train,
        tiempo_total_seg=t_total,
    )

    print(f"  done. PSNR_avg={psnr_avg:.2f} SSIM_avg={ssim_avg:.4f} ratio={ratio:.2f}x ({t_total:.1f}s)")



def main():
    torch.manual_seed(42)

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    clips_dir = os.path.join(raiz, "clips")
    resultados_dir = os.path.join(raiz, "resultados", "exp01_solo_posicion_temporal")

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
