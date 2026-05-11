"""
exp00 — entrena un modelo 2DGS estatico INDEPENDIENTE por frame y reporta
metricas agregadas sobre cada clip.

Uso:
    cd experimentos/exp00_baseline_estatico
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

from modelo import Modelo2DGSEstatico
from rasterizador import rasterizar_diferenciable, clampear_escala
from metricas import (
    psnr, calcular_ssim, calcular_lpips, calcular_loss,
    tamano_bytes_state_dict, ratio_compresion,
    asegurar_carpeta, guardar_curva, guardar_csv, guardar_frame,
    guardar_secuencia, guardar_gif, guardar_gif_comparativa,
    guardar_metricas_por_frame_png, escribir_metricas_json,
)



def cargar_clip(carpeta_clip, device):
    """Lee todos los frame_*.png en orden y devuelve tensor (T, H, W, 3) en [0, 1]."""
    archivos = sorted(f for f in os.listdir(carpeta_clip) if f.startswith("frame_") and f.endswith(".png"))
    frames = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).to(device))
    return torch.stack(frames, dim=0)



def crear_optimizador(modelo, lrs):
    return torch.optim.Adam([
        {'params': [modelo.centro],   'lr': lrs['centro']},
        {'params': [modelo.escala],   'lr': lrs['escala']},
        {'params': [modelo.theta],    'lr': lrs['theta']},
        {'params': [modelo.opacidad], 'lr': lrs['opacidad']},
        {'params': [modelo.color],    'lr': lrs['color']},
    ])



def entrenar_un_frame(target, N, iteraciones, alto, ancho, device,
                       lrs, escala_inicial, semilla, lambda_dssim=0.2):
    """Entrena un modelo estatico para un solo frame y devuelve modelo + losses."""
    modelo = Modelo2DGSEstatico(N, alto, ancho, device,
                                 escala_inicial=escala_inicial, semilla=semilla)
    opt = crear_optimizador(modelo, lrs)
    losses_frame = []

    for it in range(iteraciones):
        opt.zero_grad()
        c, e, t, o, k, p = modelo.evaluar()
        render = rasterizar_diferenciable(c, e, t, o, k, p, alto, ancho)
        loss = calcular_loss(render, target, lambda_dssim=lambda_dssim)
        loss.backward()
        opt.step()
        clampear_escala(modelo, alto, ancho)
        losses_frame.append(float(loss.item()))

    return modelo, losses_frame



def procesar_clip(nombre_clip, carpeta_clip, carpeta_salida, config, device):
    print(f"\n===== procesando clip: {nombre_clip} =====")

    frames = cargar_clip(carpeta_clip, device)
    T, H, W, _ = frames.shape

    N = config["N"]
    iteraciones = config["iteraciones_por_frame"]
    lrs = config["lrs"]

    asegurar_carpeta(carpeta_salida)
    carpeta_recon = os.path.join(carpeta_salida, "reconstruido")
    asegurar_carpeta(carpeta_recon)

    losses_global = []
    psnrs, ssims, lpipss = [], [], []
    reconstruidos = []
    state_dicts = []

    t_ini = time.time()
    for fi in range(T):
        target = frames[fi]
        modelo, losses_frame = entrenar_un_frame(
            target, N, iteraciones, H, W, device,
            lrs, config["escala_inicial"],
            semilla=config["semilla"] + fi,    # semilla distinta por frame
            lambda_dssim=config["lambda_dssim"],
        )
        losses_global.extend(losses_frame)

        # render final del frame
        with torch.no_grad():
            c, e, t, o, k, p = modelo.evaluar()
            render = rasterizar_diferenciable(c, e, t, o, k, p, H, W).clamp(0, 1)

        psnrs.append(psnr(render, target))
        ssims.append(calcular_ssim(render, target))
        lpipss.append(calcular_lpips(render, target))

        guardar_frame(render, os.path.join(carpeta_recon, f"frame_{fi:04d}.png"))
        reconstruidos.append(render.cpu())
        state_dicts.append(modelo.state_dict())

        if (fi + 1) % 10 == 0:
            print(f"  frame {fi+1:3d}/{T}  PSNR={psnrs[-1]:.2f}  SSIM={ssims[-1]:.4f}  ({time.time()-t_ini:.1f}s)")

    t_total = time.time() - t_ini

    # tamano: suma de bytes de todos los state_dicts
    bytes_modelo = sum(tamano_bytes_state_dict(sd) for sd in state_dicts)
    ratio = ratio_compresion(T, H, W, bytes_modelo)

    # guardar checkpoint
    torch.save({"frames": state_dicts, "config": config}, os.path.join(carpeta_salida, "checkpoint.pt"))

    # curvas
    guardar_curva(losses_global, f"loss acumulada (todos los frames) — {nombre_clip}", "loss",
                  os.path.join(carpeta_salida, "loss.png"))
    guardar_csv([[i, l] for i, l in enumerate(losses_global)], ["it_global", "loss"],
                os.path.join(carpeta_salida, "loss.csv"))
    guardar_metricas_por_frame_png(psnrs, ssims, lpipss,
                                    os.path.join(carpeta_salida, "metricas_por_frame.png"))
    guardar_csv(
        [[i, p_, s_, l_ if l_ is not None else ""] for i, (p_, s_, l_) in enumerate(zip(psnrs, ssims, lpipss))],
        ["frame_idx", "PSNR", "SSIM", "LPIPS"],
        os.path.join(carpeta_salida, "metricas_por_frame.csv"),
    )

    # videos
    originales_cpu = [frames[i].cpu() for i in range(T)]
    guardar_gif(reconstruidos, os.path.join(carpeta_salida, "reconstruido.gif"), paso=3)
    guardar_gif_comparativa(originales_cpu, reconstruidos,
                             os.path.join(carpeta_salida, "comparativa.gif"), paso=3)

    # metricas.json
    psnr_avg = float(np.mean(psnrs))
    ssim_avg = float(np.mean(ssims))
    lpips_validos = [l for l in lpipss if l is not None]
    lpips_avg = float(np.mean(lpips_validos)) if lpips_validos else None
    escribir_metricas_json(
        os.path.join(carpeta_salida, "metricas.json"),
        exp="exp00_baseline_estatico",
        clip=nombre_clip,
        num_frames=T,
        resolucion=[H, W],
        N=N,
        iteraciones_por_frame=iteraciones,
        psnr_promedio=psnr_avg,
        ssim_promedio=ssim_avg,
        lpips_promedio=lpips_avg,
        bytes_modelo=bytes_modelo,
        ratio_compresion=ratio,
        tiempo_total_seg=t_total,
    )

    print(f"  done. PSNR_avg={psnr_avg:.2f} SSIM_avg={ssim_avg:.4f} ratio={ratio:.2f}x ({t_total:.1f}s)")



def main():
    torch.manual_seed(42)

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    clips_dir = os.path.join(raiz, "clips")
    resultados_dir = os.path.join(raiz, "resultados", "exp00_baseline_estatico")

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
