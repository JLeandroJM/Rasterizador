"""
Entry point de exp08.

Uso:
    cd experimentos/exp08_incremental_chebyshev
    python correr.py [--config config.json]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chebyshev import construir_matriz_chebyshev
from modelo import GaussianasChebyshev2D
from rasterizador import rasterizar_diferenciable
from optimizador import construir_optimizador
from trainer import entrenar_incremental
from pruning_post import prunear_post_training
from visualizar_trayectorias import (
    generar_trayectorias_png,
    generar_heatmap_opacity,
    generar_evolucion_parametros,
    generar_reconstruccion_gif,
)
from metricas import psnr, calcular_ssim, calcular_lpips, tamano_bytes_coefs, ratio_compresion



def cargar_clip(carpeta_clip, device, max_frames=None):
    """Carga PNG sequence -> tensor (N_frames, H, W, 3) en [0, 1]."""
    archivos = sorted(f for f in os.listdir(carpeta_clip)
                        if f.startswith("frame_") and f.endswith(".png"))
    if max_frames is not None:
        archivos = archivos[:max_frames]
    frames = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr))
    return torch.stack(frames, dim=0).to(device)



def elegir_device(device_str):
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)



def guardar_curva(valores, ylabel, titulo, ruta):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel("indice")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    torch.manual_seed(42)
    # permitir fallback automatico a CPU para ops no soportadas en MPS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))

    with open(os.path.join(aqui, args.config)) as f:
        config = json.load(f)

    device = elegir_device(config["device"])
    print(f"dispositivo: {device}")

    # === cargar clip ========================================================
    clip = config["clip"]
    carpeta_clip = os.path.join(raiz, "clips", clip)
    if not os.path.isdir(carpeta_clip):
        raise FileNotFoundError(f"clip no encontrado: {carpeta_clip}")

    frames = cargar_clip(carpeta_clip, device, max_frames=config.get("max_frames"))
    N_frames, H, W, _ = frames.shape
    print(f"clip: {clip}  N_frames={N_frames}  resolucion={H}x{W}")

    # === carpeta de salida =================================================
    salida = os.path.join(raiz, "resultados", "exp08_incremental_chebyshev", clip)
    os.makedirs(salida, exist_ok=True)

    # === construir modelo y matrices de Chebyshev ==========================
    grados = config["grados"]
    grados_distintos = sorted(set(grados.values()))
    matrices_cheb = {g: construir_matriz_chebyshev(N_frames, g, device=device, dtype=torch.float32)
                     for g in grados_distintos}
    print(f"matrices de Chebyshev construidas para grados: {grados_distintos}")

    modelo = GaussianasChebyshev2D(
        n_gaussianas=config["n_gaussianas_inicial"],
        n_frames=N_frames,
        grados=grados,
        H=H, W=W,
        device=device,
        escala_inicial_px=config["escala_inicial_px"],
        frame_0_imagen=frames[0],
        semilla=42,
    )
    print(f"modelo creado: N={modelo.numero_gausianas()}  grados={grados}")

    # === entrenamiento incremental =========================================
    optimizer = construir_optimizador(modelo, config["lrs"])

    print("\n=== entrenamiento incremental ===")
    historial = entrenar_incremental(modelo, frames, matrices_cheb, optimizer, config,
                                       carpeta_salida=salida)
    t_train = historial["tiempo_total"]
    print(f"\nentrenamiento OK en {t_train:.1f}s  ({t_train/60:.1f}min)")

    # curvas
    losses_planas = [l for sub in historial["losses_por_frame"] for l in sub]
    guardar_curva(losses_planas, "loss", "loss vs iteracion (concatenado)",
                   os.path.join(salida, "loss.png"))
    guardar_curva(historial["psnr_frame0_por_k"], "PSNR(f=0) [dB]",
                   "test de no-olvido: PSNR del frame 0 conforme avanzamos en k",
                   os.path.join(salida, "psnr_frame0_no_olvido.png"))
    guardar_curva(historial["tiempo_por_frame"], "segundos",
                   "tiempo por frame (crece con k bajo replay completo)",
                   os.path.join(salida, "tiempo_por_frame.png"))

    # === metricas pre-pruning =============================================
    print("\n=== metricas pre-pruning ===")
    metricas_pre = evaluar_clip(modelo, frames, matrices_cheb, device)
    print(f"  PSNR={metricas_pre['psnr']:.2f}  SSIM={metricas_pre['ssim']:.4f}")

    sd_pre = modelo.state_dict_coefs()
    bytes_pre = tamano_bytes_coefs(sd_pre)
    ratio_pre = ratio_compresion(N_frames, H, W, bytes_pre)
    print(f"  N={modelo.numero_gausianas()}  bytes={bytes_pre}  ratio={ratio_pre:.1f}x")

    # === pruning post-training ============================================
    print("\n=== pruning post-training ===")
    n_orig, n_final, _ = prunear_post_training(
        modelo,
        umbral=config["umbral_pruning_post"],
        n_samples=200,
    )
    print(f"  N: {n_orig} -> {n_final}  ({n_orig - n_final} eliminadas)")

    metricas_post = evaluar_clip(modelo, frames, matrices_cheb, device)
    print(f"  PSNR_post={metricas_post['psnr']:.2f}  SSIM_post={metricas_post['ssim']:.4f}")

    sd_post = modelo.state_dict_coefs()
    bytes_post = tamano_bytes_coefs(sd_post)
    ratio_post = ratio_compresion(N_frames, H, W, bytes_post)
    print(f"  bytes_post={bytes_post}  ratio_post={ratio_post:.1f}x")

    # guardar checkpoint final
    torch.save({
        "state_dict_coefs": sd_post,
        "config": config,
        "metricas_pre": metricas_pre,
        "metricas_post": metricas_post,
    }, os.path.join(salida, "checkpoint_final.pt"))

    # === visualizaciones ==================================================
    print("\n=== visualizaciones ===")
    generar_trayectorias_png(modelo, frames[0], matrices_cheb,
                              os.path.join(salida, "trayectorias.png"))
    generar_heatmap_opacity(modelo, matrices_cheb,
                             os.path.join(salida, "heatmap_opacity_temporal.png"))
    generar_evolucion_parametros(modelo, matrices_cheb,
                                  os.path.join(salida, "evolucion_parametros.png"))
    generar_reconstruccion_gif(modelo, frames, matrices_cheb,
                                os.path.join(salida, "reconstruccion_vs_original.gif"))

    # === metricas.json final ==============================================
    metricas_json = {
        "exp": "exp08_incremental_chebyshev",
        "clip": clip,
        "n_frames": N_frames,
        "resolucion": [H, W],
        "config_usado": config,
        "tiempo_entrenamiento_seg": t_train,
        "pre_pruning": {
            "N": n_orig,
            "psnr": metricas_pre["psnr"],
            "ssim": metricas_pre["ssim"],
            "lpips": metricas_pre.get("lpips"),
            "bytes_modelo": bytes_pre,
            "ratio_compresion": ratio_pre,
        },
        "post_pruning": {
            "N": n_final,
            "psnr": metricas_post["psnr"],
            "ssim": metricas_post["ssim"],
            "lpips": metricas_post.get("lpips"),
            "bytes_modelo": bytes_post,
            "ratio_compresion": ratio_post,
        },
        "psnr_frame0_evolucion": historial["psnr_frame0_por_k"],
    }
    with open(os.path.join(salida, "metricas.json"), "w") as f:
        json.dump(metricas_json, f, indent=2, default=str)

    # csv compacto
    with open(os.path.join(salida, "metricas_resumen.csv"), "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["fase", "N", "PSNR", "SSIM", "LPIPS", "bytes", "ratio"])
        for fase, m, nn_, b, r in [
            ("pre",  metricas_pre,  n_orig,  bytes_pre,  ratio_pre),
            ("post", metricas_post, n_final, bytes_post, ratio_post),
        ]:
            w.writerow([fase, nn_,
                         f"{m['psnr']:.3f}", f"{m['ssim']:.4f}",
                         f"{m['lpips']:.4f}" if m.get('lpips') is not None else "",
                         b, f"{r:.2f}"])

    print(f"\nlisto. resultados en: {salida}")



@torch.no_grad()
def evaluar_clip(modelo, frames, matrices_cheb, device):
    """Calcula PSNR/SSIM/LPIPS promedio sobre todos los frames."""
    N_frames, H, W, _ = frames.shape
    psnrs, ssims, lpipss = [], [], []
    for j in range(N_frames):
        params_j = modelo.evaluar_en_frame(j, matrices_cheb)
        render_j = rasterizar_diferenciable(params_j, H, W).clamp(0, 1)
        psnrs.append(psnr(render_j, frames[j]))
        ssims.append(calcular_ssim(render_j, frames[j]))
        l = calcular_lpips(render_j, frames[j], device=device)
        lpipss.append(l)

    lp_validos = [l for l in lpipss if l is not None]
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lp_validos)) if lp_validos else None,
    }



if __name__ == "__main__":
    main()
