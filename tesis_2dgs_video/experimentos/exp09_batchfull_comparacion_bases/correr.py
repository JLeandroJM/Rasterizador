"""
Entry point de exp09.

Uso (Windows + RTX 4050):
    python correr.py --config configs/config_chebyshev.json
    python correr.py --config configs/config_monomial.json
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

from bases import construir_matriz
from modelo import GaussianasPolinomial2D
from rasterizador import rasterizar_un_frame
from optimizador import construir_optimizador
from trainer import entrenar_batch_full
from pruning_post import prunear_post
from preparar_video import extraer_frames_de_video
from metricas_calidad import reporte_completo
from metricas_compresion import reporte_compresion
from visualizar_trayectorias import (
    generar_trayectorias_png,
    generar_heatmap_opacity,
    generar_evolucion_parametros,
    generar_coeficientes_magnitudes,
    generar_reconstruccion_gif,
)



def cargar_clip(carpeta_clip, device, max_frames=None):
    archivos = sorted(f for f in os.listdir(carpeta_clip)
                        if f.startswith("frame_") and f.endswith(".png"))
    if max_frames is not None:
        archivos = archivos[:max_frames]
    fr = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta_clip, nombre)).convert("RGB")
        fr.append(torch.from_numpy(np.array(img, dtype=np.float32) / 255.0))
    return torch.stack(fr, dim=0).to(device)



def elegir_device(device_str):
    if device_str == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: device='cuda' pero torch.cuda.is_available()==False",
                  file=sys.stderr, flush=True)
            print(f"  torch version: {torch.__version__}", file=sys.stderr)
            print(f"  cuda compiled: {torch.version.cuda}", file=sys.stderr)
            sys.exit(2)
        return torch.device("cuda")
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            print("ERROR: device='mps' pero MPS no disponible", file=sys.stderr, flush=True)
            sys.exit(2)
        return torch.device("mps")
    if device_str == "cpu":
        return torch.device("cpu")
    raise ValueError(f"device desconocido: {device_str!r}")



def guardar_curva(valores, titulo, ylabel, ruta, xlabel='iteracion'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(valores)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ruta al config json")
    args = parser.parse_args()

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))

    with open(args.config) as f:
        config = json.load(f)

    # reproducibilidad
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = elegir_device(config["device"])
    print(f"dispositivo: {device}", flush=True)
    print(f"base: {config['base']}", flush=True)

    # === preparar clip ======================================================
    # Si config tiene `video_mp4`, extraemos los frames automaticamente desde
    # el mp4 hacia clips/<clip>/ antes de cargar. Si no, asumimos que ya
    # existe clips/<clip>/ con la secuencia PNG.
    clip = config["clip"]
    carpeta_clip = os.path.join(raiz, "clips", clip)

    if config.get("video_mp4"):
        ruta_video = config["video_mp4"]
        # resolvemos la ruta: si es relativa, es relativa a la raiz tesis_2dgs_video
        if not os.path.isabs(ruta_video):
            ruta_video = os.path.join(raiz, ruta_video)

        res = config.get("resolucion_extraccion", [256, 256])
        H_ext, W_ext = int(res[0]), int(res[1])

        n_ext_obj = config.get("n_frames_extraer")           # None = todos
        fps_ext   = config.get("fps_extraccion")             # None = nativo
        forzar    = bool(config.get("forzar_extraccion", False))

        print(f"=== extrayendo frames del video '{ruta_video}' ===", flush=True)
        n_extraidos = extraer_frames_de_video(
            ruta_mp4=ruta_video,
            carpeta_salida=carpeta_clip,
            n_frames=n_ext_obj,
            fps=fps_ext,
            H=H_ext, W=W_ext,
            forzar=forzar,
        )
        print(f"  frames disponibles en {carpeta_clip}: {n_extraidos}", flush=True)

    if not os.path.isdir(carpeta_clip):
        raise FileNotFoundError(
            f"clip no encontrado: {carpeta_clip}.\n"
            f"  - O bien colocaste un mp4 en `video/` y agregaste `video_mp4` al config,\n"
            f"  - o bien existe ya una secuencia PNG en `clips/{clip}/`."
        )

    frames = cargar_clip(carpeta_clip, device, max_frames=config.get("max_frames"))
    n_frames, H, W, _ = frames.shape
    print(f"clip={clip}  n_frames={n_frames}  resolucion={H}x{W}", flush=True)

    # === salida =============================================================
    nombre_exp = config["nombre_experimento"]
    salida = os.path.join(raiz, "resultados", "exp09", nombre_exp, clip)
    os.makedirs(salida, exist_ok=True)

    # === modelo y matrices de la base ======================================
    base = config["base"]
    grados = config["grados"]
    grados_distintos = sorted(set(grados.values()))
    matrices_base = {
        g: construir_matriz(base, n_frames, g, device=device, dtype=torch.float32)
        for g in grados_distintos
    }
    print(f"matrices ({base}) construidas para grados: {grados_distintos}", flush=True)

    modelo = GaussianasPolinomial2D(
        n_gaussianas=config["n_gaussianas_inicial"],
        n_frames=n_frames,
        grados=grados,
        base=base,
        H=H, W=W,
        device=device,
        escala_inicial_px=config["escala_inicial_px"],
        frame_0_imagen=frames[0],
        semilla=seed,
    )
    print(f"modelo: N={modelo.numero_gausianas()}  grados={grados}", flush=True)

    optimizer = construir_optimizador(modelo, config["lrs"])

    # === entrenamiento batch-full =========================================
    print(f"\n=== entrenamiento ({config['n_epochs']} epochs) ===", flush=True)
    historial = entrenar_batch_full(modelo, frames, matrices_base, optimizer, config,
                                       carpeta_salida=salida)
    t_train = historial["tiempo_total"]
    print(f"entrenamiento listo en {t_train:.1f}s  ({t_train/60:.1f}min)", flush=True)

    # === curvas =============================================================
    guardar_curva(historial["losses_render"], f"loss_render — {nombre_exp}",
                   "loss render", os.path.join(salida, "loss_curve.png"),
                   xlabel="epoch")
    guardar_curva(historial["losses_smooth"], f"loss_smoothness — {nombre_exp}",
                   "loss smooth", os.path.join(salida, "loss_smooth_curve.png"),
                   xlabel="epoch")
    # log csv
    with open(os.path.join(salida, "log_entrenamiento.csv"), "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["epoch", "loss_render", "loss_smooth", "tiempo_seg"])
        for i, (lr_, ls_, ts_) in enumerate(zip(
                historial["losses_render"], historial["losses_smooth"], historial["tiempos_por_epoch"])):
            w.writerow([i + 1, f"{lr_:.6f}", f"{ls_:.6f}", f"{ts_:.3f}"])

    # checkpoint final pre-pruning
    torch.save({
        'state_dict_coefs': modelo.state_dict_coefs(),
        'config': config,
    }, os.path.join(salida, "checkpoint_final.pt"))

    # === metricas pre-pruning ===============================================
    print("\n=== metricas pre-pruning ===", flush=True)
    render_pre_list = []
    with torch.no_grad():
        for j in range(n_frames):
            params_j = modelo.evaluar_en_frame(j, matrices_base)
            r = rasterizar_un_frame(params_j, H, W).clamp(0, 1)
            render_pre_list.append(r)
    render_pre = torch.stack(render_pre_list, dim=0)
    rep_pre = reporte_completo(render_pre, frames, device=device)
    print(f"  PSNR={rep_pre['psnr_promedio']:.2f}  SSIM={rep_pre['ssim_promedio']:.4f}",
          flush=True)

    # === pruning post-training ==============================================
    print("\n=== pruning post-training ===", flush=True)
    n_orig, n_final, _ = prunear_post(modelo, base,
                                        umbral=config["umbral_pruning_post"],
                                        n_samples=200)
    print(f"  N: {n_orig} -> {n_final}", flush=True)

    # render post-pruning
    render_post_list = []
    with torch.no_grad():
        for j in range(n_frames):
            params_j = modelo.evaluar_en_frame(j, matrices_base)
            r = rasterizar_un_frame(params_j, H, W).clamp(0, 1)
            render_post_list.append(r)
    render_post = torch.stack(render_post_list, dim=0)
    rep_post = reporte_completo(render_post, frames, device=device)
    print(f"  PSNR_post={rep_post['psnr_promedio']:.2f}  SSIM_post={rep_post['ssim_promedio']:.4f}",
          flush=True)

    torch.save({
        'state_dict_coefs': modelo.state_dict_coefs(),
        'config': config,
        'metricas_pre': rep_pre,
        'metricas_post': rep_post,
    }, os.path.join(salida, "modelo_pruneado.pt"))

    # guardar frames rasterizados (post-pruning) como PNG
    carpeta_frames_recon = os.path.join(salida, "frames_rasterizados")
    os.makedirs(carpeta_frames_recon, exist_ok=True)
    render_np = render_post.detach().clamp(0, 1).cpu().numpy()
    for j in range(n_frames):
        Image.fromarray((render_np[j] * 255).astype(np.uint8)).save(
            os.path.join(carpeta_frames_recon, f"frame_{j:04d}.png"))

    # === metricas de calidad json ==========================================
    with open(os.path.join(salida, "metricas_calidad.json"), "w") as f:
        json.dump({
            "exp": nombre_exp,
            "base": base,
            "clip": clip,
            "n_frames": n_frames,
            "resolucion": [H, W],
            "pre_pruning": {
                "N": n_orig,
                "psnr_promedio": rep_pre['psnr_promedio'],
                "ssim_promedio": rep_pre['ssim_promedio'],
                "lpips_promedio": rep_pre['lpips_promedio'],
                "psnr_por_frame": rep_pre['psnr_por_frame'],
                "ssim_por_frame": rep_pre['ssim_por_frame'],
            },
            "post_pruning": {
                "N": n_final,
                "psnr_promedio": rep_post['psnr_promedio'],
                "ssim_promedio": rep_post['ssim_promedio'],
                "lpips_promedio": rep_post['lpips_promedio'],
                "psnr_por_frame": rep_post['psnr_por_frame'],
                "ssim_por_frame": rep_post['ssim_por_frame'],
            },
        }, f, indent=2, default=str)

    # === metricas de compresion (AVIF) =====================================
    print("\n=== metricas de compresion ===", flush=True)
    frames_np = (frames.detach().cpu().numpy() * 255).astype(np.uint8)
    render_np_u8 = (render_np * 255).astype(np.uint8)
    rep_comp = reporte_compresion(
        modelo, render_np_u8, frames_np,
        ruta_video_original=carpeta_clip,
        calidades_avif=tuple(config.get("calidades_avif", [80, 95])),
        carpeta_avif_originales=os.path.join(salida, "frames_originales_avif"),
        carpeta_avif_rasterizados=os.path.join(salida, "frames_rasterizados_avif"),
    )
    print(f"  bytes_modelo = {rep_comp['tamano_modelo_bytes']}  "
          f"({rep_comp['tamano_modelo_kb']:.1f} KiB)", flush=True)
    print(f"  bytes_video_original = {rep_comp['tamano_video_original_bytes']}", flush=True)
    if rep_comp.get('avif_disponible', False):
        for q in config.get("calidades_avif", [80, 95]):
            sz_o = rep_comp['avif_originales_por_calidad'][q]['total_bytes']
            sz_r = rep_comp['avif_rasterizados_por_calidad'][q]['total_bytes']
            print(f"  AVIF q={q}: originales={sz_o} bytes  rasterizados={sz_r} bytes",
                  flush=True)
    else:
        print("  AVIF deshabilitado (pillow-avif-plugin no instalado).", flush=True)

    with open(os.path.join(salida, "metricas_compresion.json"), "w") as f:
        json.dump(rep_comp, f, indent=2, default=str)

    # === visualizaciones ===================================================
    print("\n=== visualizaciones ===", flush=True)
    generar_trayectorias_png(modelo, frames[0], matrices_base,
                              os.path.join(salida, "trayectorias.png"))
    generar_heatmap_opacity(modelo, matrices_base,
                             os.path.join(salida, "heatmap_opacity_temporal.png"))
    generar_evolucion_parametros(modelo, matrices_base,
                                  os.path.join(salida, "evolucion_parametros.png"))
    generar_coeficientes_magnitudes(modelo,
                                      os.path.join(salida, "coeficientes_magnitudes.png"))
    generar_reconstruccion_gif(modelo, frames, matrices_base,
                                os.path.join(salida, "reconstruccion_vs_original.gif"))

    print(f"\nlisto. resultados en: {salida}", flush=True)


if __name__ == "__main__":
    main()
