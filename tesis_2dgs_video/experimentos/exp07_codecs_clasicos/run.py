"""
exp07 — comprime los clips con H.264/H.265 a varios CRF, mide metricas y
genera la curva Rate-Distortion para compararla con tu metodo (exp05).

Uso:
    cd experimentos/exp07_codecs_clasicos
    python run.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metricas import (
    psnr, calcular_ssim, calcular_lpips,
    asegurar_carpeta, guardar_csv, escribir_metricas_json,
)



def cargar_frames_png(carpeta, device='cpu'):
    archivos = sorted(f for f in os.listdir(carpeta) if f.startswith("frame_") and f.endswith(".png"))
    frames = []
    for nombre in archivos:
        img = Image.open(os.path.join(carpeta, nombre)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).to(device))
    return torch.stack(frames, dim=0)



def codificar(carpeta_pngs, codec, crf, fps, ruta_mp4_salida):
    """Codifica una PNG sequence usando ffmpeg al codec/crf indicado."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(carpeta_pngs, "frame_%04d.png"),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        ruta_mp4_salida,
    ]
    subprocess.run(cmd, check=True, capture_output=True)



def decodificar(ruta_mp4, carpeta_salida_pngs, fps):
    """Decodifica un mp4 a PNG sequence."""
    asegurar_carpeta(carpeta_salida_pngs)
    cmd = [
        "ffmpeg", "-y",
        "-i", ruta_mp4,
        "-vf", f"fps={fps}",
        os.path.join(carpeta_salida_pngs, "frame_%04d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)



def medir_metricas_clip(originales, decodificados):
    """Compara dos pilas (T, H, W, 3) y devuelve PSNR/SSIM/LPIPS promedio."""
    T = min(originales.shape[0], decodificados.shape[0])
    psnrs, ssims, lpipss = [], [], []
    for fi in range(T):
        psnrs.append(psnr(decodificados[fi], originales[fi]))
        ssims.append(calcular_ssim(decodificados[fi], originales[fi]))
        lpipss.append(calcular_lpips(decodificados[fi], originales[fi]))
    lp_validos = [l for l in lpipss if l is not None]
    return (
        float(np.mean(psnrs)),
        float(np.mean(ssims)),
        float(np.mean(lp_validos)) if lp_validos else None,
    )



def graficar_rd_codecs(filas, ruta, titulo):
    """filas: lista de [codec, crf, bytes, ratio, PSNR, SSIM, LPIPS]."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

    codecs_distintos = sorted(set(f[0] for f in filas))
    for codec in codecs_distintos:
        sub = [f for f in filas if f[0] == codec]
        bytes_arr = [r[2] for r in sub]
        psnr_arr  = [r[4] for r in sub]
        ssim_arr  = [r[5] for r in sub]
        crfs      = [r[1] for r in sub]

        ax[0].plot(bytes_arr, psnr_arr, 'o-', label=codec)
        for x, y, c in zip(bytes_arr, psnr_arr, crfs):
            ax[0].annotate(f"crf={c}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax[1].plot(bytes_arr, ssim_arr, 'o-', label=codec)
        for x, y, c in zip(bytes_arr, ssim_arr, crfs):
            ax[1].annotate(f"crf={c}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    for a, lab in [(ax[0], "PSNR (dB)"), (ax[1], "SSIM")]:
        a.set_xlabel("bytes del video codificado")
        a.set_ylabel(lab)
        a.set_xscale("log")
        a.legend()
        a.grid(True, alpha=0.3)

    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)



def procesar_clip(nombre_clip, carpeta_clip, carpeta_salida, config):
    print(f"\n===== procesando clip: {nombre_clip} =====")

    asegurar_carpeta(carpeta_salida)
    carpeta_videos = os.path.join(carpeta_salida, "videos")
    asegurar_carpeta(carpeta_videos)

    originales = cargar_frames_png(carpeta_clip)
    T, H, W, _ = originales.shape

    filas = []
    for codec in config["codecs"]:
        for crf in config["crfs"]:
            nombre_mp4 = f"{codec}_crf{crf}.mp4"
            ruta_mp4 = os.path.join(carpeta_videos, nombre_mp4)

            print(f"  codificando {codec} crf={crf} ...", end=" ")
            codificar(carpeta_clip, codec, crf, config["fps"], ruta_mp4)
            bytes_video = os.path.getsize(ruta_mp4)

            # decodificar a PNG temporal
            tmp = tempfile.mkdtemp(prefix="decod_")
            try:
                decodificar(ruta_mp4, tmp, config["fps"])
                decodificados = cargar_frames_png(tmp)
                psnr_avg, ssim_avg, lpips_avg = medir_metricas_clip(originales, decodificados)
            finally:
                shutil.rmtree(tmp)

            ratio = (T * H * W * 3) / bytes_video
            filas.append([codec, crf, bytes_video, ratio,
                           psnr_avg, ssim_avg,
                           lpips_avg if lpips_avg is not None else ""])
            print(f"bytes={bytes_video}  PSNR={psnr_avg:.2f}  SSIM={ssim_avg:.4f}")

    guardar_csv(filas,
                 ["codec", "crf", "bytes", "ratio", "PSNR", "SSIM", "LPIPS"],
                 os.path.join(carpeta_salida, "tabla_rd.csv"))
    graficar_rd_codecs(filas, os.path.join(carpeta_salida, "rd_curve.png"),
                        f"Rate-Distortion H.264 / H.265 — {nombre_clip}")

    escribir_metricas_json(
        os.path.join(carpeta_salida, "metricas.json"),
        exp="exp07_codecs_clasicos",
        clip=nombre_clip,
        num_frames=T,
        resolucion=[H, W],
        tabla=[dict(zip(["codec", "crf", "bytes", "ratio", "PSNR", "SSIM", "LPIPS"], f)) for f in filas],
    )



def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, "..", ".."))
    clips_dir = os.path.join(raiz, "clips")
    resultados_dir = os.path.join(raiz, "resultados", "exp07_codecs_clasicos")

    with open(os.path.join(aqui, "config.json")) as f:
        config = json.load(f)

    for nombre_clip in config["clips"]:
        carpeta_clip = os.path.join(clips_dir, nombre_clip)
        if not os.path.isdir(carpeta_clip):
            print(f"[saltado] clip no encontrado: {carpeta_clip}")
            continue
        salida = os.path.join(resultados_dir, nombre_clip)
        procesar_clip(nombre_clip, carpeta_clip, salida, config)



if __name__ == "__main__":
    main()
