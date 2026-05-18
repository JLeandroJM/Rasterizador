import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from chebyshev import construir_matriz_chebyshev
from modelo import GaussianasChebyshev2D
from rasterizador import rasterizar_diferenciable


def elegir_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def cargar_frames(carpeta_clip, device, max_frames=None):
    archivos = sorted(
        f for f in os.listdir(carpeta_clip)
        if f.startswith("frame_") and f.endswith(".png")
    )

    if max_frames is not None:
        archivos = archivos[:max_frames]

    frames = []

    for nombre in archivos:
        ruta = os.path.join(carpeta_clip, nombre)
        img = Image.open(ruta).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr))

    return torch.stack(frames, dim=0).to(device)


def cargar_modelo_desde_checkpoint(ruta_checkpoint, frames, device):
    chk = torch.load(ruta_checkpoint, map_location=device)

    config = chk["config"]
    sd = chk["state_dict_coefs"]
    k_completado = int(chk.get("k_completado", frames.shape[0]))

    n_frames_total, H, W, _ = frames.shape
    grados = config["grados"]
    n_gaussianas = sd["mu_a0"].shape[0]

    modelo = GaussianasChebyshev2D(
        n_gaussianas=n_gaussianas,
        n_frames=n_frames_total,
        grados=grados,
        H=H,
        W=W,
        device=device,
        escala_inicial_px=config.get("escala_inicial_px", 5.0),
        frame_0_imagen=None,
        semilla=42,
    )

    for nombre in ["mu", "opacity", "color", "scale", "theta", "depth"]:
        a0 = sd[f"{nombre}_a0"].to(device)
        hi = sd[f"{nombre}_high"].to(device)

        setattr(modelo, f"{nombre}_a0", nn.Parameter(a0.clone()))
        setattr(modelo, f"{nombre}_high", nn.Parameter(hi.clone()))

    modelo.N = n_gaussianas
    modelo.eval()

    return modelo, config, k_completado


@torch.no_grad()
def generar_frames_desde_checkpoint(modelo, frames, matrices_cheb, salida, k_completado):
    frames_eval = frames[:k_completado]
    N_frames, H, W, _ = frames_eval.shape

    carpeta_recon = salida / "frames_reconstruidos"
    carpeta_comp = salida / "frames_comparacion"

    carpeta_recon.mkdir(parents=True, exist_ok=True)
    carpeta_comp.mkdir(parents=True, exist_ok=True)

    # Limpia frames anteriores si existen
    for carpeta in [carpeta_recon, carpeta_comp]:
        for p in carpeta.glob("frame_*.png"):
            p.unlink()

    for j in range(N_frames):
        target = frames_eval[j]

        params_j = modelo.evaluar_en_frame(j, matrices_cheb)
        render_j = rasterizar_diferenciable(params_j, H, W).clamp(0, 1)

        # 1) frame reconstruido solo
        recon_np = (render_j.detach().cpu().numpy() * 255.0).astype(np.uint8)
        Image.fromarray(recon_np).save(carpeta_recon / f"frame_{j:04d}.png")

        # 2) comparacion: original | reconstruido | diferencia
        target_np = target.detach().cpu().numpy()
        render_np = render_j.detach().cpu().numpy()

        diff = np.clip(np.abs(target_np - render_np) * 5.0, 0, 1)

        comparacion = np.concatenate([target_np, render_np, diff], axis=1)
        comparacion_np = (comparacion * 255.0).astype(np.uint8)

        Image.fromarray(comparacion_np).save(carpeta_comp / f"frame_{j:04d}.png")

        if j == 0 or (j + 1) % 10 == 0 or j == N_frames - 1:
            print(f"generado frame {j + 1}/{N_frames}")

    return carpeta_recon, carpeta_comp


def crear_mp4_desde_frames(carpeta_frames, ruta_mp4, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(carpeta_frames / "frame_%04d.png"),
        "-vf", "scale=720:-2",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        str(ruta_mp4),
    ]

    print()
    print("Creando MP4:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    aqui = Path(__file__).resolve().parent
    raiz = aqui.parent.parent

    ruta_checkpoint = Path(args.checkpoint).resolve()

    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe checkpoint: {ruta_checkpoint}")

    # Cargamos primero en CPU solo para leer config
    chk_cpu = torch.load(ruta_checkpoint, map_location="cpu")
    config = chk_cpu["config"]
    clip = config["clip"]

    device = elegir_device(args.device)
    print("dispositivo:", device)

    carpeta_clip = raiz / "clips" / clip
    if not carpeta_clip.exists():
        raise FileNotFoundError(f"No existe carpeta clip: {carpeta_clip}")

    frames = cargar_frames(
        carpeta_clip,
        device,
        max_frames=config.get("max_frames")
    )

    N_frames, H, W, _ = frames.shape
    print("clip:", clip)
    print("frames cargados:", N_frames)
    print("resolucion:", H, "x", W)

    modelo, config, k_completado = cargar_modelo_desde_checkpoint(
        ruta_checkpoint,
        frames,
        device
    )

    print("checkpoint:", ruta_checkpoint.name)
    print("k_completado:", k_completado)
    print("gaussianas:", modelo.numero_gausianas())

    grados_distintos = sorted(set(config["grados"].values()))
    matrices_cheb = {
        g: construir_matriz_chebyshev(
            N_frames,
            g,
            device=device,
            dtype=torch.float32,
        )
        for g in grados_distintos
    }

    nombre_checkpoint = ruta_checkpoint.stem
    carpeta_resultado = raiz / "resultados" / "exp08_incremental_chebyshev" / clip
    salida = carpeta_resultado / f"preview_{nombre_checkpoint}"
    salida.mkdir(parents=True, exist_ok=True)

    carpeta_recon, carpeta_comp = generar_frames_desde_checkpoint(
        modelo=modelo,
        frames=frames,
        matrices_cheb=matrices_cheb,
        salida=salida,
        k_completado=k_completado,
    )

    ruta_mp4 = salida / "comparacion_preview.mp4"
    crear_mp4_desde_frames(carpeta_comp, ruta_mp4, args.fps)

    print()
    print("Listo.")
    print("Salida:", salida)
    print("Frames reconstruidos:", carpeta_recon)
    print("Frames comparacion:", carpeta_comp)
    print("MP4:", ruta_mp4)


if __name__ == "__main__":
    main()