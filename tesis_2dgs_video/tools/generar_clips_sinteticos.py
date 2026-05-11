"""
Genera los 5 clips sinteticos de Nivel 1 (movimiento suave, escena estable).

Cada clip dura 3 segundos a 30 fps -> 90 frames, resolucion 256x256.
Salida: clips/<nombre_clip>/frame_0000.png, ..., frame_0089.png + clip_info.json

Uso:
    python tools/generar_clips_sinteticos.py
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw


# DECISION: 90 frames a 30 fps = 3 segundos. Suficiente para mostrar movimiento
# polinomico pero no tan largo como para que el training sea pesado.
NUM_FRAMES = 90
FPS = 30
RES = 256


def asegurar_carpeta(ruta):
    os.makedirs(ruta, exist_ok=True)


def guardar_clip(carpeta, frames, nombre):
    """frames: lista de arrays (H, W, 3) en uint8."""
    asegurar_carpeta(carpeta)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(carpeta, f"frame_{i:04d}.png"))

    info = {
        "nombre": nombre,
        "num_frames": len(frames),
        "fps": FPS,
        "resolucion": [RES, RES],
    }
    with open(os.path.join(carpeta, "clip_info.json"), "w") as f:
        json.dump(info, f, indent=2)



# === clip 1: bouncing_ball =================================================
# pelota de color que rebota contra paredes
def generar_bouncing_ball():
    frames = []
    pos = np.array([RES * 0.3, RES * 0.5])
    vel = np.array([3.5, -2.2])
    radio = 18
    color_pelota = (220, 80, 80)
    color_fondo = (30, 30, 40)

    for _ in range(NUM_FRAMES):
        img = Image.new("RGB", (RES, RES), color_fondo)
        d = ImageDraw.Draw(img)
        # rebote contra paredes con leve perdida (suavidad)
        for dim in range(2):
            limite_min = radio
            limite_max = RES - radio
            if pos[dim] < limite_min:
                pos[dim] = limite_min
                vel[dim] = abs(vel[dim])
            elif pos[dim] > limite_max:
                pos[dim] = limite_max
                vel[dim] = -abs(vel[dim])
        d.ellipse(
            (pos[0] - radio, pos[1] - radio, pos[0] + radio, pos[1] + radio),
            fill=color_pelota
        )
        frames.append(np.array(img))
        pos = pos + vel
    return frames



# === clip 2: drifting_circles =============================================
# 3 circulos con trayectorias suaves diferentes (senoidales con frecuencias distintas)
def generar_drifting_circles():
    frames = []
    color_fondo = (15, 25, 35)

    circulos = [
        # (color, A_x, A_y, w_x, w_y, fase_x, fase_y, radio)
        ((230, 100, 100), 0.30, 0.25, 1.0, 1.3, 0.0,  0.4, 22),
        ((100, 200, 130), 0.25, 0.30, 1.5, 0.8, 1.0,  2.0, 18),
        ((110, 130, 230), 0.20, 0.20, 2.0, 1.7, 2.5,  0.1, 24),
    ]

    for fi in range(NUM_FRAMES):
        t = fi / (NUM_FRAMES - 1)         # t en [0, 1]
        img = Image.new("RGB", (RES, RES), color_fondo)
        d = ImageDraw.Draw(img)
        for color, Ax, Ay, wx, wy, phx, phy, r in circulos:
            cx = RES * (0.5 + Ax * np.sin(2 * np.pi * wx * t + phx))
            cy = RES * (0.5 + Ay * np.sin(2 * np.pi * wy * t + phy))
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        frames.append(np.array(img))
    return frames



# === clip 3: rotating_shape ===============================================
# estrella de 5 puntas que rota sobre fondo estable
def generar_rotating_shape():
    frames = []
    color_fondo = (40, 40, 50)
    color_estrella = (240, 220, 80)
    centro = np.array([RES / 2, RES / 2])
    R_ext = 60
    R_int = 25

    for fi in range(NUM_FRAMES):
        # rotacion completa en NUM_FRAMES
        angulo = 2 * np.pi * fi / NUM_FRAMES
        img = Image.new("RGB", (RES, RES), color_fondo)
        d = ImageDraw.Draw(img)
        puntos = []
        for k in range(10):
            r = R_ext if k % 2 == 0 else R_int
            a = angulo + k * np.pi / 5
            puntos.append((centro[0] + r * np.cos(a), centro[1] + r * np.sin(a)))
        d.polygon(puntos, fill=color_estrella)
        frames.append(np.array(img))
    return frames



# === clip 4: fade_transition ==============================================
# dos imagenes (cada una con varios circulos) cross-fade lineal en t
def generar_fade_transition():
    frames = []

    def imagen_A():
        img = Image.new("RGB", (RES, RES), (20, 30, 60))
        d = ImageDraw.Draw(img)
        d.ellipse((40, 60, 100, 120), fill=(230, 100, 100))
        d.ellipse((130, 40, 220, 130), fill=(80, 200, 120))
        return np.array(img, dtype=np.float32)

    def imagen_B():
        img = Image.new("RGB", (RES, RES), (60, 30, 50))
        d = ImageDraw.Draw(img)
        d.rectangle((150, 130, 230, 220), fill=(220, 200, 80))
        d.ellipse((50, 140, 110, 200), fill=(80, 100, 220))
        return np.array(img, dtype=np.float32)

    A = imagen_A()
    B = imagen_B()
    for fi in range(NUM_FRAMES):
        alpha = fi / (NUM_FRAMES - 1)
        mix = ((1 - alpha) * A + alpha * B).clip(0, 255).astype(np.uint8)
        frames.append(mix)
    return frames



# === clip 5: swelling_blobs ===============================================
# 4 blobs gaussianos que crecen y decrecen sin moverse
def generar_swelling_blobs():
    frames = []
    color_fondo = np.array([20, 20, 30], dtype=np.float32)

    # cada blob: (cx, cy, color, sigma_base, sigma_amp, freq, fase)
    blobs = [
        (RES * 0.30, RES * 0.30, (230, 100, 100),  18, 12, 1.0, 0.0),
        (RES * 0.70, RES * 0.30, (100, 230, 120),  18, 12, 1.2, 1.0),
        (RES * 0.30, RES * 0.70, (100, 130, 230),  18, 12, 0.8, 2.0),
        (RES * 0.70, RES * 0.70, (230, 230, 100),  18, 12, 1.4, 0.5),
    ]

    # malla de pixeles
    ys, xs = np.meshgrid(np.arange(RES), np.arange(RES), indexing='ij')

    for fi in range(NUM_FRAMES):
        t = fi / (NUM_FRAMES - 1)
        img = np.tile(color_fondo, (RES, RES, 1))
        for cx, cy, color, sb, sa, freq, fase in blobs:
            sigma = sb + sa * np.sin(2 * np.pi * freq * t + fase)
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            peso = np.exp(-0.5 * d2 / (sigma ** 2))[:, :, None]   # (H, W, 1)
            color_arr = np.array(color, dtype=np.float32)
            img = img * (1 - peso) + color_arr * peso
        frames.append(img.clip(0, 255).astype(np.uint8))
    return frames



def main():
    # ruta de salida: tesis_2dgs_video/clips/
    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz = os.path.abspath(os.path.join(aqui, ".."))
    clips_dir = os.path.join(raiz, "clips")

    clips_a_generar = [
        ("bouncing_ball",     generar_bouncing_ball),
        ("drifting_circles",  generar_drifting_circles),
        ("rotating_shape",    generar_rotating_shape),
        ("fade_transition",   generar_fade_transition),
        ("swelling_blobs",    generar_swelling_blobs),
    ]

    for nombre, generador in clips_a_generar:
        print(f"generando clip: {nombre}")
        frames = generador()
        carpeta = os.path.join(clips_dir, nombre)
        guardar_clip(carpeta, frames, nombre)

    print(f"\nlisto. {len(clips_a_generar)} clips generados en: {clips_dir}")


if __name__ == "__main__":
    main()
