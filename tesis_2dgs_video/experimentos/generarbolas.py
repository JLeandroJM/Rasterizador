import os
from PIL import Image, ImageDraw

W = 64
H = 64
N = 50

carpeta_salida = r"D:\TesisProyecto\Rasterizador\tesis_2dgs_video\clips\bolitas_dobles"
os.makedirs(carpeta_salida, exist_ok=True)

for i in range(N):
    t = i / (N - 1)

    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # bolita 1
    x1 = int(8 + t * 40)
    y1 = int(8 + t * 40)
    r1 = 5
    draw.ellipse((x1 - r1, y1 - r1, x1 + r1, y1 + r1), fill=(255, 60, 60))

    # bolita 2
    x2 = int(56 - t * 40)
    y2 = int(48 - t * 32)
    r2 = 4
    draw.ellipse((x2 - r2, y2 - r2, x2 + r2, y2 + r2), fill=(60, 180, 255))

    ruta = os.path.join(carpeta_salida, f"frame_{i:04d}.png")
    img.save(ruta)

print("clip generado en:", carpeta_salida)