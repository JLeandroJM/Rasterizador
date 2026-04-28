from rasterizador import *
import time

def main():
    gaussianas = crear_escena()
    alto = 512
    ancho = 512

    sx = 256 / alto
    sy = 256 / ancho

    for g in gaussianas:
        g.centro = g.centro * np.array([sx, sy])
        
        g.escala = g.escala * np.array([sx * 0.5, sy * 0.5])

    
    print("Rasterizando escena ... ")
    t0 = time.time()
    imagen = rasterizar(gaussianas, alto, ancho)
    t_total = time.time() - t0

    print(f"Tiempo de proceso: {t_total:.2f} segundos" )

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.imshow(np.clip(imagen, 0, 1))
    fig.savefig("resultado.png")
    plt.show()

if __name__ == "__main__":
    main()







   