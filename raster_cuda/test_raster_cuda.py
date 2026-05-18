import torch
import matplotlib.pyplot as plt
import raster_cuda


def main():
    device = "cuda"

    H = 128
    W = 128

    # mu usa formato (fila, columna)
    mu = torch.tensor([
        [64.0, 64.0],
        [40.0, 40.0],
        [90.0, 80.0],
    ], device=device, dtype=torch.float32)

    scale = torch.tensor([
        [18.0, 18.0],
        [10.0, 20.0],
        [25.0, 8.0],
    ], device=device, dtype=torch.float32)

    theta = torch.tensor([
        0.0,
        0.7,
        -0.5,
    ], device=device, dtype=torch.float32)

    opacity = torch.tensor([
        0.9,
        0.8,
        0.7,
    ], device=device, dtype=torch.float32)

    color = torch.tensor([
        [1.0, 0.2, 0.1],
        [0.1, 1.0, 0.2],
        [0.2, 0.3, 1.0],
    ], device=device, dtype=torch.float32)

    render = raster_cuda.forward(
        mu.contiguous(),
        scale.contiguous(),
        theta.contiguous(),
        opacity.contiguous(),
        color.contiguous(),
        H,
        W
    )

    print("render shape:", render.shape)
    print("render min:", render.min().item())
    print("render max:", render.max().item())

    img = render.detach().clamp(0, 1).cpu().numpy()

    plt.imshow(img)
    plt.axis("off")
    plt.title("Raster CUDA forward simple")
    plt.savefig("raster_cuda_test.png", dpi=150)
    print("guardado: raster_cuda_test.png")


if __name__ == "__main__":
    main()