#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void raster_forward_kernel(
    const float* mu,
    const float* scale,
    const float* theta,
    const float* opacity,
    const float* color,
    float* out,
    int N,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (idx >= total) {
        return;
    }

    int fila = idx / W;
    int col = idx % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;

    float T = 1.0f;

    for (int i = 0; i < N; i++) {
        float mu_f = mu[i * 2 + 0];
        float mu_c = mu[i * 2 + 1];

        float sx = scale[i * 2 + 0];
        float sy = scale[i * 2 + 1];

        float th = theta[i];

        float df = pr - mu_f;
        float dc = pc - mu_c;

        float c = cosf(th);
        float s = sinf(th);

        float inv_sx2 = 1.0f / (sx * sx + 1e-8f);
        float inv_sy2 = 1.0f / (sy * sy + 1e-8f);

        float m00 = c * c * inv_sx2 + s * s * inv_sy2;
        float m01 = c * s * (inv_sx2 - inv_sy2);
        float m11 = s * s * inv_sx2 + c * c * inv_sy2;

        float quad = df * (m00 * df + m01 * dc) +
                     dc * (m01 * df + m11 * dc);

        float G = expf(-0.5f * quad);

        float alpha = opacity[i] * G;

        if (alpha > 0.99f) {
            alpha = 0.99f;
        }

        float peso = alpha * T;

        r_final += peso * color[i * 3 + 0];
        g_final += peso * color[i * 3 + 1];
        b_final += peso * color[i * 3 + 2];

        T *= (1.0f - alpha);

        if (T < 1e-4f) {
            break;
        }
    }

    out[idx * 3 + 0] = r_final;
    out[idx * 3 + 1] = g_final;
    out[idx * 3 + 2] = b_final;
}

__global__ void raster_forward_conic_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    float* __restrict__ out,
    int N,
    int H,
    int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (idx >= total) {
        return;
    }

    int fila = idx / W;
    int col = idx % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;

    float T = 1.0f;

    for (int i = 0; i < N; i++) {
        float mu_f = mu[i * 2 + 0];
        float mu_c = mu[i * 2 + 1];

        float df = pr - mu_f;
        float dc = pc - mu_c;

        float m00 = conic[i * 3 + 0];
        float m01 = conic[i * 3 + 1];
        float m11 = conic[i * 3 + 2];

        float quad = df * (m00 * df + m01 * dc) +
                     dc * (m01 * df + m11 * dc);

        float G = expf(-0.5f * quad);

        float alpha = opacity[i] * G;

        if (alpha > 0.99f) {
            alpha = 0.99f;
        }

        float peso = alpha * T;

        r_final += peso * color[i * 3 + 0];
        g_final += peso * color[i * 3 + 1];
        b_final += peso * color[i * 3 + 2];

        T *= (1.0f - alpha);

        if (T < 1e-4f) {
            break;
        }
    }

    out[idx * 3 + 0] = r_final;
    out[idx * 3 + 1] = g_final;
    out[idx * 3 + 2] = b_final;
}


__global__ void raster_forward_tiled_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const long long* __restrict__ gaussian_ids,
    const long long* __restrict__ ranges,
    float* __restrict__ out,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int tile_id = blockIdx.x;
    int local_idx = threadIdx.x;

    int local_y = local_idx / tile_size;
    int local_x = local_idx % tile_size;

    int tile_y = tile_id / tiles_x;
    int tile_x = tile_id % tiles_x;

    int fila = tile_y * tile_size + local_y;
    int col = tile_x * tile_size + local_x;

    if (fila >= H || col >= W) {
        return;
    }

    long long start = ranges[tile_id * 2 + 0];
    long long end = ranges[tile_id * 2 + 1];

    int out_idx = (fila * W + col) * 3;

    if (start < 0 || end <= start) {
        out[out_idx + 0] = 0.0f;
        out[out_idx + 1] = 0.0f;
        out[out_idx + 2] = 0.0f;
        return;
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float r_final = 0.0f;
    float g_final = 0.0f;
    float b_final = 0.0f;

    float T = 1.0f;

    for (long long p = start; p < end; p++) {
        int i = (int)gaussian_ids[p];

        float mu_f = mu[i * 2 + 0];
        float mu_c = mu[i * 2 + 1];

        float df = pr - mu_f;
        float dc = pc - mu_c;

        float m00 = conic[i * 3 + 0];
        float m01 = conic[i * 3 + 1];
        float m11 = conic[i * 3 + 2];

        float quad = df * (m00 * df + m01 * dc) +
                     dc * (m01 * df + m11 * dc);

        float G = expf(-0.5f * quad);
        float alpha = opacity[i] * G;

        if (alpha > 0.99f) {
            alpha = 0.99f;
        }

        float peso = alpha * T;

        r_final += peso * color[i * 3 + 0];
        g_final += peso * color[i * 3 + 1];
        b_final += peso * color[i * 3 + 2];

        T *= (1.0f - alpha);

        if (T < 1e-4f) {
            break;
        }
    }

    out[out_idx + 0] = r_final;
    out[out_idx + 1] = g_final;
    out[out_idx + 2] = b_final;
}



__device__ __forceinline__ float evaluar_alpha_conic_device(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    int i,
    float pr,
    float pc,
    float* G_out,
    float* dx_out,
    float* dy_out,
    float* power_out,
    bool* unclamped_out
) {
    float mu_f = mu[i * 2 + 0];
    float mu_c = mu[i * 2 + 1];

    float dx = pr - mu_f;
    float dy = pc - mu_c;

    float m00 = conic[i * 3 + 0];
    float m01 = conic[i * 3 + 1];
    float m11 = conic[i * 3 + 2];

    float quad = dx * (m00 * dx + m01 * dy) +
                 dy * (m01 * dx + m11 * dy);

    float power = -0.5f * quad;
    float G = expf(power);

    float alpha_pre = opacity[i] * G;
    bool unclamped = alpha_pre < 0.99f;

    float alpha = alpha_pre;
    if (alpha > 0.99f) {
        alpha = 0.99f;
    }

    if (G_out) {
        *G_out = G;
    }
    if (dx_out) {
        *dx_out = dx;
    }
    if (dy_out) {
        *dy_out = dy;
    }
    if (power_out) {
        *power_out = power;
    }
    if (unclamped_out) {
        *unclamped_out = unclamped;
    }

    return alpha;
}


__global__ void raster_backward_conic_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_color,
    int N,
    int H,
    int W
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (pix >= total) {
        return;
    }

    int fila = pix / W;
    int col = pix % W;

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float go_r = grad_out[pix * 3 + 0];
    float go_g = grad_out[pix * 3 + 1];
    float go_b = grad_out[pix * 3 + 2];

    // Backward brute-force v1:
    // Para cada gaussiana i:
    //   1. Recomputa T_i con un loop 0..i-1
    //   2. Recomputa el color de lo que viene despues de i
    //   3. Calcula dL/dalpha_i
    //   4. Propaga a color, opacity, mu y conic
    //
    // Esto es O(P*N^2). Es para validar gradientes, no para performance final.

    for (int i = 0; i < N; i++) {
        // -------- prefix T_i --------
        float T_i = 1.0f;

        for (int j = 0; j < i; j++) {
            float alpha_j = evaluar_alpha_conic_device(
                mu, conic, opacity, j, pr, pc,
                nullptr, nullptr, nullptr, nullptr, nullptr
            );

            T_i *= (1.0f - alpha_j);

            if (T_i < 1e-4f) {
                break;
            }
        }

        if (T_i < 1e-4f) {
            continue;
        }

        // -------- alpha_i --------
        float G_i = 0.0f;
        float dx_i = 0.0f;
        float dy_i = 0.0f;
        float power_i = 0.0f;
        bool unclamped_i = true;

        float alpha_i = evaluar_alpha_conic_device(
            mu, conic, opacity, i, pr, pc,
            &G_i, &dx_i, &dy_i, &power_i, &unclamped_i
        );

        // -------- tail color despues de i --------
        float tail_r = 0.0f;
        float tail_g = 0.0f;
        float tail_b = 0.0f;
        float T_tail = 1.0f;

        for (int k = i + 1; k < N; k++) {
            float alpha_k = evaluar_alpha_conic_device(
                mu, conic, opacity, k, pr, pc,
                nullptr, nullptr, nullptr, nullptr, nullptr
            );

            float peso_k = T_tail * alpha_k;

            tail_r += peso_k * color[k * 3 + 0];
            tail_g += peso_k * color[k * 3 + 1];
            tail_b += peso_k * color[k * 3 + 2];

            T_tail *= (1.0f - alpha_k);

            if (T_tail < 1e-4f) {
                break;
            }
        }

        // dC/dalpha_i = T_i * (color_i - tail)
        float dC_da_r = T_i * (color[i * 3 + 0] - tail_r);
        float dC_da_g = T_i * (color[i * 3 + 1] - tail_g);
        float dC_da_b = T_i * (color[i * 3 + 2] - tail_b);

        float dL_dalpha = go_r * dC_da_r + go_g * dC_da_g + go_b * dC_da_b;

        // grad color_i = grad_out * T_i * alpha_i
        float peso_i = T_i * alpha_i;

        atomicAdd(&grad_color[i * 3 + 0], go_r * peso_i);
        atomicAdd(&grad_color[i * 3 + 1], go_g * peso_i);
        atomicAdd(&grad_color[i * 3 + 2], go_b * peso_i);

        // Si alpha fue clamp a 0.99, la derivada del clamp es 0.
        if (!unclamped_i) {
            continue;
        }

        // alpha = opacity * G
        float dL_dopacity = dL_dalpha * G_i;
        atomicAdd(&grad_opacity[i], dL_dopacity);

        float dL_dG = dL_dalpha * opacity[i];

        // G = exp(power)
        float dL_dpower = dL_dG * G_i;

        float m00 = conic[i * 3 + 0];
        float m01 = conic[i * 3 + 1];
        float m11 = conic[i * 3 + 2];

        // power = -0.5 * (m00*dx^2 + 2*m01*dx*dy + m11*dy^2)
        // dx = pixel_fila - mu_fila
        // dy = pixel_col  - mu_col

        float dpower_dmu_f = m00 * dx_i + m01 * dy_i;
        float dpower_dmu_c = m01 * dx_i + m11 * dy_i;

        atomicAdd(&grad_mu[i * 2 + 0], dL_dpower * dpower_dmu_f);
        atomicAdd(&grad_mu[i * 2 + 1], dL_dpower * dpower_dmu_c);

        float dpower_dm00 = -0.5f * dx_i * dx_i;
        float dpower_dm01 = -1.0f * dx_i * dy_i;
        float dpower_dm11 = -0.5f * dy_i * dy_i;

        atomicAdd(&grad_conic[i * 3 + 0], dL_dpower * dpower_dm00);
        atomicAdd(&grad_conic[i * 3 + 1], dL_dpower * dpower_dm01);
        atomicAdd(&grad_conic[i * 3 + 2], dL_dpower * dpower_dm11);
    }
}

__global__ void raster_backward_tiled_kernel(
    const float* __restrict__ mu,
    const float* __restrict__ conic,
    const float* __restrict__ opacity,
    const float* __restrict__ color,
    const int64_t* __restrict__ gaussian_ids,
    const int64_t* __restrict__ ranges,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_conic,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_color,
    int H,
    int W,
    int tile_size,
    int tiles_x
) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;

    if (pix >= total) {
        return;
    }

    int fila = pix / W;
    int col = pix % W;

    int tile_x = col / tile_size;
    int tile_y = fila / tile_size;
    int tile_id = tile_y * tiles_x + tile_x;

    int64_t start = ranges[tile_id * 2 + 0];
    int64_t end = ranges[tile_id * 2 + 1];

    if (start < 0 || end <= start) {
        return;
    }

    float pr = (float)fila + 0.5f;
    float pc = (float)col + 0.5f;

    float go_r = grad_out[pix * 3 + 0];
    float go_g = grad_out[pix * 3 + 1];
    float go_b = grad_out[pix * 3 + 2];

    // Version v1:
    // Para cada gaussiana del tile:
    //   - recomputa T_i con las gaussianas anteriores del mismo tile
    //   - recomputa tail con las gaussianas posteriores del mismo tile
    //   - acumula gradientes con atomicAdd
    //
    // Esto valida el backward tiled. No es la version final optimizada.

    for (int64_t ii = start; ii < end; ii++) {
        int gid_i = (int)gaussian_ids[ii];

        float T_i = 1.0f;

        for (int64_t jj = start; jj < ii; jj++) {
            int gid_j = (int)gaussian_ids[jj];

            float alpha_j = evaluar_alpha_conic_device(
                mu,
                conic,
                opacity,
                gid_j,
                pr,
                pc,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );

            T_i *= (1.0f - alpha_j);

            if (T_i < 1e-4f) {
                break;
            }
        }

        if (T_i < 1e-4f) {
            continue;
        }

        float G_i = 0.0f;
        float dx_i = 0.0f;
        float dy_i = 0.0f;
        float power_i = 0.0f;
        bool unclamped_i = true;

        float alpha_i = evaluar_alpha_conic_device(
            mu,
            conic,
            opacity,
            gid_i,
            pr,
            pc,
            &G_i,
            &dx_i,
            &dy_i,
            &power_i,
            &unclamped_i
        );

        float tail_r = 0.0f;
        float tail_g = 0.0f;
        float tail_b = 0.0f;
        float T_tail = 1.0f;

        for (int64_t kk = ii + 1; kk < end; kk++) {
            int gid_k = (int)gaussian_ids[kk];

            float alpha_k = evaluar_alpha_conic_device(
                mu,
                conic,
                opacity,
                gid_k,
                pr,
                pc,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr
            );

            float peso_k = T_tail * alpha_k;

            tail_r += peso_k * color[gid_k * 3 + 0];
            tail_g += peso_k * color[gid_k * 3 + 1];
            tail_b += peso_k * color[gid_k * 3 + 2];

            T_tail *= (1.0f - alpha_k);

            if (T_tail < 1e-4f) {
                break;
            }
        }

        float ci_r = color[gid_i * 3 + 0];
        float ci_g = color[gid_i * 3 + 1];
        float ci_b = color[gid_i * 3 + 2];

        float dC_da_r = T_i * (ci_r - tail_r);
        float dC_da_g = T_i * (ci_g - tail_g);
        float dC_da_b = T_i * (ci_b - tail_b);

        float dL_dalpha = go_r * dC_da_r + go_g * dC_da_g + go_b * dC_da_b;

        float peso_i = T_i * alpha_i;

        atomicAdd(&grad_color[gid_i * 3 + 0], go_r * peso_i);
        atomicAdd(&grad_color[gid_i * 3 + 1], go_g * peso_i);
        atomicAdd(&grad_color[gid_i * 3 + 2], go_b * peso_i);

        if (!unclamped_i) {
            continue;
        }

        float dL_dopacity = dL_dalpha * G_i;
        atomicAdd(&grad_opacity[gid_i], dL_dopacity);

        float dL_dG = dL_dalpha * opacity[gid_i];
        float dL_dpower = dL_dG * G_i;

        float m00 = conic[gid_i * 3 + 0];
        float m01 = conic[gid_i * 3 + 1];
        float m11 = conic[gid_i * 3 + 2];

        float dpower_dmu_f = m00 * dx_i + m01 * dy_i;
        float dpower_dmu_c = m01 * dx_i + m11 * dy_i;

        atomicAdd(&grad_mu[gid_i * 2 + 0], dL_dpower * dpower_dmu_f);
        atomicAdd(&grad_mu[gid_i * 2 + 1], dL_dpower * dpower_dmu_c);

        float dpower_dm00 = -0.5f * dx_i * dx_i;
        float dpower_dm01 = -1.0f * dx_i * dy_i;
        float dpower_dm11 = -0.5f * dy_i * dy_i;

        atomicAdd(&grad_conic[gid_i * 3 + 0], dL_dpower * dpower_dm00);
        atomicAdd(&grad_conic[gid_i * 3 + 1], dL_dpower * dpower_dm01);
        atomicAdd(&grad_conic[gid_i * 3 + 2], dL_dpower * dpower_dm11);
    }
}

torch::Tensor raster_forward_cuda(
    torch::Tensor mu,
    torch::Tensor scale,
    torch::Tensor theta,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    int N = mu.size(0);

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_forward_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        scale.data_ptr<float>(),
        theta.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}



torch::Tensor raster_forward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    int H,
    int W
) {
    int N = mu.size(0);

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_forward_conic_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}


torch::Tensor raster_forward_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    int H,
    int W,
    int tile_size
) {
    int tiles_x = (W + tile_size - 1) / tile_size;
    int tiles_y = (H + tile_size - 1) / tile_size;
    int total_tiles = tiles_x * tiles_y;

    auto out = torch::zeros(
        {H, W, 3},
        torch::TensorOptions()
            .device(mu.device())
            .dtype(torch::kFloat32)
    );

    int threads = tile_size * tile_size;

    raster_forward_tiled_kernel<<<total_tiles, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<long long>(),
        ranges.data_ptr<long long>(),
        out.data_ptr<float>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}


std::vector<torch::Tensor> raster_backward_conic_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor grad_out,
    int H,
    int W
) {
    int N = mu.size(0);

    auto grad_mu = torch::zeros_like(mu);
    auto grad_conic = torch::zeros_like(conic);
    auto grad_opacity = torch::zeros_like(opacity);
    auto grad_color = torch::zeros_like(color);

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    raster_backward_conic_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        N,
        H,
        W
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_mu, grad_conic, grad_opacity, grad_color};
}


std::vector<torch::Tensor> raster_backward_tiled_cuda(
    torch::Tensor mu,
    torch::Tensor conic,
    torch::Tensor opacity,
    torch::Tensor color,
    torch::Tensor gaussian_ids,
    torch::Tensor ranges,
    torch::Tensor grad_out,
    int H,
    int W,
    int tile_size
) {
    auto grad_mu = torch::zeros_like(mu);
    auto grad_conic = torch::zeros_like(conic);
    auto grad_opacity = torch::zeros_like(opacity);
    auto grad_color = torch::zeros_like(color);

    int total = H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    int tiles_x = (W + tile_size - 1) / tile_size;

    raster_backward_tiled_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        conic.data_ptr<float>(),
        opacity.data_ptr<float>(),
        color.data_ptr<float>(),
        gaussian_ids.data_ptr<int64_t>(),
        ranges.data_ptr<int64_t>(),
        grad_out.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_conic.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_color.data_ptr<float>(),
        H,
        W,
        tile_size,
        tiles_x
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return {grad_mu, grad_conic, grad_opacity, grad_color};
}