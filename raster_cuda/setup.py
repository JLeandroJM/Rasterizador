from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="raster_cuda",
    ext_modules=[
        CUDAExtension(
            name="raster_cuda",
            sources=[
                "raster_cuda.cpp",
                "raster_cuda_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)