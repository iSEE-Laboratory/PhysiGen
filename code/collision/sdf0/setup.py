
import os
import subprocess
import sys
# try:
#     import torch
# except ImportError:
#     print("Error: PyTorch (torch) is required. Please install it first using `pip install torch` or `conda install pytorch ...`")
#     sys.exit(1)
# try:
#     from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
# except ImportError:
#     print("Error: PyTorch (torch) and its C++ extension utilities are required for building this package.", file=sys.stderr)
#     print("Please ensure PyTorch is installed in your active environment.", file=sys.stderr)
#     print("You can install it via: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` (adjust CUDA version if needed)", file=sys.stderr)
#     print("Or: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (adjust CUDA version if needed)", file=sys.stderr)
#     sys.exit(1)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension # 现在应该能够直接导入了




CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []


ARCH = [
    "-gencode=arch=compute_75,code=sm_75",   # 2080
    "-gencode=arch=compute_86,code=sm_86",   # 3090
    # 对 4090 如下二选一 -----------------------------
    # "-gencode=arch=compute_89,code=sm_89",   # a) 真·sm_89（需 nvcc ≥ 11.8）
    # "-gencode=arch=compute_86,code=compute_86",  # b) 仅 PTX fallback（nvcc ≤ 11.6 也行）
]


# ext_modules=[
#     CUDAExtension('sdf.csrc', [
#         'sdf/csrc/sdf_cuda.cpp',
#         'sdf/csrc/sdf_cuda_kernel.cu',
#         ]),
#     ]

ext_modules = [
    CUDAExtension(
        "sdf.csrc",
        ["sdf/csrc/sdf_cuda.cpp", "sdf/csrc/sdf_cuda_kernel.cu"],
        extra_compile_args={"cxx": ["-O3"],
                            "nvcc": ["-O3"] + ARCH},
    )
]

from setuptools import setup, find_packages

setup(
    # description='PyTorch implementation of SDF loss',
    # author='Nikos Kolotouros',
    # author_email='nkolot@seas.upenn.edu',
    # license='MIT License',
    version='0.0.1',
    name='sdf_pytorch',
    packages=['sdf', 'sdf.csrc'],
    # setup_requires=['torch'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
