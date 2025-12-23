from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 确保找到 nvcc (如果你环境变量没配好，可以在这里硬编码路径)
# os.environ["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"

setup(
    name='my_matmul', # 你的包名
    ext_modules=[
        CUDAExtension(
            name='my_matmul', 
            sources=['binding.cpp', 'matrix_mul_kernel.cu'], # 编译这两个文件
            extra_compile_args={
                'cxx': [],
                # arch=compute_86,code=sm_86 对应 RTX 30系
                # 如果你是 40系，改成 sm_89
                'nvcc': ['-O3', '-gencode=arch=compute_89,code=sm_89'] 
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)