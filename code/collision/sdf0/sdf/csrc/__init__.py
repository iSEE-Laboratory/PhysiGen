# sdf/csrc/__init__.py
from importlib import import_module as _imp
from pathlib import Path as _Path

# 动态导入同目录下编译生成的 .so 扩展
_so_name = next(p.stem for p in _Path(__file__).parent.glob('*.so'))
# 只保留前缀  csrc
_module_name = _so_name.split('.')[0]
_ext = _imp(f".{_module_name}", __package__)

# 把 C++/CUDA 函数暴露到当前命名空间
sdf = _ext.sdf
__all__ = ["sdf"]