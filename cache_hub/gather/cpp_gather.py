import os
import importlib.util
from functools import lru_cache
from typing import Any

import torch
from torch.utils.cpp_extension import load


@lru_cache()
def load_gather_ext(verbose: bool = False) -> Any:
    """
    Build & load the CPU gather extension once per process.
    Set env var POLAR_GATHER_EXT_VERBOSE=1 to see build logs.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(this_dir, "build")
    module_path = os.path.join(build_dir, "polar_gather_ext.so")
    source_path = os.path.join(this_dir, "gather_cpu.cpp")
    os.makedirs(build_dir, exist_ok=True)

    extra_cflags = ["-O3", "-std=c++17"]
    # Rely on PyTorch threadpool (at::parallel_for), no OpenMP required.
    name = "polar_gather_ext"

    if os.path.exists(module_path):
        if os.path.getmtime(module_path) >= os.path.getmtime(source_path):
            spec = importlib.util.spec_from_file_location(name, module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return load(
        name=name,
        sources=[source_path],
        extra_cflags=extra_cflags,
        with_cuda=False,
        build_directory=build_dir,
        verbose=verbose or (os.environ.get("POLAR_GATHER_EXT_VERBOSE", "0") == "1"),
    )

