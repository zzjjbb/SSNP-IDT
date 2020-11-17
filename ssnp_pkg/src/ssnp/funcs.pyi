from typing import Literal

import numpy as np
from pycuda.gpuarray import GPUArray
from pycuda.elementwise import ElementwiseKernel
from pycuda.driver import Stream
from pycuda.reduction import ReductionKernel
from reikna.core.computation import ComputationCallable
from .utils import Multipliers


class Funcs:
    shape: tuple
    batch: int
    res: tuple
    n0: float
    stream: Stream
    kz: np.ndarray
    kz_gpu: GPUArray
    eva: np.ndarray
    multiplier: Multipliers
    _fft_callable: ComputationCallable
    # __temp_memory_pool: dict
    _prop_cache: dict
    reduce_mse_cr_krn: ReductionKernel
    reduce_mse_cc_krn: ReductionKernel
    mse_cr_grad_krn: ElementwiseKernel
    mse_cc_grad_krn: ElementwiseKernel
    mul_grad_bp_krn: ElementwiseKernel
    _fft_reikna: callable
    _fft_sk: callable

    def __init__(self, arr_like: GPUArray, res, n0, stream: Stream = None,
                 fft_type: Literal["reikna", "skcuda"] = "reikna"): ...

    @staticmethod
    def _compile_fft(shape, dtype, stream): ...

    def fft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False, inverse=False) -> GPUArray: ...

    def ifft(self, arr: GPUArray, output: GPUArray = None, copy: bool = False) -> GPUArray: ...

    def fourier(self, arr: GPUArray, copy: bool = False): ...

    def diffract(self, *args) -> None: ...

    def scatter(self, *args) -> None: ...

    def _get_prop(self, dz): ...

    # @staticmethod
    # def get_temp_mem(arr_like: GPUArray, index=0): ...

    def reduce_mse_cr(self, u: GPUArray, m: GPUArray) -> GPUArray: ...

    def reduce_mse_cc(self, u: GPUArray, m: GPUArray) -> GPUArray: ...

    def mse_cr_grad(self, u: GPUArray, m: GPUArray, out: GPUArray): ...

    def mse_cc_grad(self, u: GPUArray, m: GPUArray, out: GPUArray): ...

    def mul_grad_bp(self, ug: GPUArray, mul: GPUArray): ...


class BPMFuncs(Funcs):
    def _get_prop(self, dz): ...

    def diffract(self, a: GPUArray, dz) -> None: ...

    def diffract_g(self, ag, dz): ...

    def scatter(self, u, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, ng, dz): ...


class SSNPFuncs(Funcs):
    _fused_mam_callable_krn: ElementwiseKernel
    _merge_prop_krn: ElementwiseKernel
    _split_prop_krn: ElementwiseKernel
    _merge_grad_krn: ElementwiseKernel

    def merge_prop(self, af, ab): ...

    def split_prop(self, a, a_d): ...

    def merge_grad(self, afg, abg): ...

    def split_grad(self, ag, a_dg): ...

    def _get_prop(self, dz): ...

    def diffract(self, a, a_d, dz) -> None: ...

    def diffract_g(self, ag, a_dg, dz): ...

    def scatter(self, u, u_d, n, dz) -> None: ...

    def scatter_g(self, u, n, ug, u_dg, ng, dz): ...
