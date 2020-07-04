from typing import Tuple

from pycuda.gpuarray import GPUArray


def ssnp_step(u: GPUArray, u_d: GPUArray, dz: float, n: GPUArray = None) -> Tuple[GPUArray, GPUArray]: ...


def bpm_step(u: GPUArray, dz: float, n: GPUArray = None) -> GPUArray: ...


def pure_forward_d(u: GPUArray, out: GPUArray = None) -> GPUArray: ...


def tilt(img: GPUArray, c_ab: tuple, *, trunc: bool = False, copy: bool = False): ...


def binary_pupil(u: GPUArray, na: float) -> GPUArray: ...


def merge_prop(ub: GPUArray, uf: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...


def split_prop(u: GPUArray, u_d: GPUArray, copy: bool = False) -> Tuple[GPUArray, GPUArray]: ...

def get_funcs(arr_like: GPUArray, res, model):
