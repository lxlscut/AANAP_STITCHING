import cupy as cp
def gpu_svd(a):
    a = cp.array(a)
    [u, s, v] = cp.linalg.svd(a)
    v = cp.asnumpy(v)
    return v
