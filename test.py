import time
import numpy as np
import cupy as cp


if __name__ == '__main__':

    a = np.random.randint(0,100,[1322,9])
    start = time.time()
    a = cp.array(a)
    [u, s, v] = cp.linalg.svd(a)
    v = cp.asnumpy(v)
    print(type(v))
    end = time.time()
    print('processing_matrix time: {} Seconds'.format(end - start))