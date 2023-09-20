import numpy as np


def trans(A):
    W, H = A.shape
    A11 = A[:W//2, :H//2]
    A12 = A[:W // 2, H // 2:]
    A21 = A[W // 2:, :H // 2]
    A22 = A[W // 2:, H // 2:]
    B = np.concatenate([np.concatenate([A22, A21], 1), np.concatenate([A12, A11], 1)], 0)
    return B


if __name__ == '__main__':
    A = np.random.rand(4,4)
    print(A)
    B = trans(A)
    print(B)