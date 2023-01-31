import numpy as np


def op_mse(y_val):
    m = y_val.shape[0]

    def mse_f(y_approx):
        return np.sum((y_approx - y_val) ** 2) / m

    def mse_dy_approx(de, y_approx):
        return de * 2 * (y_approx - y_val) / m

    return mse_f, (mse_dy_approx,)


def relu_f(z):
    return np.maximum(0, z)


def relu_dz(da, z):
    return da * (z > 0)


op_relu = relu_f, (relu_dz,)


def matmul_f(w, b, a):
    return np.matmul(a, w) + b


def matmul_dw(dz, w, b, a):
    return np.matmul(a.T, dz)


def matmul_db(dz, w, b, a):
    return np.matmul(np.ones((1, dz.shape[0])), dz)


def matmul_da(dz, w, b, a):
    return np.matmul(dz, w.T)


op_matmul = matmul_f, (matmul_dw, matmul_db, matmul_da)
