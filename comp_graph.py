import math

import numpy
import numpy as np


class CGraph:
    def __init__(self):
        self.inputs = dict()
        self.consumers = dict()
        self.operations = dict()
        self.n_nodes = 0

    def new_var(self):
        u = self.n_nodes
        self.n_nodes += 1

        self.inputs[u] = []
        self.consumers[u] = []
        return u

    def apply(self, op, *xs):
        y = self.new_var()

        self.inputs[y].extend(xs)
        self.operations[y] = op
        for x in xs:
            self.consumers[x].append(y)

        return y

    @staticmethod
    def dict2graph(d, n_nodes):
        graph = [None for _ in range(n_nodes)]
        for k in d:
            graph[k] = d[k]
        return graph

    def forward_full(self, xs):
        f_graph = [None for _ in range(self.n_nodes)]
        for y in range(self.n_nodes):
            op = self.operations.get(y)
            if op:
                f, _ = op
                x_values = [f_graph[x] for x in self.inputs[y]]
                f_graph[y] = f(*x_values)
            else:
                f_graph[y] = xs[y]
        return f_graph

    def forward_partial(self, y, f_graph):
        if f_graph[y] is not None:
            return f_graph

        x_values = []
        for x in self.inputs[y]:
            if f_graph[x] is None:
                self.forward_partial(x, f_graph)
            x_values.append(f_graph[x])

        f, _ = self.operations.get(y)
        y_value = f(*x_values)
        f_graph[y] = y_value
        return f_graph

    def backward_full(self, f_graph):
        b_graph = [None for _ in range(self.n_nodes)]
        b_graph[-1] = 1
        for y in reversed(range(self.n_nodes)):
            xs = self.inputs.get(y)
            if xs:
                _, df = self.operations.get(y)
                x_values = [f_graph[x] for x in self.inputs[y]]
                for j, x in enumerate(xs):
                    g = df[j](b_graph[y], *x_values)
                    if b_graph[x] is None:
                        b_graph[x] = g
                    else:
                        b_graph[x] += g
        return b_graph

    def backward_partial(self, x, b_graph, f_graph):
        if b_graph[x] is not None:
            return b_graph

        if x == self.n_nodes - 1:
            b_graph[x] = 1
            return b_graph

        for y in self.consumers[x]:
            if b_graph[y] is None:
                self.backward_partial(y, b_graph, f_graph)
            _, df = self.operations.get(y)
            x_values = [f_graph[x] for x in self.inputs[y]]
            g = df[self.inputs[y].index(x)](b_graph[y], *x_values)
            if b_graph[x] is None:
                b_graph[x] = g
            else:
                b_graph[x] += g
        return b_graph


def test1():
    cg1 = CGraph()
    x1 = cg1.new_var()
    y1 = cg1.apply(
        (
            lambda x: x ** 2,
            [lambda dy, x: dy * 2 * x]
        ),
        x1)  # y1 = x^2, dy3/dy1 =
    y2 = cg1.apply(
        (
            lambda x: math.exp(x),
            [lambda dy, x: dy * math.exp(x)]
        ),
        y1)  # y2 = exp(y1), dy3/dy2 = -1/y2^2
    y3 = cg1.apply(
        (
            lambda x, y: math.log(x) + 1 / y,
            [
                lambda dy, x, y: dy * (1 / x),
                lambda dy, x, y: dy * (- 1 / (y * y))
            ]
        ),
        y1, y2)  # y3 = ln(y1) + 1/y2 = 2*ln(x) + exp(-x^2)

    print(f'idx(x1)={x1}')
    print(f'idx(y3)={y3}')
    print(f'cg1.inputs={cg1.inputs}')
    print(f'cg1.consumers={cg1.consumers}')
    print(f'cg1.operations={cg1.operations}')
    print(f'cg1.u_counter={cg1.n_nodes}')
    fg1_full = cg1.forward_full({x1: 1})
    fg1_partial = cg1.forward_partial(x1, CGraph.dict2graph({x1: 1}, cg1.n_nodes))
    print(f'fg1_full={fg1_full}')
    print(f'fg1_partial={fg1_partial}')
    print(f'cg1.backward_full(fg1)={cg1.backward_full(fg1_full)}')
    print(f'cg1.backward_partial(fg1)={cg1.backward_partial(y2, CGraph.dict2graph({}, cg1.n_nodes), fg1_full)}')


def test2():
    cg2 = CGraph()
    w1 = cg2.new_var()
    b1 = cg2.new_var()
    X1 = np.identity(5)
    y1 = cg2.apply(
        (
            lambda w, b: np.matmul(X1, w) + b,
            [
                lambda dy, w, b: np.matmul(X1.T, dy),
                lambda dy, w, b: np.matmul(dy.T, np.ones(dy.shape[0]))
            ]
        ),
        w1, b1
    )
    z1 = cg2.apply(
        (
            lambda x: np.dot(x, x),
            [lambda dy, x: dy * 2 * x]
        ),
        y1
    )
    p1 = {
        w1: np.random.rand(5),
        b1: 1
    }
    fg2_full = cg2.forward_full(p1)

    fg2_partial = cg2.forward_partial(
        y1,
        CGraph.dict2graph(p1, cg2.n_nodes)
    )
    print(f'fg2_full={fg2_full}')
    print(f'fg2_partial={fg2_partial}')

    y1_ = np.matmul(X1, fg2_full[w1]) + fg2_full[b1]
    print(fg2_full[y1] - y1_)
    z1_ = np.dot(y1_, y1_)
    print(fg2_full[z1] - z1_)

    bg2_full = cg2.backward_full(fg2_full)
    print(fg2_full[y1])
    print(bg2_full[y1])

    bg2_partial = cg2.backward_partial(z1, CGraph.dict2graph({}, cg2.n_nodes), fg2_full)
    print(bg2_full[z1] - bg2_partial[z1])


def test3():
    cg3 = CGraph()
    w1 = cg3.new_var()
    b1 = cg3.new_var()
    w2 = cg3.new_var()
    b2 = cg3.new_var()
    X1 = np.identity(5)

    m = X1.shape[0]
    z1 = cg3.apply(
        (
            lambda w, b: np.matmul(X1, w) + b,
            [
                lambda dz, w, b: np.matmul(X1.T, dz),
                lambda dz, w, b: np.matmul(np.ones((1, m)), dz)
            ]
        ),
        w1, b1
    )
    a1 = cg3.apply(
        (
            lambda z: np.maximum(0, z),
            [lambda da, z: da * (z > 0)]
        ),
        z1
    )
    z2 = cg3.apply(
        (
            lambda w, b, a: np.matmul(a, w) + b,
            [
                lambda dz, w, b, a: np.matmul(a.T, dz),  # (m*3)^T * m*1 = 3*1
                lambda dz, w, b, a: np.matmul(np.ones((1, m)), dz),
                lambda dz, w, b, a: np.matmul(dz, w.T)  # m*1 * (3*1)^T= m*3
            ]
        ),
        w2, b2, a1
    )
    a2 = cg3.apply(
        (
            lambda z: np.maximum(0, z),
            [lambda da, z: da * (z > 0)]
        ),
        z2
    )
    loss = cg3.apply(
        (
            lambda a: np.matmul(a.T, a),
            [lambda dloss, a: dloss * 2 * a]
        ),
        a2
    )
    p1 = {
        w1: np.random.rand(5, 3),
        b1: np.random.rand(1, 3),
        w2: np.random.rand(3, 1),
        b2: np.random.rand(1, 1)
    }
    fg3_full = cg3.forward_full(p1)
    var = loss
    fg3_partial = cg3.forward_partial(var, CGraph.dict2graph(p1, cg3.n_nodes))
    bg3_full = cg3.backward_full(fg3_full)
    bg3_partial = cg3.backward_partial(var, CGraph.dict2graph({}, cg3.n_nodes), fg3_full)
    print(f'idx(var)={var}')
    print(f'fg3_full[var]={fg3_full[var]}')
    print(f'fg3_partial[var]={fg3_partial[var]}')
    print(f'bg3_full[var]={bg3_full[var]}')
    print(f'bg3_partial[var]={bg3_partial[var]}')


def init_rand_param(w_vars, b_vars, ns, a0_var, x_val, w_mult=0.01, b_mult=1):
    parameters = {a0_var: x_val}
    for i in range(1, len(ns)):
        parameters[w_vars[i]] = w_mult * np.random.rand(ns[i - 1], ns[i])
        parameters[b_vars[i]] = b_mult * np.random.rand(1, ns[i])
    return parameters


def matmul_f(w, b, a):
    return np.matmul(a, w) + b


def matmul_dw(dz, w, b, a):
    return np.matmul(a.T, dz)


def matmul_db(dz, w, b, a):
    return np.matmul(np.ones((1, dz.shape[0])), dz)


def matmul_da(dz, w, b, a):
    return np.matmul(dz, w.T)


op_matmul = matmul_f, (matmul_dw, matmul_db, matmul_da)


def relu_f(z):
    return np.maximum(0, z)


def relu_dz(da, z):
    return da * (z > 0)


op_relu = relu_f, (relu_dz,)


def op_mse(y_val):
    m = y_val.shape[0]

    def mse_f(y_approx):
        return np.sum((y_approx - y_val) ** 2) / m

    def mse_dy_approx(de, y_approx):
        return de * 2 * (y_approx - y_val) / m

    return mse_f, (mse_dy_approx,)


def gen_comp_graph(x_val, y_val, layer_dims):
    cg = CGraph()
    w_vars = {}
    b_vars = {}
    z_vars = {}
    a_vars = {}

    a_vars[0] = cg.new_var()
    ns = [x_val.shape[1]] + layer_dims

    for i in range(1, len(ns)):
        w_vars[i] = cg.new_var()
        b_vars[i] = cg.new_var()
        z_vars[i] = cg.apply(op_matmul, w_vars[i], b_vars[i], a_vars[i - 1])
        a_vars[i] = cg.apply(op_relu, z_vars[i])

    loss = cg.apply(op_mse(y_val), a_vars[len(ns) - 1])

    return cg, loss, w_vars, b_vars, z_vars, a_vars, ns


def test_fit(x_val, y_val, layer_dims, learning_rate=0.001, n_epoch=10000):  # x: (m, n)
    cg, loss, w_vars, b_vars, z_vars, a_vars, ns = gen_comp_graph(x_val, y_val, layer_dims)

    parameters = init_rand_param(w_vars, b_vars, ns, a_vars[0], x_val)
    for i in range(n_epoch):
        fg = cg.forward_full(parameters)
        if (i+1) % 100 == 0:
            print(f'mse[{i+1}] = {fg[-1]}')
        bg = cg.backward_full(fg)
        for p in parameters:
            if p != a_vars[0]:
                parameters[p] = parameters[p] - learning_rate * bg[p]

        # print((fg[-2] - y_val) ** 2)
        # print(fg[-1])
        # bg = cg.backward_full(fg)
        # print(np.concatenate((fg[z_vars[3]], bg[z_vars[3]]), axis=1))
        # print(np.concatenate((fg[a_vars[3]], y_val, bg[a_vars[3]]), axis=1))

    def f_approx(x):
        parameters[a_vars[0]] = x
        forward_graph = cg.forward_full(parameters)
        return forward_graph, cg.backward_full(bg)

    print(f'w_vars={w_vars}')

    return f_approx, loss, w_vars, b_vars, z_vars, a_vars


if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    x = 5 * np.random.rand(100, 2)
    y = np.sum(x * x, axis=1, keepdims=True)
    f_approx, loss, w_vars, b_vars, z_vars, a_vars = test_fit(x, y, [5, 5, 1])
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    fg, bg = f_approx(x)
    print(f'check: {np.concatenate((x, y, fg[-2], fg[-2] - y), axis=1)[:10]}')
    i = 2
    print(f'w{i} = {fg[w_vars[i]]}')
    print(f'dw{i} = {bg[w_vars[i]]}')
