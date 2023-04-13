import numpy as np

from op_lib import op_mse, op_relu, op_matmul


class CompGraphDropout:
    def __init__(self):
        self.n_nodes = 0
        self.inputs = []
        self.consumers = []
        self.operations = []
        self.drop_probs = []
        self.masks = []

    def new_var(self, drop_prob=0):
        u = self.n_nodes
        self.n_nodes += 1

        self.inputs.append([])
        self.consumers.append([])
        self.operations.append(None)
        self.drop_probs.append(drop_prob)
        self.masks.append(1)
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

    def forward_full(self, xs, in_training=True, reuse_mask=False):
        f_graph = [None for _ in range(self.n_nodes)]
        for y in range(self.n_nodes):
            op = self.operations[y]
            if op:
                f, _ = op
                x_values = []
                for x in self.inputs[y]:
                    if not reuse_mask:
                        if in_training and self.drop_probs[x] > 0:
                            self.masks[x] = (1.0 / self.drop_probs[x]) * np.random.uniform(size=f_graph[x].shape) > self.drop_probs[x]
                        else:
                            self.masks[x] = 1.0
                    x_values.append(f_graph[x] * self.masks[x])
                f_graph[y] = f(*x_values)
            else:
                f_graph[y] = xs[y]
        return f_graph

    def forward_partial(self, y, f_graph, in_training=True, reuse_mask=False):
        if f_graph[y] is not None:
            return f_graph

        x_values = []
        for x in self.inputs[y]:
            if f_graph[x] is None:
                self.forward_partial(x, f_graph)
            if not reuse_mask:
                if in_training and self.drop_probs[x] > 0:
                    self.masks[x] = (1.0 / self.drop_probs[x]) * np.random.uniform(size=f_graph[x].shape) > self.drop_probs[x]
                else:
                    self.masks[x] = 1.0
            x_values.append(f_graph[x] * self.masks[x])

        f, _ = self.operations[y]
        y_value = f(*x_values)
        f_graph[y] = y_value
        return f_graph

    def backward_full(self, f_graph):
        b_graph = [None for _ in range(self.n_nodes)]
        b_graph[-1] = 1
        for y in reversed(range(self.n_nodes)):
            xs = self.inputs[y]
            if xs:
                _, df = self.operations[y]
                x_values = [f_graph[x] * self.masks[x] for x in self.inputs[y]]
                for j, x in enumerate(xs):
                    g = df[j](b_graph[y] * self.masks[y], *x_values)
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
            _, df = self.operations[y]
            x_values = [f_graph[x] * self.masks[x] for x in self.inputs[y]]
            g = df[self.inputs[y].index(x)](b_graph[y] * self.masks[y], *x_values)
            if b_graph[x] is None:
                b_graph[x] = g
            else:
                b_graph[x] += g
        return b_graph


def init_rand_param(w_vars, b_vars, ns, a0_var, x_val, w_mult=0.01, b_mult=1):
    parameters = {a0_var: x_val}
    for i in range(1, len(ns)):
        parameters[w_vars[i]] = w_mult * np.random.rand(ns[i - 1], ns[i])
        parameters[b_vars[i]] = b_mult * np.random.rand(1, ns[i])
    return parameters


def generate_fcn_dropout(layer_dims_inc_input, op_cost):
    cg = CompGraphDropout()
    w_vars = {}
    b_vars = {}
    z_vars = {}
    a_vars = {}

    a_vars[0] = cg.new_var()

    for i in range(1, len(layer_dims_inc_input)):
        w_vars[i] = cg.new_var(drop_prob=0.2)
        b_vars[i] = cg.new_var(drop_prob=0.2)
        z_vars[i] = cg.apply(op_matmul, w_vars[i], b_vars[i], a_vars[i - 1])
        a_vars[i] = cg.apply(op_relu, z_vars[i])

    loss = cg.apply(op_cost, a_vars[len(layer_dims_inc_input) - 1])

    return cg, loss, w_vars, b_vars, z_vars, a_vars


def test_fit_dropout(x_val, y_val, layer_dims, learning_rate=0.001, n_epoch=100000):  # x: (m, n)
    layer_dims_inc_input = [x_val.shape[1]] + layer_dims
    cg, loss, w_vars, b_vars, z_vars, a_vars = generate_fcn_dropout(layer_dims_inc_input, op_mse(y_val))
    parameters = init_rand_param(w_vars, b_vars, layer_dims_inc_input, a_vars[0], x_val)
    for i in range(n_epoch):
        fg0 = cg.forward_full(parameters)
        bg = cg.backward_full(fg0)
        for p in parameters:
            if p != a_vars[0]:
                parameters[p] = parameters[p] - learning_rate * bg[p]
        fg1 = cg.forward_full(parameters, reuse_mask=True)

        if (i + 1) % 100 == 0:
            fg_predict = cg.forward_full(parameters, in_training=False)
            print(f'mse[{i + 1}] = {fg_predict[-1]}, last step improvement={fg1[-1] - fg0[-1]}, {fg1[-1] < fg0[-1]}')

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
    f_approx, loss, w_vars, b_vars, z_vars, a_vars = test_fit_dropout(x, y, [10, 10, 1])
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    fg, bg = f_approx(x)
    print(f'check: {np.concatenate((x, y, fg[-2], fg[-2] - y), axis=1)[:10]}')
    i = 2
    print(f'w{i} = {fg[w_vars[i]]}')
    print(f'dw{i} = {bg[w_vars[i]]}')
